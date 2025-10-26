import math
import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means
from weighted_flash_decoding import weighted_flash_decoding

import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# update segment size
THRESHOLD_LENGTH = 1024


class retroinfer_cache(KV_Cache):
    """
    A class representing the KV Cache of RetroInfer.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        n_centroids: int,
        n_segment: int,
        nprobe: int,
        max_compute_cluster_num: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int, 
        use_cluster_estimation: bool = False
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
        self.valid_start = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = cache_unit_size

        self.core = core
        self.dtype = dtype

        self.use_cluster_estimation = use_cluster_estimation

        self.input_length = self.max_length - max_new_length
        self.max_new_length = min(max_new_length-1, THRESHOLD_LENGTH)   # already generated one token when prefilling
        # used for index update, when exceed THRESHOLD_LENGTH, we need to update the index
        self.input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
        self.n_centroids_per_update_segment = THRESHOLD_LENGTH // 16    # default avg 16 vectors per cluster
        self.n_centroids_per_update_segment = (self.n_centroids_per_update_segment // 32) * 32 # must be divisible by 32
        self.n_centroids_new = ((max_new_length-2) // THRESHOLD_LENGTH) * self.n_centroids_per_update_segment  
        if self.input_length_new > 0:
            self.offload_update_keys = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        # store steady zone
        self.steady_zone_keys = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros((self.batch_size, self.kv_head, self.static_pattern_total+self.max_new_length, self.head_dim), 
            dtype=self.dtype, device=self.layer_mapping[str(ldx)]
            ) for ldx in range(self.layer_num)
        ]
        self.static_stride = self.static_pattern_total + self.max_new_length

        # index parameters
        self.n_centroids = n_centroids
        self.n_segment = n_segment
        self.nprobe = nprobe    # retrieve zone size
        self.max_compute_cluster_num = max_compute_cluster_num
        self.es_cluster_num = max_compute_cluster_num - nprobe  # estimation zone size

        # initialize thread pool
        self.thread_pool = ThreadPool(core)
        thread_pool_pointer = self.thread_pool.get()

        # calculate the gpu cache size, buffer size and max total pages for each group
        avg_cluster_size = (self.input_length - self.static_pattern_total) // self.n_centroids
        pages_per_cluster = math.ceil(avg_cluster_size / self.page_size)
        self.cache_size = cache_cluster_num * pages_per_cluster
        # enlarge these values may solve warning and error when decoding
        self.buffer_size = max(int(self.nprobe * 4), 16) * pages_per_cluster

        # whether to pre-allocate GPU buffer and cache before prefilling
        self.allocated = self.pre_allocate_decision()

        # initialize the CPU Wave Buffer
        self.wave_buffer = [WaveBufferCPU(
            self.batch_size, self.kv_head, self.head_dim, self.nprobe, self.page_size, self.n_centroids, 
            self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_size, self.core, thread_pool_pointer)
            for _ in range(self.layer_num)
        ]

        # pin memory indices for hit clusters
        self.hit_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for missing clusters
        self.miss_unit_idices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory indices for cache update clusters
        self.update_buffer_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros((self.batch_size*self.kv_head, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]

        # store compute cluster ids
        self.cI = [
            torch.empty((self.batch_size*self.kv_head, self.max_compute_cluster_num), dtype=torch.int64, pin_memory=True).contiguous(), 
            torch.empty((self.batch_size*self.kv_head, self.max_compute_cluster_num), dtype=torch.int64, pin_memory=True).contiguous()
        ]

        # store searched topk cluster ids
        self.cluster_ids = [
            torch.empty((self.batch_size*self.kv_head, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous(), 
            torch.empty((self.batch_size*self.kv_head, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()
        ]

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx], self.hit_unit_sizes[ldx], self.hit_unit_sizes_cumsum[ldx], self.hit_num_units[ldx],
                self.miss_unit_idices[ldx], self.miss_unit_sizes[ldx], self.miss_unit_sizes_cumsum[ldx], self.miss_num_units[ldx],
                self.update_buffer_indices[ldx], self.update_unit_sizes[ldx], self.update_cache_indices[ldx], self.update_num_units[ldx],
                self.cluster_ids[ldx % 2]
            )

        if self.allocated:
            self.cache_keys = []
            self.cache_values = []
            self.centroids = []
            self.value_sum = []
            self.centroids_mask = []
            self.cluster_size = []
            # allocate GPU Cache data and meta index
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.value_sum.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids_mask.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                                dtype=torch.bool, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cluster_size.append(
                    torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
        else:
            # allocate meta index in CPU
            self.centroids = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.value_sum = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids, self.head_dim), 
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.centroids_mask = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids), 
                            dtype=torch.bool, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]
            self.cluster_size = [
                torch.zeros((self.batch_size*self.kv_head, self.n_centroids),
                            dtype=self.dtype, device="cpu").contiguous()
                for ldx in range(self.layer_num)
            ]

        # layer-share cpu pin buffer, transfer gpu keys & values to cpu for segmented k-means
        self.offload_keys = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()
        self.offload_values = torch.empty(
            (self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
            dtype=self.dtype, pin_memory=True
        ).contiguous()
        
        # allocate pin memory to store organized keys & values in CPU
        self.list_keys = []
        self.list_values = []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim), 
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
            self.list_values.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
        self.list_stride = self.input_length-self.static_pattern_total+self.input_length_new
        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_keys, self.offload_values)

        # whitening
        self.whitened_keys = []
        for _ in range(self.layer_num):
            self.whitened_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )

        # create multi-streams and events
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()

        # statistics tracking for cache hit/miss
        self.cache_stats = {
            'total_hits': [0] * self.layer_num,
            'total_misses': [0] * self.layer_num,
            'total_accesses': [0] * self.layer_num
        }

        # statistics tracking for cluster overlap between consecutive decoding steps
        self.cluster_overlap_stats = {
            'total_overlap': [0] * self.layer_num,
            'total_clusters': [0] * self.layer_num,
            'num_samples': [0] * self.layer_num
        }

        # store previous cluster indices for overlap calculation - one per layer
        self.prev_cluster_ids = [
            torch.empty((self.batch_size*self.kv_head, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.prev_cluster_valid = [False] * self.layer_num  # track if previous clusters are valid for each layer

        # Visualization settings
        self.enable_prefill_visualization = False
        self.viz_save_dir = "plots/prefill_clusters"

        # Store queries from decoding steps for visualization
        self.decode_queries = [[] for _ in range(self.layer_num)]  # List of queries per layer
        self.max_decode_steps_to_visualize = 50  # Limit number of decode steps to store

        # Statistics for query similarity between decoding steps
        self.query_similarity_stats = {
            'cosine_similarities': [[] for _ in range(self.layer_num)],
            'l2_distances': [[] for _ in range(self.layer_num)]
        }

        # Statistics for cluster overlap when using previous query
        self.prev_query_cluster_overlap_stats = {
            'total_overlap': [0] * self.layer_num,
            'total_clusters': [0] * self.layer_num,
            'num_samples': [0] * self.layer_num
        }

        # Store previous queries (as tensors) for cluster selection simulation
        self.prev_queries = [None] * self.layer_num

    def enable_visualization(self, save_dir="plots/prefill_clusters"):
        """Enable visualization of key values and centroids during prefill."""
        self.enable_prefill_visualization = True
        self.viz_save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"Prefill visualization enabled. Plots will be saved to: {save_dir}")

    def plot_query_similarity(self, layer_indices=None):
        """
        Plot query similarity statistics across decoding steps.

        Args:
            layer_indices: list of layer indices to visualize (default: all layers)
        """
        if layer_indices is None:
            layer_indices = range(self.layer_num)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: Cosine similarity over time for selected layers
        ax = axes[0, 0]
        for layer_idx in layer_indices:
            cosine_sims = self.query_similarity_stats['cosine_similarities'][layer_idx]
            if len(cosine_sims) > 0:
                ax.plot(range(len(cosine_sims)), cosine_sims, marker='o',
                       markersize=3, alpha=0.7, label=f'Layer {layer_idx}')
        ax.set_xlabel('Decoding Step')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Query Cosine Similarity Between Consecutive Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: L2 distance over time for selected layers
        ax = axes[0, 1]
        for layer_idx in layer_indices:
            l2_dists = self.query_similarity_stats['l2_distances'][layer_idx]
            if len(l2_dists) > 0:
                ax.plot(range(len(l2_dists)), l2_dists, marker='o',
                       markersize=3, alpha=0.7, label=f'Layer {layer_idx}')
        ax.set_xlabel('Decoding Step')
        ax.set_ylabel('L2 Distance')
        ax.set_title('Query L2 Distance Between Consecutive Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Average cosine similarity per layer
        ax = axes[1, 0]
        avg_cosine_per_layer = []
        for ldx in range(self.layer_num):
            cosine_sims = self.query_similarity_stats['cosine_similarities'][ldx]
            avg_cosine_per_layer.append(np.mean(cosine_sims) if len(cosine_sims) > 0 else 0)
        ax.bar(range(self.layer_num), avg_cosine_per_layer, color='skyblue', edgecolor='navy')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Cosine Similarity')
        ax.set_title('Average Query Cosine Similarity per Layer')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Average L2 distance per layer
        ax = axes[1, 1]
        avg_l2_per_layer = []
        for ldx in range(self.layer_num):
            l2_dists = self.query_similarity_stats['l2_distances'][ldx]
            avg_l2_per_layer.append(np.mean(l2_dists) if len(l2_dists) > 0 else 0)
        ax.bar(range(self.layer_num), avg_l2_per_layer, color='lightcoral', edgecolor='darkred')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average L2 Distance')
        ax.set_title('Average Query L2 Distance per Layer')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = 'query_similarity_stats.png'
        plt.savefig(os.path.join(self.viz_save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved query similarity visualization: {filename}")

    def generate_all_visualizations(self, batch_idx=0, head_idx=0, layer_indices=None):
        """
        Generate all visualizations after decoding is complete.
        This includes queries from decoding steps and query similarity plots.

        Args:
            batch_idx: which batch to visualize (default 0)
            head_idx: which attention head to visualize (default 0)
            layer_indices: list of layer indices to visualize (default: all layers)
        """
        if not self.enable_prefill_visualization:
            print("Visualization is not enabled. Call enable_visualization() first.")
            return

        if layer_indices is None:
            layer_indices = range(self.layer_num)

        print(f"\nGenerating visualizations for {len(layer_indices)} layers...")

        for layer_idx in layer_indices:
            num_queries = len(self.decode_queries[layer_idx])
            print(f"Layer {layer_idx}: {num_queries} decode queries captured")
            self.plot_layer_clusters(layer_idx, batch_idx, head_idx)

        self.plot_all_layers_summary(batch_idx)
        self.plot_query_similarity(layer_indices)

        # Print query similarity statistics
        self.print_query_similarity_stats()

        print(f"\nAll visualizations saved to: {self.viz_save_dir}")

    def plot_all_layer_clusters(self, batch_idx=0, head_idx=0):
        for layer_idx in range(self.layer_num):
            num_queries = len(self.decode_queries[layer_idx])
            print(f"Layer {layer_idx}: {num_queries} decode queries captured")
            self.plot_layer_clusters(layer_idx, batch_idx, head_idx)

    def plot_layer_clusters(self, layer_idx, batch_idx=0, head_idx=0):
        """
        Visualize key values and their centroids for a specific layer after prefill.
        Also plots decode queries if available.

        Args:
            layer_idx: which layer to visualize
            batch_idx: which batch to visualize (default 0)
            head_idx: which attention head to visualize (default 0)
        """
        if not self.enable_prefill_visualization:
            return

        with torch.no_grad():
            # Get centroids for this layer and head
            centroids = self.centroids[layer_idx][batch_idx*self.kv_head + head_idx].cpu().float().numpy()  # [n_centroids, head_dim]
            cluster_size = self.cluster_size[layer_idx][batch_idx*self.kv_head + head_idx].cpu().float().numpy()  # [n_centroids]

            # Get key values for this layer and head
            keys = self.list_keys[layer_idx][batch_idx, head_idx].cpu().float().numpy()  # [num_tokens, head_dim]

            # Get whitened keys for this layer and head
            whitened_keys = self.whitened_keys[layer_idx][batch_idx, head_idx].cpu().float().numpy()  # [num_tokens, head_dim]

            # Get decode queries if available
            queries_list = self.decode_queries[layer_idx]
            has_queries = len(queries_list) > 0

            # Filter out empty clusters
            valid_mask = cluster_size > 0
            valid_centroids = centroids[valid_mask]
            valid_sizes = cluster_size[valid_mask]

            # Use PCA to reduce to 2D for original keys
            pca = PCA(n_components=2)
            all_data = [keys]
            if has_queries:
                queries = np.vstack(queries_list)  # [num_decode_steps, head_dim]
                all_data.append(queries)

            all_data_combined = np.vstack(all_data)
            reduced_all = pca.fit_transform(all_data_combined)

            keys_2d = reduced_all[:len(keys)]
            if has_queries:
                queries_2d = reduced_all[len(keys):]

            # Apply whitening to centroids to get them in whitened space
            # The centroids are already computed on whitened keys, so they're in whitened space
            # Just use them directly
            whitened_centroids = valid_centroids  # Centroids are already in whitened space

            # Use PCA for whitened keys and centroids together
            pca_whitened = PCA(n_components=2)
            whitened_all_data = np.vstack([whitened_keys, whitened_centroids])
            whitened_all_2d = pca_whitened.fit_transform(whitened_all_data)
            whitened_keys_2d = whitened_all_2d[:len(whitened_keys)]
            whitened_centroids_2d = whitened_all_2d[len(whitened_keys):]

            # Create visualization with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))

            # Plot 1: Keys and Decode Queries (no centroids)
            ax = axes[0]
            ax.scatter(keys_2d[:, 0], keys_2d[:, 1], alpha=0.3, s=10, c='lightblue', label='Key Vectors')

            # Plot decode queries as a trajectory
            if has_queries:
                ax.plot(queries_2d[:, 0], queries_2d[:, 1], 'o-', color='orange',
                       markersize=6, linewidth=1.5, alpha=0.8, label='Decode Queries')
                # Add step numbers
                for i, (x, y) in enumerate(queries_2d[::max(1, len(queries_2d)//10)]):  # Label every 10th or fewer
                    ax.annotate(str(i), (x, y), fontsize=8, color='darkred',
                              xytext=(3, 3), textcoords='offset points')

            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            title = f'Layer {layer_idx} Head {head_idx}: Keys'
            if has_queries:
                title += f' + {len(queries_list)} Decode Queries'
            title += f'\n({len(keys)} keys)'
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Whitened Keys with Centroids
            ax = axes[1]
            ax.scatter(whitened_keys_2d[:, 0], whitened_keys_2d[:, 1], alpha=0.3, s=10, c='lightcoral', label='Whitened Keys')
            scatter_w = ax.scatter(whitened_centroids_2d[:, 0], whitened_centroids_2d[:, 1],
                                  s=valid_sizes * 3, c=valid_sizes, cmap='viridis',
                                  alpha=0.7, edgecolors='darkred', linewidth=2, label='Centroids')
            cbar_w = plt.colorbar(scatter_w, ax=ax)
            cbar_w.set_label('Cluster Size', rotation=270, labelpad=20)
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            ax.set_title(f'Layer {layer_idx} Head {head_idx}: Whitened Keys & Centroids\n({len(whitened_keys)} keys, {valid_mask.sum()} clusters)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Cluster size distribution
            ax = axes[2]
            ax.hist(valid_sizes, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.axvline(valid_sizes.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {valid_sizes.mean():.1f}')
            ax.set_xlabel('Cluster Size')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Layer {layer_idx} Head {head_idx}: Cluster Size Distribution\n'
                        f'Total clusters: {len(valid_sizes)}, Total keys: {valid_sizes.sum():.0f}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filename = f'layer_{layer_idx}_batch_{batch_idx}_head_{head_idx}.png'
            plt.savefig(os.path.join(self.viz_save_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved visualization: {filename}")

    def plot_all_layers_summary(self, batch_idx=0):
        """
        Create summary plots showing cluster statistics across all layers.

        Args:
            batch_idx: which batch to visualize (default 0)
        """
        if not self.enable_prefill_visualization:
            return

        with torch.no_grad():
            # Collect statistics for all layers
            avg_cluster_sizes = []
            num_empty_clusters = []
            num_valid_clusters = []

            for layer_idx in range(self.layer_num):
                cluster_size = self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head].cpu().numpy()

                # Average across all heads
                valid_sizes = cluster_size[cluster_size > 0]
                avg_cluster_sizes.append(valid_sizes.mean() if len(valid_sizes) > 0 else 0)
                num_empty_clusters.append((cluster_size == 0).sum())
                num_valid_clusters.append((cluster_size > 0).sum())

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Average cluster size per layer
            ax = axes[0, 0]
            ax.bar(range(self.layer_num), avg_cluster_sizes, color='skyblue', edgecolor='navy')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Average Cluster Size')
            ax.set_title('Average Cluster Size per Layer')
            ax.grid(True, alpha=0.3, axis='y')

            # Plot 2: Valid vs empty clusters
            ax = axes[0, 1]
            width = 0.35
            x = np.arange(self.layer_num)
            ax.bar(x - width/2, num_valid_clusters, width, label='Valid', color='green', alpha=0.7)
            ax.bar(x + width/2, num_empty_clusters, width, label='Empty', color='red', alpha=0.7)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Valid vs Empty Clusters per Layer')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Plot 3: Heatmap of cluster sizes
            ax = axes[1, 0]
            cluster_data = []
            for layer_idx in range(self.layer_num):
                cluster_size = self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head].cpu().numpy()
                cluster_data.append(cluster_size.mean(axis=0))  # Average across heads

            cluster_data = np.array(cluster_data)
            im = ax.imshow(cluster_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel('Centroid Index')
            ax.set_ylabel('Layer Index')
            ax.set_title(f'Cluster Size Heatmap Across Layers\n(averaged across {self.kv_head} heads)')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cluster Size', rotation=270, labelpad=20)

            # Plot 4: Cluster size variance across layers
            ax = axes[1, 1]
            variances = []
            for layer_idx in range(self.layer_num):
                cluster_size = self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head].cpu().numpy()
                valid_sizes = cluster_size[cluster_size > 0]
                variances.append(valid_sizes.std() if len(valid_sizes) > 0 else 0)

            ax.plot(range(self.layer_num), variances, marker='o', linewidth=2, markersize=8, color='purple')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Cluster Size Std Dev')
            ax.set_title('Cluster Size Variance per Layer')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f'all_layers_summary_batch_{batch_idx}.png'
            plt.savefig(os.path.join(self.viz_save_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved summary visualization: {filename}")

    # decide whether to pre-allocate GPU memory before prefilling
    def pre_allocate_decision(self):
        # estimate the KV Cache GPU memory consumption
        self.esitimate_gpu_memory = 2 * self.layer_num * self.batch_size * self.kv_head * (self.cache_size*self.page_size + self.n_centroids + self.static_pattern_total + self.max_new_length) * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.buffer_size * self.page_size * self.head_dim * 2
        self.esitimate_gpu_memory += 2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        self.esitimate_gpu_memory += 4 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        self.esitimate_gpu_memory /= 1024 * 1024 * 1024
        # print(f"Estimate KV Cache GPU memory consumption: {self.esitimate_gpu_memory:.4f} GB")

        return self.free_memory > self.esitimate_gpu_memory*1.5
    
    # allocate layer-share buffer for computation
    def allocate_computation_buffer(self):
        # execution buffer to store keys & values used to compute attention, shared across layers
        self.execution_buffer_keys = [torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                 dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous(), 
                                      torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim),
                                                 dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()]
        self.execution_buffer_values = [torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim), 
                                                   dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous(), 
                                        torch.zeros((self.batch_size*self.kv_head, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim),
                                                   dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()]
        self.valid_lengths = [torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, 
                                          device=self.layer_mapping[str(0)]).contiguous(), 
                              torch.zeros((self.batch_size*self.kv_head), dtype=torch.int32, 
                                          device=self.layer_mapping[str(0)]).contiguous()]
        self.execution_stride = self.buffer_size * self.page_size + self.static_stride
        
        # allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        
        # allocate layer-share buffer for estimation zone
        self.es_centroids = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                        dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_value_sum = torch.zeros((self.batch_size*self.kv_head, self.es_cluster_num, 1, self.head_dim),
                                         dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()
        self.es_cluster_size = torch.zeros((self.batch_size*self.kv_head, 1, 1, self.es_cluster_num),
                                           dtype=self.dtype, device=self.layer_mapping[str(0)]).contiguous()

    def prepare_cache(self):
        # sync the last batch of the last layer
        torch.cuda.synchronize()
        self.wave_buffer[self.layer_num-1].construction_sync()

        # clear temp memory
        self.clusters_cpu = None
        self.cluster_size_cpu = None
        self.temp_keys = None
        self.temp_values = None

        if not self.allocated:  # allocate GPU memory after prefilling
            self.cache_keys = []
            self.cache_values = []
            for ldx in range(self.layer_num):
                # allocate GPU Cache data
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_size, self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                # move meta index to gpu
                self.centroids[ldx] = self.centroids[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.value_sum[ldx] = self.value_sum[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.centroids_mask[ldx] = self.centroids_mask[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
                self.cluster_size[ldx] = self.cluster_size[ldx].to(self.layer_mapping[str(ldx)]).contiguous()
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
    

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx, embed_keys_whitened=None): 
        """
        Prefill update the key & value cache for per batch for per layer
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            batch_idx: batch index
        """    
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == 1, f"Multi-batch prefilling only support prefill single batch one by one."
        assert seq_len <= self.input_length, f"seq_len({seq_len}) should less than input_length({self.input_length})"
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start[batch_idx]
        valid_length = seq_len - self.static_pattern_total - valid_start

        # sync for the previous layer and batch finish organize pages
        if layer_idx > 0:
            self.wave_buffer[layer_idx-1].construction_sync()
        elif batch_idx > 0: # layer_idx == 0
            self.wave_buffer[self.layer_num-1].construction_sync()
        
        # store in self to avoid deleting when async offload to cpu, shape: (group_num, seq_len, dim)
        self.temp_keys = key_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.temp_values = value_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.mainevents[self.layer_mapping[str(layer_idx)]].record()

        self.temp_whitened_keys = embed_keys_whitened[0, :, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :].contiguous()
        self.whitened_keys[layer_idx][batch_idx, :, :valid_length, :].copy_(embed_keys_whitened[0, :, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :], non_blocking=True)

        # async offload keys & values to cpu
        with torch.cuda.stream(self.copystream):
            self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
            self.offload_keys[:, :valid_length, :].copy_(self.temp_keys, non_blocking=True)
            self.offload_values[:, :valid_length, :].copy_(self.temp_values, non_blocking=True)
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        # copy steady zone to pre-allocated memory
        self.steady_zone_keys[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            key_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_keys[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            key_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, :self.static_pattern_start, :] = \
            value_states[0, valid_start:valid_start+self.static_pattern_start, :, :].transpose(0, 1)
        self.steady_zone_values[layer_idx][batch_idx, :, self.static_pattern_start:self.static_pattern_total, :] = \
            value_states[0, seq_len-self.static_pattern_end:seq_len, :, :].transpose(0, 1)

        # compute key mean, shape (group_num, 1, head_dim)
        mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

        # segmented k-means
        _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
            key=self.temp_keys-mean_key,    # centering to 0
            value=self.temp_values,
            num_centroids=self.n_centroids,
            num_segments=self.n_segment,
        )

        # whitened segmented k-means
        # _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
        #     key=self.temp_whitened_keys,
        #     value=self.temp_values,
        #     num_centroids=self.n_centroids,
        #     num_segments=self.n_segment,
        # )

        # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids
        # print (_cluster_size)

        # copy meta index
        self.centroids[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_centroids + mean_key)         # (group_num, n_centroids, dim)
        # self.centroids[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_centroids)
        self.value_sum[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_value_sum)                    # (group_num, n_centroids, dim)
        self.centroids_mask[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size == 0)          # (group_num, n_centroids)
        self.cluster_size[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :].copy_(_cluster_size.to(self.dtype))  # (group_num, n_centroids)

        # these data will be used to organize the cpu kv
        self.cluster_size_cpu = _cluster_size.cpu().contiguous()    # (group_num, n_centroids)
        self.clusters_cpu = _clusters.cpu().contiguous()            # (group_num, n_centroids, max_cluster_size)
        
        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += seq_len
        
        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]   # ignore mask tokens, shape: (bsz, seq_len, group_num, dim)

    def sync(
        self,
        layer_idx,
        batch_idx
    ):
        """
        wait async offloading on copystream -> organize kv
        """
        # wait for offload finish
        self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
        # async organize kv
        self.wave_buffer[layer_idx].async_construction(
            self.clusters_cpu,      # (group_num, n_centroids, max_cluster_size)
            self.cluster_size_cpu,  # (group_num, n_centroids)
            batch_idx
        )


    # update KV cache when generate tokens exceed THRESHOLD_LENGTH
    def _update_kv_cache(self):
        for ldx in range(self.layer_num):
            torch.cuda.set_device(self.layer_mapping[str(ldx)])
            update_keys = self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            update_values = self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_size*self.kv_head, THRESHOLD_LENGTH, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(ldx)]].record()

            # move local window
            self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_keys[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]
            self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_values[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]

            # async offload
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(ldx)]].wait()
                self.offload_update_keys.copy_(update_keys, non_blocking=True)
                self.offload_update_values.copy_(update_values, non_blocking=True)
                self.copyevents[self.layer_mapping[str(ldx)]].record()
            
            # compute key mean, shape (batch_size*group_num, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)
            
            # segmented k-means
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys-mean_key,   # centering to 0, (batch_size*group_num, THRESHOLD_LENGTH, dim)
                value=update_values,        # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                num_centroids=self.n_centroids_per_update_segment,
                num_segments=1,
            )
            _centroids += mean_key
            assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids_per_update_segment

            # append to meta index
            self.centroids[ldx] = torch.cat((self.centroids[ldx], _centroids), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.value_sum[ldx] = torch.cat((self.value_sum[ldx], _value_sum), dim=1)  # (batch_szie*group_num, new_n_centroids, dim)
            self.centroids_mask[ldx] = torch.cat((self.centroids_mask[ldx], _cluster_size == 0), dim=1) # (batch_szie*group_num, new_n_centroids)
            self.cluster_size[ldx] = torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1) # (batch_szie*group_num, new_n_centroids)
            assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.n_centroids_per_update_segment

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,           # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                self.offload_update_values,         # (batch_size*group_num, THRESHOLD_LENGTH, dim)
                _clusters.cpu().contiguous(),       # (batch_size*group_num, n_centroids_per_update_segment, max_cluster_size)
                _cluster_size.cpu().contiguous()    # (batch_size*group_num, n_centroids_per_update_segment)
            )
        torch.cuda.set_device(self.layer_mapping[str(0)])
        
        # update n_centroids
        self.n_centroids += self.n_centroids_per_update_segment
        # re-allocate layer-share buffer for batch_gemm_softmax kernel
        self.gemm_o = torch.zeros((self.batch_size, self.kv_head, self.group_size, self.n_centroids), 
                                  device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.softmax_o = torch.zeros((self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                                     device=self.layer_mapping[str(0)], dtype=self.dtype).contiguous()
        self.norm = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                 device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        self.sum = torch.zeros((self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                                device=self.layer_mapping[str(0)], dtype=torch.float32).contiguous()
        # reset static pattern
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end


    def decode_update_kv_cache(self,
        key_states,         # (bs, length(=1), group_num, dim)
        value_states,       # (bs, length(=1), group_num, dim)
        layer_idx
    ):
        # index update
        if self.static_pattern_total == self.static_pattern_start + self.static_pattern_end + THRESHOLD_LENGTH:
            # print("Updating KV cache ...")
            self._update_kv_cache()
            # print("KV cache updated, continue decoding ...")

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None   # no use the return value
    

    def update_cache_stats(self, layer_idx):
        """
        Update cache hit/miss statistics for a specific layer
        """
        # Sum up hits and misses across all batch groups
        total_hits = torch.sum(self.hit_num_units[layer_idx]).item()
        total_misses = torch.sum(self.miss_num_units[layer_idx]).item()

        self.cache_stats['total_hits'][layer_idx] += total_hits
        self.cache_stats['total_misses'][layer_idx] += total_misses
        self.cache_stats['total_accesses'][layer_idx] += total_hits + total_misses

    def update_prev_query_cluster_overlap_stats(self, layer_idx, buffer_idx):
        """
        Calculate cluster overlap if we used the previous query instead of current query.
        This shows how many clusters would overlap if we reused the previous query.

        Args:
            layer_idx: current layer index
            buffer_idx: buffer index for current layer
        """
        with torch.no_grad():
            # Use previous query to select clusters
            prev_query = self.prev_queries[layer_idx]

            # Calculate distances with previous query
            batch_gemm_softmax(prev_query, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                            self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                            self.RSQRT_DIM, 0)
            dist_prev = torch.sum(self.softmax_o, dim=1)  # [batch_size*group_num, n_centroids]
            dist_prev.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)

            # Get top-k clusters using previous query
            prev_query_cluster_ids = torch.topk(dist_prev, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1]

            # Compare with actual clusters selected by current query
            # current_cluster_ids = self.cluster_ids[buffer_idx]
            current_cluster_ids = self.cI[buffer_idx]

            # Calculate overlap
            total_overlap = 0
            for i in range(self.batch_size * self.kv_head):
                current_set = set(current_cluster_ids[i].cpu().tolist())
                prev_query_set = set(prev_query_cluster_ids[i].cpu().tolist())
                overlap = len(current_set & prev_query_set)
                total_overlap += overlap

            self.prev_query_cluster_overlap_stats['total_overlap'][layer_idx] += total_overlap
            self.prev_query_cluster_overlap_stats['total_clusters'][layer_idx] += self.batch_size * self.kv_head * self.max_compute_cluster_num
            self.prev_query_cluster_overlap_stats['num_samples'][layer_idx] += 1

    def update_cluster_overlap_stats(self, layer_idx):
        """
        Update cluster overlap statistics by comparing current cluster indices
        with previous decoding step for a specific layer
        """
        buffer_idx = layer_idx % 2

        # Only calculate overlap if we have valid previous clusters for this layer
        if self.prev_cluster_valid[layer_idx]:
            # Calculate overlap for each batch group
            current_clusters = self.cluster_ids[buffer_idx].cpu()
            prev_clusters = self.prev_cluster_ids[layer_idx]

            total_overlap = 0
            for i in range(self.batch_size * self.kv_head):
                # Convert to sets and calculate intersection
                current_set = set(current_clusters[i].tolist())
                prev_set = set(prev_clusters[i].tolist())
                overlap = len(current_set & prev_set)
                total_overlap += overlap

            self.cluster_overlap_stats['total_overlap'][layer_idx] += total_overlap
            self.cluster_overlap_stats['total_clusters'][layer_idx] += self.batch_size * self.kv_head * self.nprobe
            self.cluster_overlap_stats['num_samples'][layer_idx] += 1

        # Store current clusters as previous for next iteration of this layer
        self.prev_cluster_ids[layer_idx].copy_(self.cluster_ids[buffer_idx].cpu())
        self.prev_cluster_valid[layer_idx] = True

    def get_cache_stats(self, reset=False):
        """
        Get cache statistics for all layers
        Args:
            reset: if True, reset statistics after retrieval
        Returns:
            Dictionary with per-layer statistics including hit rate
        """
        stats = {}
        for ldx in range(self.layer_num):
            total_accesses = self.cache_stats['total_accesses'][ldx]
            total_hits = self.cache_stats['total_hits'][ldx]
            total_misses = self.cache_stats['total_misses'][ldx]

            hit_rate = (total_hits / total_accesses * 100) if total_accesses > 0 else 0.0

            stats[f'layer_{ldx}'] = {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_accesses': total_accesses,
                'hit_rate': hit_rate
            }

        if reset:
            self.cache_stats = {
                'total_hits': [0] * self.layer_num,
                'total_misses': [0] * self.layer_num,
                'total_accesses': [0] * self.layer_num
            }

        return stats

    def get_cluster_overlap_stats(self, reset=False):
        """
        Get cluster overlap statistics for all layers
        Args:
            reset: if True, reset statistics after retrieval
        Returns:
            Dictionary with per-layer cluster overlap statistics
        """
        stats = {}
        for ldx in range(self.layer_num):
            total_overlap = self.cluster_overlap_stats['total_overlap'][ldx]
            total_clusters = self.cluster_overlap_stats['total_clusters'][ldx]
            num_samples = self.cluster_overlap_stats['num_samples'][ldx]

            overlap_rate = (total_overlap / total_clusters * 100) if total_clusters > 0 else 0.0
            avg_overlap_per_sample = (total_overlap / num_samples) if num_samples > 0 else 0.0

            stats[f'layer_{ldx}'] = {
                'total_overlap': total_overlap,
                'total_clusters': total_clusters,
                'num_samples': num_samples,
                'overlap_rate': overlap_rate,
                'avg_overlap_per_sample': avg_overlap_per_sample
            }

        if reset:
            self.cluster_overlap_stats = {
                'total_overlap': [0] * self.layer_num,
                'total_clusters': [0] * self.layer_num,
                'num_samples': [0] * self.layer_num
            }

        return stats

    def print_cache_stats(self, reset=False):
        """
        Print cache statistics in a formatted table
        Args:
            reset: if True, reset statistics after printing
        """
        stats = self.get_cache_stats(reset=False)

        # print("\n" + "="*80)
        # print("Cache Hit/Miss Statistics")
        # print("="*80)
        # print(f"{'Layer':<10} {'Hits':<15} {'Misses':<15} {'Accesses':<15} {'Hit Rate':<15}")
        # print("-"*80)

        # for ldx in range(self.layer_num):
        #     layer_stats = stats[f'layer_{ldx}']
        #     print(f"{ldx:<10} {layer_stats['total_hits']:<15} {layer_stats['total_misses']:<15} "
        #           f"{layer_stats['total_accesses']:<15} {layer_stats['hit_rate']:<14.2f}%")

        # print("="*80 + "\n")
        # print average hit rate
        total_hits = sum(stats[f'layer_{ldx}']['total_hits'] for ldx in range(self.layer_num))
        total_misses = sum(stats[f'layer_{ldx}']['total_misses'] for ldx in range(self.layer_num))
        total_accesses = sum(stats[f'layer_{ldx}']['total_accesses'] for ldx in range(self.layer_num))
        avg_hit_rate = (total_hits / total_accesses * 100) if total_accesses > 0 else 0.0
        print(f"Average Hit Rate across all layers: {avg_hit_rate:.2f}% ({total_hits} hits, {total_misses} misses, {total_accesses} accesses)") 

        if reset:
            self.cache_stats = {
                'total_hits': [0] * self.layer_num,
                'total_misses': [0] * self.layer_num,
                'total_accesses': [0] * self.layer_num
            }

    def get_prev_query_cluster_overlap_stats(self, reset=False):
        """
        Get cluster overlap statistics when using previous query
        Args:
            reset: if True, reset statistics after retrieval
        Returns:
            Dictionary with per-layer cluster overlap statistics
        """
        stats = {}
        for ldx in range(self.layer_num):
            total_overlap = self.prev_query_cluster_overlap_stats['total_overlap'][ldx]
            total_clusters = self.prev_query_cluster_overlap_stats['total_clusters'][ldx]
            num_samples = self.prev_query_cluster_overlap_stats['num_samples'][ldx]

            overlap_rate = (total_overlap / total_clusters * 100) if total_clusters > 0 else 0.0
            avg_overlap_per_sample = (total_overlap / num_samples) if num_samples > 0 else 0.0

            stats[f'layer_{ldx}'] = {
                'total_overlap': total_overlap,
                'total_clusters': total_clusters,
                'num_samples': num_samples,
                'overlap_rate': overlap_rate,
                'avg_overlap_per_sample': avg_overlap_per_sample
            }

        if reset:
            self.prev_query_cluster_overlap_stats = {
                'total_overlap': [0] * self.layer_num,
                'total_clusters': [0] * self.layer_num,
                'num_samples': [0] * self.layer_num
            }

        return stats

    def print_prev_query_cluster_overlap_stats(self, reset=False):
        """
        Print cluster overlap statistics when using previous query
        Args:
            reset: if True, reset statistics after printing
        """
        import matplotlib.pyplot as plt

        stats = self.get_prev_query_cluster_overlap_stats(reset=False)

        print("\n" + "="*100)
        print("Cluster Overlap Statistics (Using Previous Query)")
        print("="*100)
        print(f"{'Layer':<10} {'Samples':<12} {'Avg Overlap':<20} {'Overlap Rate':<20} {'nprobe':<10}")
        print("-"*100)

        layers = []
        overlap_rates = []

        for ldx in range(self.layer_num):
            layer_stats = stats[f'layer_{ldx}']
            if layer_stats['num_samples'] > 0:
                print(f"{ldx:<10} {layer_stats['num_samples']:<12} "
                      f"{layer_stats['avg_overlap_per_sample']:<20.2f} "
                      f"{layer_stats['overlap_rate']:<19.2f}% "
                      f"{self.max_compute_cluster_num:<10}")
                layers.append(ldx)
                overlap_rates.append(layer_stats['overlap_rate'])

        print("="*100 + "\n")

        # Calculate average across all layers
        total_overlap = sum(self.prev_query_cluster_overlap_stats['total_overlap'])
        total_clusters = sum(self.prev_query_cluster_overlap_stats['total_clusters'])
        total_samples = sum(self.prev_query_cluster_overlap_stats['num_samples'])

        if total_clusters > 0:
            avg_overlap_rate = (total_overlap / total_clusters * 100)
            avg_overlap_per_sample = (total_overlap / total_samples) if total_samples > 0 else 0
            print(f"Average across all layers:")
            print(f"  Overlap Rate: {avg_overlap_rate:.2f}%")
            print(f"  Avg Overlap per Sample: {avg_overlap_per_sample:.2f} / {self.max_compute_cluster_num}\n")

        # Plot overlap rate
        if layers:
            plt.figure(figsize=(10, 6))
            plt.plot(layers, overlap_rates, marker='o', linewidth=2, markersize=6)
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Overlap Rate (%)', fontsize=12)
            plt.title('Cluster Overlap Rate by Layer (Using Previous Query)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/prev_query_cluster_overlap_rate.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: prev_query_cluster_overlap_rate.png\n")

        if reset:
            self.prev_query_cluster_overlap_stats = {
                'total_overlap': [0] * self.layer_num,
                'total_clusters': [0] * self.layer_num,
                'num_samples': [0] * self.layer_num
            }

    def print_cluster_overlap_stats(self, reset=False):
        """
        Print cluster overlap statistics in a formatted table
        Args:
            reset: if True, reset statistics after printing
        """
        stats = self.get_cluster_overlap_stats(reset=False)

        print("\n" + "="*90)
        print("Cluster Overlap Statistics (Consecutive Decoding Steps)")
        print("="*90)
        print(f"{'Layer':<10} {'Samples':<12} {'Avg Overlap':<15} {'Overlap Rate':<15} {'nprobe':<10}")
        print("-"*90)

        for ldx in range(self.layer_num):
            layer_stats = stats[f'layer_{ldx}']
            print(f"{ldx:<10} {layer_stats['num_samples']:<12} "
                  f"{layer_stats['avg_overlap_per_sample']:<15.2f} "
                  f"{layer_stats['overlap_rate']:<14.2f}% "
                  f"{self.nprobe:<10}")

        print("="*90 + "\n")

        if reset:
            self.cluster_overlap_stats = {
                'total_overlap': [0] * self.layer_num,
                'total_clusters': [0] * self.layer_num,
                'num_samples': [0] * self.layer_num
            }

    def print_all_stats(self, reset=False):
        """
        Print all statistics: cache, cluster overlap, and query similarity
        Args:
            reset: if True, reset statistics after printing
        """
        self.print_cache_stats(reset=False)
        self.print_cluster_overlap_stats(reset=False)
        self.print_prev_query_cluster_overlap_stats(reset=False)
        self.print_query_similarity_stats(reset=reset)

    def get_query_similarity_stats(self, reset=False):
        """
        Get query similarity statistics between consecutive decoding steps
        Args:
            reset: if True, reset statistics after retrieval
        Returns:
            Dictionary with per-layer query similarity statistics
        """
        stats = {}
        for ldx in range(self.layer_num):
            cosine_sims = self.query_similarity_stats['cosine_similarities'][ldx]
            l2_dists = self.query_similarity_stats['l2_distances'][ldx]

            if len(cosine_sims) > 0:
                avg_cosine = np.mean(cosine_sims)
                std_cosine = np.std(cosine_sims)
                min_cosine = np.min(cosine_sims)
                max_cosine = np.max(cosine_sims)
            else:
                avg_cosine = std_cosine = min_cosine = max_cosine = 0.0

            if len(l2_dists) > 0:
                avg_l2 = np.mean(l2_dists)
                std_l2 = np.std(l2_dists)
                min_l2 = np.min(l2_dists)
                max_l2 = np.max(l2_dists)
            else:
                avg_l2 = std_l2 = min_l2 = max_l2 = 0.0

            stats[f'layer_{ldx}'] = {
                'num_steps': len(cosine_sims),
                'cosine_similarity': {
                    'mean': avg_cosine,
                    'std': std_cosine,
                    'min': min_cosine,
                    'max': max_cosine,
                },
                'l2_distance': {
                    'mean': avg_l2,
                    'std': std_l2,
                    'min': min_l2,
                    'max': max_l2,
                }
            }

        if reset:
            self.query_similarity_stats = {
                'cosine_similarities': [[] for _ in range(self.layer_num)],
                'l2_distances': [[] for _ in range(self.layer_num)]
            }

        return stats

    def print_query_similarity_stats(self, reset=False):
        """
        Print query similarity statistics in a formatted table
        Args:
            reset: if True, reset statistics after printing
        """
        import matplotlib.pyplot as plt

        stats = self.get_query_similarity_stats(reset=False)

        print("\n" + "="*100)
        print("Query Similarity Statistics (Consecutive Decoding Steps)")
        print("="*100)
        print(f"{'Layer':<8} {'Steps':<8} {'Cosine Sim (avg±std)':<25} {'Cosine Range':<20} {'L2 Dist (avg±std)':<25}")
        print("-"*100)

        layers = []
        cosine_means = []
        cosine_stds = []
        l2_means = []
        l2_stds = []

        for ldx in range(self.layer_num):
            layer_stats = stats[f'layer_{ldx}']
            if layer_stats['num_steps'] > 0:
                cos_stats = layer_stats['cosine_similarity']
                l2_stats = layer_stats['l2_distance']

                print(f"{ldx:<8} {layer_stats['num_steps']:<8} "
                      f"{cos_stats['mean']:.4f}±{cos_stats['std']:.4f}        "
                      f"[{cos_stats['min']:.4f}, {cos_stats['max']:.4f}]      "
                      f"{l2_stats['mean']:.4f}±{l2_stats['std']:.4f}")

                layers.append(ldx)
                cosine_means.append(cos_stats['mean'])
                cosine_stds.append(cos_stats['std'])
                l2_means.append(l2_stats['mean'])
                l2_stds.append(l2_stats['std'])

        print("="*100 + "\n")

        # Print average across all layers
        all_cosine = []
        all_l2 = []
        for ldx in range(self.layer_num):
            all_cosine.extend(self.query_similarity_stats['cosine_similarities'][ldx])
            all_l2.extend(self.query_similarity_stats['l2_distances'][ldx])

        if len(all_cosine) > 0:
            print(f"Average across all layers:")
            print(f"  Cosine Similarity: {np.mean(all_cosine):.4f}±{np.std(all_cosine):.4f}")
            print(f"  L2 Distance: {np.mean(all_l2):.4f}±{np.std(all_l2):.4f}\n")

        # Plot similarities
        if layers:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Cosine similarity plot
            ax1.errorbar(layers, cosine_means, yerr=cosine_stds, marker='o', linewidth=2,
                        markersize=6, capsize=5, capthick=2)
            ax1.set_xlabel('Layer', fontsize=12)
            ax1.set_ylabel('Cosine Similarity', fontsize=12)
            ax1.set_title('Query Cosine Similarity by Layer', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # L2 distance plot
            ax2.errorbar(layers, l2_means, yerr=l2_stds, marker='s', linewidth=2,
                        markersize=6, capsize=5, capthick=2, color='orange')
            ax2.set_xlabel('Layer', fontsize=12)
            ax2.set_ylabel('L2 Distance', fontsize=12)
            ax2.set_title('Query L2 Distance by Layer', fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('query_similarity_stats.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved to: query_similarity_stats.png\n")

        if reset:
            self.query_similarity_stats = {
                'cosine_similarities': [[] for _ in range(self.layer_num)],
                'l2_distances': [[] for _ in range(self.layer_num)]
            }

    def compute(self, queries, layer_idx, queries_next=None, embed_queries=None):
        """
        queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
        """
        # assert queries.size(0) == self.batch_size
        # assert queries.size(1) == 1
        # assert queries.size(2) == self.kv_head * self.group_size == self.num_heads
        # assert queries.size(3) == self.head_dim

        # Calculate query similarity between consecutive decoding steps
        with torch.no_grad():
            # queries shape: [batch_size, 1, num_heads, head_dim]
            # Take first batch, first head (head 0) for similarity calculation
            query_vector = queries[0, 0, 0, :].cpu()  # [head_dim]

            # Calculate similarity with previous query if available
            if len(self.decode_queries[layer_idx]) > 0:
                prev_query = torch.from_numpy(self.decode_queries[layer_idx][-1])

                # Cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(
                    query_vector.unsqueeze(0),
                    prev_query.unsqueeze(0)
                ).item()
                self.query_similarity_stats['cosine_similarities'][layer_idx].append(cosine_sim)

                # L2 distance
                l2_dist = torch.norm(query_vector - prev_query, p=2).item()
                self.query_similarity_stats['l2_distances'][layer_idx].append(l2_dist)

            # Store query for visualization if enabled
            if self.enable_prefill_visualization and len(self.decode_queries[layer_idx]) < self.max_decode_steps_to_visualize:
                self.decode_queries[layer_idx].append(query_vector.float().numpy())

        torch.cuda.nvtx.range_push("kv_cache_compute")
        static_len = self.static_pattern_total if layer_idx == self.layer_num - 1 else self.static_pattern_total + 1

        buffer_idx = layer_idx % 2
        next_buffer_idx = (layer_idx + 1) % 2

        # first two layers search current layer clusters
        if layer_idx < 2 or not self.use_cluster_estimation:
            torch.cuda.nvtx.range_push("search_topk_clusters")
            # search for TopK centroids
            # start = time.perf_counter()
            batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                            self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                            self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
            # batch_gemm_softmax(embed_queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
            #                 self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
            #                 self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
            dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
            dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
            self.cI[buffer_idx] = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
            self.cluster_ids[buffer_idx].copy_(self.cI[buffer_idx][..., :self.nprobe])

            # Calculate cluster overlap if we had used previous query
            if self.prev_queries[layer_idx] is not None:
                self.update_prev_query_cluster_overlap_stats(layer_idx, buffer_idx)

            # Update cluster overlap statistics (consecutive steps)
            self.update_cluster_overlap_stats(layer_idx)
            # print ("layer ", layer_idx, "selected clusters:", self.cluster_ids[layer_idx])
            # end = time.perf_counter()
            # print (f"layer {layer_idx} select clusters: {(end-start) * 1000:.4f} ms")
            torch.cuda.nvtx.range_pop()

            # Store current query for next step's comparison
            self.prev_queries[layer_idx] = queries.clone()

        # cache access and submit cache update tasks to thread pool
        if layer_idx < 2 or not self.use_cluster_estimation:
            torch.cuda.nvtx.range_push("current_layer_access")
            self.wave_buffer[layer_idx].batch_access()
            torch.cuda.nvtx.range_pop()

        # estimate next layer clusters
        if self.use_cluster_estimation and layer_idx > 0 and layer_idx < self.layer_num - 1 and queries_next is not None:
            torch.cuda.nvtx.range_push("search_next_topk_clusters")
            # search for TopK centroids
            # start = time.perf_counter()
            batch_gemm_softmax(queries_next, self.centroids[layer_idx + 1], self.gemm_o, self.norm, self.sum, self.softmax_o,
                            self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                            self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
            dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
            dist.masked_fill_(self.centroids_mask[layer_idx + 1], self.DTYPE_MIN)
            self.cI[next_buffer_idx] = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
            self.cluster_ids[next_buffer_idx].copy_(self.cI[next_buffer_idx][..., :self.nprobe])
            # Update cluster overlap statistics for next layer
            self.update_cluster_overlap_stats(layer_idx + 1)
            # end = time.perf_counter()
            # print (f"layer {layer_idx + 1} estimate clusters: {(end-start) * 1000:.4f} ms")
            torch.cuda.nvtx.range_pop()

        # cache access and submit cache update tasks to thread pool
        # no estimated clusters
        # start = time.perf_counter()
        # estimated clusters
        if self.use_cluster_estimation and layer_idx > 1 and layer_idx < self.layer_num - 1 and queries_next is not None:
            torch.cuda.nvtx.range_push("next_layer_access")
            self.wave_buffer[layer_idx + 1].batch_access()
            torch.cuda.nvtx.range_pop()
        # end = time.perf_counter()
        # print (f"layer {layer_idx} wave_buffer access time: {(end-start) * 1000:.4f} ms")

        # estimation zone computation
        if self.es_cluster_num > 0:
            torch.cuda.nvtx.range_push("estimation_zone")
            # start = time.perf_counter()
            gather_copy_vectors(self.centroids[layer_idx], self.es_centroids, 
                                self.value_sum[layer_idx], self.es_value_sum, 
                                self.cluster_size[layer_idx], self.es_cluster_size,
                                self.cI[buffer_idx], self.batch_groups, self.n_centroids, self.es_cluster_num, 
                                self.max_compute_cluster_num, self.nprobe, self.es_cluster_num)
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # print (f"layer {layer_idx} gather estimate centroids: {(end-start) * 1000:.4f} ms")

            # start = time.perf_counter()
            es_out, es_lse = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                self.es_centroids,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_value_sum,       # [batch_size*group_num, es_cluster, 1, dim]
                self.es_cluster_size,    # [batch_size*group_num, 1, 1, es_cluster]
                previous_out=None, previous_lse=None,
                return_softmax_lse=True)
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # print (f"layer {layer_idx} estimation zone flash attention: {(end-start) * 1000:.4f} ms")

            torch.cuda.nvtx.range_pop()
        else:
            es_out, es_lse = None, None

        # no estimation, current layer access sync
        if layer_idx < 2 or not self.use_cluster_estimation:
            self.wave_buffer[layer_idx].sync()
            self.wave_buffer[layer_idx].batch_update()
            # Update statistics after cache access
            self.update_cache_stats(layer_idx)
        # estimation, next layer access sync
        if self.use_cluster_estimation and layer_idx > 1 and layer_idx < self.layer_num - 1:
            self.wave_buffer[layer_idx + 1].sync()
            self.wave_buffer[layer_idx].batch_update()
            # Update statistics after cache access
            self.update_cache_stats(layer_idx)

        # assemble the execution buffer
        # start = time.perf_counter()
        # current layer copy
        if layer_idx < 2 or not self.use_cluster_estimation:
            # print (self.list_keys[layer_idx].device, self.cache_keys[layer_idx].device, self.execution_buffer_keys.device)
            # print ("hit ", torch.sum (self.hit_unit_sizes[layer_idx], dim=1))
            # print ("miss ", torch.sum (self.miss_unit_sizes[layer_idx], dim=1))
            
            torch.cuda.nvtx.range_push("current_layer_copy")
            # start = time.perf_counter()
            gather_copy_and_concat(self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys[buffer_idx],
                                self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values[buffer_idx],
                                self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                                self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                                self.valid_lengths[buffer_idx], self.batch_groups, 
                                self.static_stride, self.list_stride, self.cache_stride,
                                self.execution_stride, self.buffer_size, static_len)
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # print (f"layer {layer_idx} gather_copy_and_concat time: {(end-start) * 1000:.4f} ms")
            torch.cuda.nvtx.range_pop()
        # next layer copy
        if self.use_cluster_estimation and layer_idx > 1 and layer_idx < self.layer_num - 1 and queries_next is not None:
            torch.cuda.nvtx.range_push("next_layer_copy")
            with torch.cuda.stream(self.copystream):
                gather_copy_and_concat(self.steady_zone_keys[layer_idx + 1], self.list_keys[layer_idx + 1], self.cache_keys[layer_idx + 1], self.execution_buffer_keys[next_buffer_idx],
                                    self.steady_zone_values[layer_idx + 1], self.list_values[layer_idx + 1], self.cache_values[layer_idx + 1], self.execution_buffer_values[next_buffer_idx],
                                    self.miss_unit_idices[layer_idx + 1], self.miss_unit_sizes[layer_idx + 1], self.miss_unit_sizes_cumsum[layer_idx + 1], self.miss_num_units[layer_idx + 1],
                                    self.hit_unit_idices[layer_idx + 1], self.hit_unit_sizes[layer_idx + 1], self.hit_unit_sizes_cumsum[layer_idx + 1], self.hit_num_units[layer_idx + 1],
                                    self.valid_lengths[next_buffer_idx], self.batch_groups, 
                                    self.static_stride, self.list_stride, self.cache_stride,
                                    self.execution_stride, self.buffer_size, static_len)
            torch.cuda.nvtx.range_pop()
        # torch.cuda.synchronize()
        # end = time.perf_counter()
        # print (f"gather_copy_and_concat time: {(end-start) * 1000:.4f} ms")

        if self.use_cluster_estimation and layer_idx == 1:
            torch.cuda.nvtx.range_push("next_layer_access")
            self.wave_buffer[layer_idx + 1].batch_access()
            torch.cuda.nvtx.range_pop()

        # flash attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        torch.cuda.nvtx.range_push("flash_attention")
        # start = time.perf_counter()
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys[buffer_idx],    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values[buffer_idx],  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths[buffer_idx],
            return_softmax_lse=False
        )
        # torch.cuda.synchronize()
        # end = time.perf_counter()
        # print (f"layer {layer_idx} flash attention time: {(end-start) * 1000:.4f} ms")
        torch.cuda.nvtx.range_pop()

        # admiss pages from execution buffer to GPU cache
        # start = time.perf_counter()
        # update sync
        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        # end = time.perf_counter()
        # print (f"layer {layer_idx} wave_buffer sync time: {(end-start) * 1000:.4f} ms")

        # start = time.perf_counter()
        with torch.cuda.stream(self.copystream):
            gather_copy_and_scatter(self.execution_buffer_keys[buffer_idx], self.cache_keys[layer_idx], self.execution_buffer_values[buffer_idx], self.cache_values[layer_idx],
                                    self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], self.update_cache_indices[layer_idx], 
                                    self.update_num_units[layer_idx], self.batch_groups, self.execution_stride, self.cache_stride,
                                    self.buffer_size, static_len)
        # torch.cuda.synchronize()
        # end = time.perf_counter()
        # print (f"gather copy and scatter {(end-start) * 1000:.4f} ms")

        # layer 1 access and copy
        if self.use_cluster_estimation and layer_idx == 1:
            self.wave_buffer[layer_idx + 1].sync()
            # Update statistics for layer 2 when using cluster estimation at layer 1
            self.update_cache_stats(layer_idx + 1)

            with torch.cuda.stream(self.copystream):
                gather_copy_and_concat(self.steady_zone_keys[layer_idx + 1], self.list_keys[layer_idx + 1], self.cache_keys[layer_idx + 1], self.execution_buffer_keys[next_buffer_idx],
                                    self.steady_zone_values[layer_idx + 1], self.list_values[layer_idx + 1], self.cache_values[layer_idx + 1], self.execution_buffer_values[next_buffer_idx],
                                    self.miss_unit_idices[layer_idx + 1], self.miss_unit_sizes[layer_idx + 1], self.miss_unit_sizes_cumsum[layer_idx + 1], self.miss_num_units[layer_idx + 1],
                                    self.hit_unit_idices[layer_idx + 1], self.hit_unit_sizes[layer_idx + 1], self.hit_unit_sizes_cumsum[layer_idx + 1], self.hit_num_units[layer_idx + 1],
                                    self.valid_lengths[next_buffer_idx], self.batch_groups,
                                    self.static_stride, self.list_stride, self.cache_stride,
                                    self.execution_stride, self.buffer_size, static_len)

        torch.cuda.nvtx.range_pop()

        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
