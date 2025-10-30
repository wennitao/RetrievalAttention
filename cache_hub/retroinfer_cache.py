import math
import torch
import numpy as np
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means, balanced_k_means, balanced_k_means_v2, sklearn_balanced_k_means, page_partition
from weighted_flash_decoding import weighted_flash_decoding
import flashinfer

import time
import matplotlib.pyplot as plt

from .profiling import *

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
        use_cluster_estimation: bool = False, 
        use_cache: bool = True
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
        self.use_cache = use_cache

        self.input_length = self.max_length - max_new_length
        self.max_new_length = max(max_new_length-1, THRESHOLD_LENGTH)   # already generated one token when prefilling
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

        print (self.static_pattern_total + self.max_new_length)
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

        # Test keys and values in GPU memory
        # self.offload_keys = torch.empty(
        #     (self.batch_size*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
        #     dtype=self.dtype, device="cuda:0"
        # )
        # self.offload_values = torch.empty(
        #     (self.batch_size*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
        #     dtype=self.dtype, device="cuda:0"
        # )

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

        # test kv cache in gpu memory
        self.device_list_key = torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim), dtype=self.dtype, device="cuda:0").contiguous()
        self.device_list_value = torch.empty((self.batch_size, self.kv_head, self.input_length-self.static_pattern_total+self.input_length_new, self.head_dim), dtype=self.dtype, device="cuda:0").contiguous()

        # create multi-streams and events
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()
    
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

    def plot_cluster_contiguity(self, clusters, layer_idx, save_dir="cluster_plots"):
        """
        Plot cluster assignments for all heads:
        1. Transition heatmap (green=contiguous, red=transition)
        2. Cluster span histogram (max_index - min_index per cluster)

        Args:
            clusters: cluster assignment tensor, shape (group_num, n_centroids, max_cluster_size)
            layer_idx: layer index for labeling
            save_dir: directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        num_heads = clusters.shape[0]
        n_centroids, max_cluster_size = clusters.shape[1], clusters.shape[2]

        # Create two figures
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols

        # Figure 1: Transition heatmaps
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if num_heads == 1:
            axes1 = np.array([axes1])
        axes1 = axes1.flatten()

        # Figure 2: Span histograms
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if num_heads == 1:
            axes2 = np.array([axes2])
        axes2 = axes2.flatten()

        for head_idx in range(num_heads):
            # Get assignments for this head: (n_centroids, max_cluster_size)
            head_clusters = clusters[head_idx].cpu().numpy()

            # Calculate cluster spans (max_idx - min_idx for each cluster)
            cluster_spans = []
            for cluster_id in range(n_centroids):
                valid_indices = head_clusters[cluster_id][head_clusters[cluster_id] > 0]
                if len(valid_indices) > 0:
                    span = int(valid_indices.max()) - int(valid_indices.min())
                    cluster_spans.append(span)

            # Plot span histogram
            if len(cluster_spans) > 0:
                axes2[head_idx].hist(cluster_spans, bins=50, edgecolor='black', alpha=0.7)
                axes2[head_idx].set_xlabel('Cluster Span (max_idx - min_idx)', fontsize=9)
                axes2[head_idx].set_ylabel('Count', fontsize=9)
                axes2[head_idx].set_title(f'Head {head_idx}\nMean: {np.mean(cluster_spans):.1f}, Median: {np.median(cluster_spans):.1f}', fontsize=10)
                axes2[head_idx].grid(True, alpha=0.3)
            else:
                axes2[head_idx].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                    transform=axes2[head_idx].transAxes)
                axes2[head_idx].set_title(f'Head {head_idx}', fontsize=11)

            # Create inverse mapping: index -> cluster_id
            max_idx = -1
            for cluster_id in range(n_centroids):
                for idx in head_clusters[cluster_id]:
                    if idx >= 0:
                        max_idx = max(max_idx, int(idx))

            if max_idx < 0:
                axes1[head_idx].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                                   transform=axes1[head_idx].transAxes)
                axes1[head_idx].set_title(f'Head {head_idx}', fontsize=11)
                continue

            # Create index->cluster mapping
            index_to_cluster = np.full(max_idx + 1, -1, dtype=np.int32)
            for cluster_id in range(n_centroids):
                for idx in head_clusters[cluster_id]:
                    if idx >= 0:
                        index_to_cluster[int(idx)] = cluster_id

            # Calculate cluster changes
            cluster_change = np.zeros_like(index_to_cluster, dtype=np.float32)
            for i in range(1, len(index_to_cluster)):
                if index_to_cluster[i] >= 0 and index_to_cluster[i-1] >= 0:
                    cluster_change[i] = 1.0 if index_to_cluster[i] != index_to_cluster[i-1] else 0.0
                else:
                    cluster_change[i] = -1  # invalid/padding

            # Reshape into 2D for heatmap
            chunk_size = 100
            n_chunks = (len(cluster_change) + chunk_size - 1) // chunk_size
            padded_size = n_chunks * chunk_size
            padded = np.pad(cluster_change, (0, padded_size - len(cluster_change)),
                           constant_values=-1)
            reshaped = padded.reshape(n_chunks, chunk_size)
            masked_data = np.ma.masked_where(reshaped == -1, reshaped)

            # Plot transition heatmap
            im = axes1[head_idx].imshow(masked_data, aspect='auto', cmap='RdYlGn_r',
                                       interpolation='nearest', vmin=0, vmax=1)
            axes1[head_idx].set_xlabel('Index within chunk', fontsize=9)
            axes1[head_idx].set_ylabel('Chunk ID', fontsize=9)
            axes1[head_idx].set_title(f'Head {head_idx}', fontsize=11)

        # Hide unused subplots
        for idx in range(num_heads, len(axes1)):
            axes1[idx].axis('off')
        for idx in range(num_heads, len(axes2)):
            axes2[idx].axis('off')

        # Finalize and save figures
        plt.figure(fig1.number)
        plt.suptitle(f'Layer {layer_idx} - Cluster Transition Map (All Heads)\nGreen = contiguous, Red = transition',
                     fontsize=14, y=0.995)
        plt.tight_layout()
        save_path1 = os.path.join(save_dir, f"cluster_transitions_layer{layer_idx}.png")
        plt.savefig(save_path1, dpi=150, bbox_inches='tight')
        print(f"Cluster transition map saved to {save_path1}")
        plt.close(fig1)

        plt.figure(fig2.number)
        plt.suptitle(f'Layer {layer_idx} - Cluster Span Distribution (All Heads)\nLower span = more contiguous',
                     fontsize=14, y=0.995)
        plt.tight_layout()
        save_path2 = os.path.join(save_dir, f"cluster_spans_layer{layer_idx}.png")
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        print(f"Cluster span histogram saved to {save_path2}")
        plt.close(fig2)


    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx): 
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

        # residual_len = valid_length % 16
        # self.static_pattern_end += residual_len
        # self.static_pattern_total += residual_len
        # valid_length -= residual_len
        
        # store in self to avoid deleting when async offload to cpu, shape: (group_num, seq_len, dim)
        self.temp_keys = key_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.temp_values = value_states[0, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(0, 1).contiguous()
        self.mainevents[self.layer_mapping[str(layer_idx)]].record()

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

        # print (self.temp_keys.shape)

        # cuvs balanced kmeans
        # _centroids, _value_sum, _clusters, _cluster_size = balanced_k_means(
        #     key=self.temp_keys-mean_key,    # centering to 0
        #     value=self.temp_values,
        #     num_centroids=valid_length // 16,
        #     buffer_num_centroids=self.n_centroids,
        # )
        # print (_cluster_size)

        # Sinkhorn balanced kmeans
        # _centroids, _value_sum, _clusters, _cluster_size = balanced_k_means_v2(
        #     key=self.temp_keys-mean_key,    # centering to 0
        #     value=self.temp_values,
        #     num_centroids=valid_length // 16,
        #     buffer_num_centroids=self.n_centroids,
        #     num_iters=10,
        # )
        # print (_centroids)
        # print (_cluster_size.shape)
        # print (_cluster_size)

        # linear sum assignment
        # _centroids, _value_sum, _clusters, _cluster_size = sklearn_balanced_k_means(
        #     key=self.temp_keys-mean_key,    # centering to 0
        #     value=self.temp_values,
        #     num_centroids=valid_length // 16,
        #     buffer_num_centroids=self.n_centroids,
        # )

        # page parition
        # _centroids, _value_sum, _clusters, _cluster_size = page_partition(
        #     key=self.temp_keys-mean_key,    # centering to 0
        #     value=self.temp_values,
        #     page_size=16, 
        #     buffer_num_centroids=self.n_centroids,
        # )
        # print (_centroids.shape, _value_sum.shape, _clusters.shape, _cluster_size.shape)
        # print (_centroids.dtype, _value_sum.dtype, _clusters.dtype, _cluster_size.dtype)

        # segmented k-means
        # centroids: (group_num, n_centroids, dim)
        # value_sum: (group_num, n_centroids, dim)
        # clusters: (group_num, n_centroids, max_cluster_size)
        # cluster_size: (group_num, n_centroids)
        _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
            key=self.temp_keys-mean_key,    # centering to 0
            value=self.temp_values,
            num_centroids=self.n_centroids,
            num_segments=self.n_segment,
        )
        assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids
        # print (_centroids.shape, _value_sum.shape, _cluster_size.shape, _clusters.shape)
        # print (_cluster_size)
        # print (_clusters)

        # save cluster assignment
        # np.savetxt(f"cluster_data/qa/layer{layer_idx}_cluster.txt", _clusters.view(-1, _clusters.shape[-1]).cpu().numpy(), fmt='%d')

        # plot cluster contiguity for all heads
        # self.plot_cluster_contiguity(_clusters, layer_idx, save_dir="cluster_plots/qa")

        # copy meta index
        self.centroids[layer_idx][batch_idx*self.kv_head:(batch_idx+1)*self.kv_head, :, :].copy_(_centroids + mean_key)         # (group_num, n_centroids, dim)
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
    

    def compute(self, queries, layer_idx, queries_next=None):
        """
        queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
        """
        # assert queries.size(0) == self.batch_size
        # assert queries.size(1) == 1
        # assert queries.size(2) == self.kv_head * self.group_size == self.num_heads
        # assert queries.size(3) == self.head_dim

        torch.cuda.nvtx.range_push("kv_cache_compute")
        static_len = self.static_pattern_total if layer_idx == self.layer_num - 1 else self.static_pattern_total + 1

        buffer_idx = layer_idx % 2
        next_buffer_idx = (layer_idx + 1) % 2

        # first two layers search current layer clusters
        if layer_idx < 2 or not self.use_cluster_estimation:
            torch.cuda.nvtx.range_push("search_topk_clusters")
            # search for TopK centroids
            start = time.perf_counter()
            batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                            self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                            self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
            dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
            dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
            self.cI[buffer_idx] = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
            self.cluster_ids[buffer_idx].copy_(self.cI[buffer_idx][..., :self.nprobe])
            # print ("layer ", layer_idx, "selected clusters:", self.cluster_ids[layer_idx])
            torch.cuda.synchronize()
            end = time.perf_counter()
            # print (f"layer {layer_idx} select clusters: {(end-start) * 1000:.4f} ms")
            select_clusters_time.append((end-start) * 1000)
            torch.cuda.nvtx.range_pop()

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
            start = time.perf_counter()
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
            torch.cuda.synchronize()
            end = time.perf_counter()
            # print (f"layer {layer_idx} estimation zone flash attention: {(end-start) * 1000:.4f} ms")
            estimation_time.append((end-start) * 1000)
            torch.cuda.nvtx.range_pop()
        else:
            es_out, es_lse = None, None

        # no estimation, current layer access sync
        if layer_idx < 2 or not self.use_cluster_estimation:
            start = time.perf_counter()
            self.wave_buffer[layer_idx].sync()
            end = time.perf_counter()
            buffer_access_time.append((end-start) * 1000)
            if self.use_cache:
                self.wave_buffer[layer_idx].batch_update()
        # estimation, next layer access sync
        if self.use_cluster_estimation and layer_idx > 1 and layer_idx < self.layer_num - 1:
            self.wave_buffer[layer_idx + 1].sync()
            self.wave_buffer[layer_idx].batch_update()

        start = time.perf_counter()
        self.device_list_key.copy_(self.list_keys[layer_idx])
        self.device_list_value.copy_(self.list_values[layer_idx])
        torch.cuda.synchronize()
        end = time.perf_counter()
        kv_copy_time.append((end-start) * 1000)

        # assemble the execution buffer
        start = time.perf_counter()
        # current layer copy
        if layer_idx < 2 or not self.use_cluster_estimation:
            # print (self.list_keys[layer_idx].device, self.cache_keys[layer_idx].device, self.execution_buffer_keys.device)
            # print ("hit ", torch.sum (self.hit_unit_sizes[layer_idx], dim=1))
            # print ("miss ", torch.sum (self.miss_unit_sizes[layer_idx], dim=1))
            torch.cuda.nvtx.range_push("current_layer_copy")
            gather_copy_and_concat(self.steady_zone_keys[layer_idx], self.device_list_key, self.cache_keys[layer_idx], self.execution_buffer_keys[buffer_idx],
                                self.steady_zone_values[layer_idx], self.device_list_value, self.cache_values[layer_idx], self.execution_buffer_values[buffer_idx],
                                self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                                self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                                self.valid_lengths[buffer_idx], self.batch_groups, 
                                self.static_stride, self.list_stride, self.cache_stride,
                                self.execution_stride, self.buffer_size, static_len)
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
        torch.cuda.synchronize()
        end = time.perf_counter()
        # print (f"layer {layer_idx} gather_copy_and_concat time: {(end-start) * 1000:.4f} ms")
        gather_time.append((end-start) * 1000)

        if self.use_cluster_estimation and layer_idx == 1:
            torch.cuda.nvtx.range_push("next_layer_access")
            self.wave_buffer[layer_idx + 1].batch_access()
            torch.cuda.nvtx.range_pop()

        # flash attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        torch.cuda.nvtx.range_push("flash_attention")
        start = time.perf_counter()
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys[buffer_idx],    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values[buffer_idx],  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths[buffer_idx],
            return_softmax_lse=False
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        # print (f"layer {layer_idx} flash attention time: {(end-start) * 1000:.4f} ms")
        flash_attn_time.append((end-start) * 1000)
        torch.cuda.nvtx.range_pop()

        if self.use_cache:
            # admiss pages from execution buffer to GPU cache
            start = time.perf_counter()
            # update sync
            self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
            end = time.perf_counter()
            # print (f"layer {layer_idx} wave_buffer sync time: {(end-start) * 1000:.4f} ms")
            buffer_update_time.append((end-start) * 1000)

            start = time.perf_counter()
            with torch.cuda.stream(self.copystream):
                gather_copy_and_scatter(self.execution_buffer_keys[buffer_idx], self.cache_keys[layer_idx], self.execution_buffer_values[buffer_idx], self.cache_values[layer_idx],
                                        self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx], self.update_cache_indices[layer_idx], 
                                        self.update_num_units[layer_idx], self.batch_groups, self.execution_stride, self.cache_stride,
                                        self.buffer_size, static_len)
            torch.cuda.synchronize()
            end = time.perf_counter()
            # print (f"gather copy and scatter {(end-start) * 1000:.4f} ms")
            cache_update_time.append((end-start) * 1000)

            # layer 1 access and copy
            if self.use_cluster_estimation and layer_idx == 1:
                self.wave_buffer[layer_idx + 1].sync()

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

    def init_flashinfer (self):
        self.workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.decode_scatter_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        self.paged_cache_size = self.batch_size * self.kv_head * self.nprobe * 8
        self.scattered_cache_size = self.batch_size * self.kv_head * self.nprobe * 8

        self.paged_key_cache = []
        self.paged_value_cache = []
        self.scattered_key_cache = []
        self.scattered_value_cache = []
        
        for ldx in range(self.layer_num):
            self.paged_key_cache.append(
                torch.zeros((self.paged_cache_size, self.page_size, 1, self.head_dim),
                            dtype=self.dtype, device="cuda:0").contiguous()
            )
            self.paged_value_cache.append(
                torch.zeros((self.paged_cache_size, self.page_size, 1, self.head_dim),
                            dtype=self.dtype, device="cuda:0").contiguous()
            )
            self.scattered_key_cache.append(
                torch.zeros((self.scattered_cache_size, 1, 1, self.head_dim),
                            dtype=self.dtype, device="cuda:0").contiguous()
            )
            self.scattered_value_cache.append(
                torch.zeros((self.scattered_cache_size, 1, 1, self.head_dim),
                            dtype=self.dtype, device="cuda:0").contiguous()
            )

        selected_pages = int(self.nprobe / (self.n_centroids * 0.018) * 175)
        selected_scatter_tokens = int(self.nprobe / (self.n_centroids * 0.018) * 478)

        self.kv_page_indptr = torch.tensor (
            [idx * selected_pages for idx in range (self.batch_size * self.kv_head + 1)], dtype=torch.int32
        )
        self.kv_page_indices = torch.randperm(self.paged_cache_size, dtype=torch.int32)[:self.batch_size * self.kv_head * selected_pages]
        self.kv_last_page_len = torch.full((self.batch_size * self.kv_head, ), self.page_size, dtype=torch.int32)

        self.kv_scatter_indptr = torch.tensor (
            [idx * selected_scatter_tokens for idx in range (self.batch_size * self.kv_head + 1)], dtype=torch.int32
        )
        self.kv_scatter_indices = torch.randperm(self.scattered_cache_size, dtype=torch.int32)[:self.batch_size * self.kv_head * selected_scatter_tokens]
        self.kv_last_scatter_len = torch.full((self.batch_size * self.kv_head, ), 1, dtype=torch.int32)

    def compute_flashinfer(self, queries, layer_idx):
        static_len = self.static_pattern_total if layer_idx == self.layer_num - 1 else self.static_pattern_total + 1

        buffer_idx = layer_idx % 2

        torch.cuda.nvtx.range_push("search_topk_clusters")
        # search for TopK centroids
        start = time.perf_counter()
        batch_gemm_softmax(queries, self.centroids[layer_idx], self.gemm_o, self.norm, self.sum, self.softmax_o,
                        self.batch_groups, self.group_size, self.n_centroids, self.head_dim,
                        self.RSQRT_DIM, 0)       # [batch_size*group_num, group_size, n_centroids]
        dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
        dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
        self.cI[buffer_idx] = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
        self.cluster_ids[buffer_idx].copy_(self.cI[buffer_idx][..., :self.nprobe])
        # print ("layer ", layer_idx, "selected clusters:", self.cluster_ids[layer_idx])
        torch.cuda.synchronize()
        end = time.perf_counter()
        # print (f"layer {layer_idx} select clusters: {(end-start) * 1000:.4f} ms")
        select_clusters_time.append((end-start) * 1000)
        torch.cuda.nvtx.range_pop()

        self.wave_buffer[layer_idx].batch_access()

        # estimation zone computation
        if self.es_cluster_num > 0:
            torch.cuda.nvtx.range_push("estimation_zone")
            start = time.perf_counter()
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
            torch.cuda.synchronize()
            end = time.perf_counter()
            # print (f"layer {layer_idx} estimation zone flash attention: {(end-start) * 1000:.4f} ms")
            estimation_time.append((end-start) * 1000)
            torch.cuda.nvtx.range_pop()
        else:
            es_out, es_lse = None, None

        self.wave_buffer[layer_idx].sync()

        # flashinfer decode
        start = time.perf_counter()
        self.decode_wrapper.plan (
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_last_page_len,
            4,
            1,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        self.decode_scatter_wrapper.plan (
            self.kv_scatter_indptr,
            self.kv_scatter_indices,
            self.kv_last_scatter_len,
            4,
            1,
            self.head_dim,
            1,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        end = time.perf_counter()
        plan_time.append((end-start) * 1000)

        queries = queries.view(self.batch_size * self.kv_head, 4, self.head_dim)

        start = time.perf_counter()
        attn_out, lse_out = self.decode_wrapper.run(
            queries, 
            (self.paged_key_cache[layer_idx], self.paged_value_cache[layer_idx]),
            return_lse=True
        )
        attn_scatter_out, lse_scatter_out = self.decode_scatter_wrapper.run(
            queries, 
            (self.scattered_key_cache[layer_idx], self.scattered_value_cache[layer_idx]),
            return_lse=True
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        flash_attn_time.append((end-start) * 1000)
        start = time.perf_counter()
        es_out = es_out.view (self.batch_size, self.num_heads, self.head_dim)
        es_lse = es_lse.view (self.batch_size, self.num_heads)
        attn_out = attn_out.view (self.batch_size, self.num_heads, self.head_dim)
        lse_out = lse_out.view (self.batch_size, self.num_heads)
        attn_scatter_out = attn_scatter_out.view (self.batch_size, self.num_heads, self.head_dim)
        lse_scatter_out = lse_scatter_out.view (self.batch_size, self.num_heads)
        flashinfer.cascade.merge_state_in_place(attn_out, lse_out, es_out, es_lse)
        flashinfer.cascade.merge_state_in_place(attn_out, lse_out, attn_scatter_out, lse_scatter_out)
        torch.cuda.synchronize()
        end = time.perf_counter()
        merge_time.append((end-start) * 1000)

        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)