import math
import os
import torch
import numpy as np
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax

from .cache import KV_Cache
from .kmeans import segment_k_means, balanced_k_means, balanced_k_means_v2, sklearn_balanced_k_means, page_partition
from weighted_flash_decoding import weighted_flash_decoding

import time

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
        rope_cos_sin_cache: torch.Tensor = None,
        enable_rope_correction: bool = True
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
        
        # Store RoPE cos/sin cache for query smoothing
        self.rope_cos_sin_cache = rope_cos_sin_cache
        self.enable_rope_correction = enable_rope_correction
        
        # AR(1)/AR(2) prediction parameters
        self.enable_ar_prediction = os.getenv("ENABLE_AR_PREDICTION", "0") == "1"
        self.ar_alpha = float(os.getenv("AR_ALPHA", "1.0"))  # weight for current query
        self.ar_beta = float(os.getenv("AR_BETA", "0.1"))    # weight for momentum term
        # Optional AR(2) term - additional momentum on previous delta
        self.ar_order = int(os.getenv("AR_ORDER", "1"))
        self.ar_gamma = float(os.getenv("AR_GAMMA", "0.0"))  # second-order momentum term
        # Lightweight on-the-fly ridge calibration for beta on neutral deltas
        self.calibrate_ar = os.getenv("CALIBRATE_AR", "0") == "1"
        self.ar_calib_window = int(os.getenv("AR_CALIB_WINDOW", "64"))
        self.ar_calib_l2 = float(os.getenv("AR_CALIB_L2", "1e-3"))
        self.ar_beta_min = float(os.getenv("AR_BETA_MIN", "0.0"))
        self.ar_beta_max = float(os.getenv("AR_BETA_MAX", "0.3"))
        # Per-layer calibration state
        self.prev_delta_neutral = [None for _ in range(self.layer_num)]
        self.ar_calib_num = [0.0 for _ in range(self.layer_num)]  # numerator accumulator
        self.ar_calib_den = [0.0 for _ in range(self.layer_num)]  # denominator accumulator
        self.ar_calib_steps = [0 for _ in range(self.layer_num)]
        self.ar_calibrated = [False for _ in range(self.layer_num)]
        
        # Store previous queries for AR prediction (rotation-neutral space)
        self.prev_queries_neutral = [None for _ in range(self.layer_num)]
        self.prev_prev_queries_neutral = [None for _ in range(self.layer_num)]  # for AR(2)

        # MLP predictor (residual) for neutral-space query prediction
        self.enable_mlp_prediction = os.getenv("ENABLE_MLP_PREDICTION", "0") == "1"
        self.mlp_weights_path = os.getenv("MLP_WEIGHTS_PATH", "")
        self.mlp_input_mode = os.getenv("MLP_INPUT_MODE", "concat_delta")
        self.mlp_apply_to_sim_only = os.getenv("MLP_APPLY_TO_SIM_ONLY", "1") == "1"
        self.mlp_models = [None for _ in range(self.layer_num)]  # per-layer MLP
        if self.enable_mlp_prediction and self.mlp_weights_path:
            self._load_mlp_models()
        
        # Sample collection for MLP training
        self.collect_samples = os.getenv("COLLECT_NEUTRAL_SAMPLES", "0") == "1"
        self.sample_collector = None
        if self.collect_samples:
            try:
                from tools.collect_neutral_samples import NeutralSampleCollector
                max_samples = int(os.getenv("MAX_SAMPLES_PER_LAYER", "50000"))
                self.sample_collector = NeutralSampleCollector(self.layer_num, self.head_dim, max_samples)
            except ImportError:
                print("Warning: Could not import NeutralSampleCollector; sample collection disabled.")
                self.collect_samples = False

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

        # create multi-streams and events
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()

        # query similarity instrumentation (optional, no behavior change)
        # Enable by setting environment variable QUERY_SIM_LOG=1
        self.sim_log_enabled = os.getenv("QUERY_SIM_LOG", "0") == "1"
        base_logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        # Summary CSV only: per-token-pair, per-layer, mean/min/max/std over the H heads
        self.sim_summary_path = os.getenv("QUERY_SIM_SUMMARY_PATH", os.path.join(base_logs_dir, "query_sim_qhead_summary.csv"))
        self.prev_queries = [None for _ in range(self.layer_num)]  # store previous step queries per layer
        self.layer_step = [0 for _ in range(self.layer_num)]       # per-layer decode step counter
        if self.sim_log_enabled:
            log_dir = os.path.dirname(self.sim_summary_path)
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Prepare summary header
                if not os.path.exists(self.sim_summary_path):
                    with open(self.sim_summary_path, "w") as fsum:
                        fsum.write("token_pair,layer,mean,min,max,std\n")
                else:
                    # If an older header exists, reset to the new header to avoid mixed formats
                    try:
                        with open(self.sim_summary_path, "r") as fsum:
                            first = fsum.readline().strip()
                        if first != "token_pair,layer,mean,min,max,std":
                            with open(self.sim_summary_path, "w") as fsumw:
                                fsumw.write("token_pair,layer,mean,min,max,std\n")
                    except Exception:
                        # If any issue reading, reset file to new header
                        try:
                            with open(self.sim_summary_path, "w") as fsumw:
                                fsumw.write("token_pair,layer,mean,min,max,std\n")
                        except Exception:
                            pass
            except Exception:
                # If logging path can't be created, silently disable to avoid impacting inference
                self.sim_log_enabled = False
    
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

        residual_len = valid_length % 16
        self.static_pattern_end += residual_len
        self.static_pattern_total += residual_len
        valid_length -= residual_len
        
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
        _centroids, _value_sum, _clusters, _cluster_size = page_partition(
            key=self.temp_keys-mean_key,    # centering to 0
            value=self.temp_values,
            page_size=16, 
            buffer_num_centroids=self.n_centroids,
        )
        # print (_centroids.shape, _value_sum.shape, _clusters.shape, _cluster_size.shape)
        # print (_centroids.dtype, _value_sum.dtype, _clusters.dtype, _cluster_size.dtype)

        # segmented k-means
        # centroids: (group_num, n_centroids, dim)
        # value_sum: (group_num, n_centroids, dim)
        # clusters: (group_num, n_centroids, max_cluster_size)
        # cluster_size: (group_num, n_centroids)
        # _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
        #     key=self.temp_keys-mean_key,    # centering to 0
        #     value=self.temp_values,
        #     num_centroids=self.n_centroids,
        #     num_segments=self.n_segment,
        # )
        assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids
        # print (_centroids.shape, _value_sum.shape, _cluster_size.shape, _clusters.shape)
        # print (_cluster_size)
        # print (_clusters)

        # save cluster assignment
        # np.savetxt(f"cluster_data/layer{layer_idx}_cluster.txt", _clusters.view(-1, _clusters.shape[-1]).cpu().numpy(), fmt='%d')

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
        
        # Return key/value states to satisfy caller's expected interface
        return key_states, value_states

    
    def _load_mlp_models(self):
        """Load per-layer MLP models from checkpoint."""
        import torch.nn as nn
        
        class ResidualMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                if input_dim != output_dim:
                    self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
                else:
                    self.residual_proj = nn.Identity()
            
            def forward(self, x):
                residual = self.residual_proj(x)
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out + residual
        
        # Try loading per-layer or global model
        for layer_idx in range(self.layer_num):
            # Check for per-layer checkpoint
            layer_path = self.mlp_weights_path.replace('.pt', f'_layer{layer_idx}.pt')
            if os.path.exists(layer_path):
                ckpt = torch.load(layer_path, map_location='cpu')
            elif os.path.exists(self.mlp_weights_path):
                # Fall back to global model
                ckpt = torch.load(self.mlp_weights_path, map_location='cpu')
            else:
                continue
            
            # Build model
            model = ResidualMLP(ckpt['input_dim'], ckpt['hidden_dim'], ckpt['output_dim'])
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            # Move to layer device
            model = model.to(self.layer_mapping[str(layer_idx)], dtype=self.dtype)
            self.mlp_models[layer_idx] = model
            print(f"Loaded MLP for layer {layer_idx} from {layer_path if os.path.exists(layer_path) else self.mlp_weights_path}")
    
    def apply_rope_inverse(self, queries, position):
        """
        Apply inverse RoPE rotation R(-θ) to move queries to rotation-neutral space.
        queries: [B, H, D]
        position: scalar position index
        Returns: queries in rotation-neutral space [B, H, D]
        """
        if self.rope_cos_sin_cache is None or position >= self.rope_cos_sin_cache.shape[0]:
            return queries
        
        B, H, D = queries.shape
        half_d = D // 2
        
        # Get cos/sin for the position (move the single row to the queries device/dtype)
        cos_sin = self.rope_cos_sin_cache[position]
        cos_sin = cos_sin.to(queries.device, dtype=queries.dtype, non_blocking=True)
        cos_theta, sin_theta = cos_sin[:half_d], cos_sin[half_d:]
        
        # Reshape for rotation
        q_reshape = queries.view(B, H, half_d, 2)
        x1, x2 = q_reshape[..., 0], q_reshape[..., 1]
        
        # Apply R(-θ) = [[cos, sin], [-sin, cos]]
        x1_neutral = cos_theta * x1 + sin_theta * x2
        x2_neutral = -sin_theta * x1 + cos_theta * x2
        
        return torch.stack([x1_neutral, x2_neutral], dim=-1).view(B, H, D)
    
    
    def apply_rope_forward(self, queries, position):
        """
        Apply forward RoPE rotation R(θ) from rotation-neutral space.
        queries: [B, H, D] in rotation-neutral space
        position: scalar position index
        Returns: queries with RoPE applied [B, H, D]
        """
        if self.rope_cos_sin_cache is None or position >= self.rope_cos_sin_cache.shape[0]:
            return queries
        
        B, H, D = queries.shape
        half_d = D // 2
        
        # Get cos/sin for the position (move the single row to the queries device/dtype)
        cos_sin = self.rope_cos_sin_cache[position]
        cos_sin = cos_sin.to(queries.device, dtype=queries.dtype, non_blocking=True)
        cos_theta, sin_theta = cos_sin[:half_d], cos_sin[half_d:]
        
        # Reshape for rotation
        q_reshape = queries.view(B, H, half_d, 2)
        x1, x2 = q_reshape[..., 0], q_reshape[..., 1]
        
        # Apply R(θ) = [[cos, -sin], [sin, cos]]
        x1_rotated = cos_theta * x1 - sin_theta * x2
        x2_rotated = sin_theta * x1 + cos_theta * x2
        
        return torch.stack([x1_rotated, x2_rotated], dim=-1).view(B, H, D)

        return None, None   # no use the return value
    

    def compute(self, queries, layer_idx, queries_next=None):
        """
        queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
        """
        # assert queries.size(0) == self.batch_size
        # assert queries.size(1) == 1
        # assert queries.size(2) == self.kv_head * self.group_size == self.num_heads
        # assert queries.size(3) == self.head_dim
        
        # Residual MLP prediction in rotation-neutral space (optional, controlled by env)
        if (self.enable_mlp_prediction or self.collect_samples) and hasattr(self, 'rope_cos_sin_cache') and self.rope_cos_sin_cache is not None:
            B = queries.size(0)
            cur_pos = self.context - 1 if layer_idx == self.layer_num - 1 else self.context
            cur_pos = max(0, cur_pos)
            
            # Reshape queries to [B, H, D]
            queries_orig = queries.view(B, self.num_heads, self.head_dim)
            
            # Unwind to rotation-neutral space
            q_neutral_t = self.apply_rope_inverse(queries_orig, cur_pos)
            
            # Get previous neutral queries
            prev_neutral = self.prev_queries_neutral[layer_idx]
            prev_prev_neutral = self.prev_prev_queries_neutral[layer_idx]
            
            # Collect samples for training
            if self.collect_samples and self.sample_collector and prev_neutral is not None:
                # We have (prev_neutral, q_neutral_t); need q_next for triplet
                # Store for next step collection (will be saved when q_{t+1} arrives)
                pass
            
            # MLP prediction
            if self.enable_mlp_prediction and self.mlp_models[layer_idx] is not None and prev_neutral is not None:
                # Build input features based on mlp_input_mode
                delta_cur = q_neutral_t - prev_neutral
                if self.mlp_input_mode == 'concat':
                    mlp_input = torch.cat([q_neutral_t, prev_neutral], dim=-1)  # [B, H, 2D]
                elif self.mlp_input_mode == 'concat_delta':
                    mlp_input = torch.cat([q_neutral_t, delta_cur], dim=-1)  # [B, H, 2D]
                elif self.mlp_input_mode == 'delta_only':
                    mlp_input = delta_cur  # [B, H, D]
                else:
                    mlp_input = torch.cat([q_neutral_t, delta_cur], dim=-1)
                
                # Flatten batch & heads for MLP: [B*H, input_dim]
                mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
                
                # Predict delta: delta_pred = MLP(input) (residual is inside MLP)
                with torch.no_grad():
                    delta_pred = self.mlp_models[layer_idx](mlp_input_flat)  # [B*H, D]
                
                # Reconstruct predicted next query in neutral space
                q_pred_neutral = q_neutral_t.view(-1, self.head_dim) + delta_pred
                q_pred_neutral = q_pred_neutral.view(B, self.num_heads, self.head_dim)
                
                # Rewind to position t+1
                q_pred_rotated = self.apply_rope_forward(q_pred_neutral, cur_pos + 1)
                
                # Replace queries (or only for sim logging if mlp_apply_to_sim_only=True)
                if not self.mlp_apply_to_sim_only:
                    queries = q_pred_rotated.view(B, 1, self.num_heads, self.head_dim)
            
            # Update history for next step (and sample collection)
            if prev_neutral is not None and prev_prev_neutral is not None and self.collect_samples and self.sample_collector:
                # Now we have triplet: (prev_prev, prev, cur) -> save
                self.sample_collector.add_sample(layer_idx, prev_prev_neutral, prev_neutral, q_neutral_t)
            
            self.prev_prev_queries_neutral[layer_idx] = prev_neutral
            self.prev_queries_neutral[layer_idx] = q_neutral_t.detach()

        # Optional: measure cosine similarity between adjacent-step queries (per layer)
        if self.sim_log_enabled:
            try:
                # Whitening for similarity logging is disabled; using RoPE-only corrections.

                # reshape to [B, KV, G, D], and derive [B, H, D] for per-Query-head stats
                B = queries.size(0)
                q4d = queries.view(B, 1, self.num_heads, self.head_dim).squeeze(1)  # [B, H, D]
                q4d = q4d.view(B, self.kv_head, self.group_size, self.head_dim).contiguous()  # [B, KV, G, D]
                prev = self.prev_queries[layer_idx]
                if prev is not None and prev.shape == q4d.shape:
                    eps = 1e-6
                    # token pair id for this step (compare step-1 & step)
                    step = self.layer_step[layer_idx]
                    if step > 0:
                        token_pair = f"token{step-1}&{step}"
                        # compute per-query-head cos: reshape prev/current to [B, H, D]
                        prev_q = prev.view(B, self.kv_head * self.group_size, self.head_dim)  # [B, H, D]
                        cur_q = q4d.view(B, self.kv_head * self.group_size, self.head_dim)    # [B, H, D]
                        
                        # Apply RoPE unwinding/rewinding correction:
                        # For prev_q at position (step-1), apply rotation to position step
                        # This removes the purely positional phase shift
                        if self.enable_rope_correction and hasattr(self, 'rope_cos_sin_cache'):
                            # Get cos/sin for positions step-1 and step
                            pos_prev = self.context + step - 1
                            pos_cur = self.context + step
                            if pos_prev < self.rope_cos_sin_cache.shape[0] and pos_cur < self.rope_cos_sin_cache.shape[0]:
                                # Extract cos and sin for both positions
                                cos_sin_prev = self.rope_cos_sin_cache[pos_prev]  # [D]
                                cos_sin_cur = self.rope_cos_sin_cache[pos_cur]    # [D]
                                # Move to the same device/dtype as prev_q for safe ops
                                cos_sin_prev = cos_sin_prev.to(prev_q.device, dtype=prev_q.dtype, non_blocking=True)
                                cos_sin_cur = cos_sin_cur.to(prev_q.device, dtype=prev_q.dtype, non_blocking=True)
                                half_d = self.head_dim // 2
                                cos_prev, sin_prev = cos_sin_prev[:half_d], cos_sin_prev[half_d:]
                                cos_cur, sin_cur = cos_sin_cur[:half_d], cos_sin_cur[half_d:]
                                
                                # Apply R(θ_{step}) R(-θ_{step-1}) to prev_q
                                # First unwind: apply R(-θ_{step-1})
                                prev_q_reshape = prev_q.view(B, self.kv_head * self.group_size, half_d, 2)  # [B, H, D/2, 2]
                                x1_prev, x2_prev = prev_q_reshape[..., 0], prev_q_reshape[..., 1]
                                # R(-θ) = [[cos, sin], [-sin, cos]]
                                x1_unwound = cos_prev * x1_prev + sin_prev * x2_prev
                                x2_unwound = -sin_prev * x1_prev + cos_prev * x2_prev
                                
                                # Then rewind: apply R(θ_{step})
                                # R(θ) = [[cos, -sin], [sin, cos]]
                                x1_rewound = cos_cur * x1_unwound - sin_cur * x2_unwound
                                x2_rewound = sin_cur * x1_unwound + cos_cur * x2_unwound
                                
                                # Stack back
                                prev_q_mod = torch.stack([x1_rewound, x2_rewound], dim=-1).view(B, self.kv_head * self.group_size, self.head_dim)
                                prev_q = prev_q_mod
                        
                        # Optionally apply online remove-top-1-PC whitening to prev_q and cur_q
                        if self.whiten_online:
                            # operate on flattened [B*H, D]
                            flat_prev = prev_q.view(-1, self.head_dim)
                            flat_cur = cur_q.view(-1, self.head_dim)
                            layer_mean = self.whiten_mean[layer_idx]
                            pc = self.whiten_pc[layer_idx]
                            # init if needed
                            if layer_mean is None:
                                layer_mean = flat_prev.mean(dim=0, keepdim=True)
                                pc = torch.zeros((self.head_dim,), device=flat_prev.device, dtype=flat_prev.dtype)
                                self.whiten_mean[layer_idx] = layer_mean
                                self.whiten_pc[layer_idx] = pc

                            # center
                            flat_prev_c = flat_prev - layer_mean
                            flat_cur_c = flat_cur - layer_mean

                            # update running mean (very small step to avoid instability)
                            # using simple exponential moving average
                            decay = 1.0 - self.whiten_lr
                            self.whiten_mean[layer_idx] = decay * layer_mean + self.whiten_lr * flat_prev.mean(dim=0, keepdim=True)

                            # Oja update for top-1 PC: pc <- pc + lr * (x*(x·pc) - (pc)(x·x)) approximated
                            # Simpler stable variant: pc += lr * (mean(x * (x@pc))) then normalize
                            # compute projection of flat_prev_c on pc
                            if pc.abs().sum() == 0:
                                # initialize pc as first principal direction approx via mean
                                # (avoid abs() which biases sign; use mean to capture common direction)
                                pc = flat_prev_c.mean(dim=0).to(flat_prev.device)
                            else:
                                proj = torch.matmul(flat_prev_c, pc)
                                update = (flat_prev_c * proj.unsqueeze(1)).mean(dim=0)
                                pc = pc + self.whiten_lr * update
                            # normalize pc
                            pc_norm = pc.norm(p=2)
                            if pc_norm > 0:
                                pc = pc / pc_norm
                            self.whiten_pc[layer_idx] = pc

                            # increment step counter and only remove the component after warmup
                            self.whiten_steps[layer_idx] += 1
                            if self.whiten_steps[layer_idx] >= self.whiten_min_steps:
                                # remove top-1 component: x' = x - (x·pc) pc
                                proj_prev = torch.matmul(flat_prev_c, pc)
                                proj_cur = torch.matmul(flat_cur_c, pc)
                                flat_prev_c = flat_prev_c - proj_prev.unsqueeze(1) * pc.unsqueeze(0)
                                flat_cur_c = flat_cur_c - proj_cur.unsqueeze(1) * pc.unsqueeze(0)

                            # reshape back
                            prev_q = flat_prev_c.view(prev_q.shape)
                            cur_q = flat_cur_c.view(cur_q.shape)

                        num = (cur_q * prev_q).sum(dim=-1)  # [B, H]
                        denom = (cur_q.norm(dim=-1) * prev_q.norm(dim=-1)).clamp_min(eps)
                        cos_h = (num / denom).clamp(-1.0, 1.0).detach().float()  # [B, H]
                        # Aggregate across batch, then summarize across heads in one row
                        cos_h_mean = cos_h.mean(dim=0)  # [H]
                        overall_mean = cos_h_mean.mean().item()
                        overall_min = cos_h_mean.min().item()
                        overall_max = cos_h_mean.max().item()
                        overall_std = cos_h_mean.std(unbiased=False).item()
                        with open(self.sim_summary_path, "a") as fsum:
                            fsum.write(f"{token_pair},{layer_idx+1},{overall_mean:.4f},{overall_min:.4f},{overall_max:.4f},{overall_std:.4f}\n")
                # update previous to current
                self.prev_queries[layer_idx] = q4d.detach()
                self.layer_step[layer_idx] += 1
            except Exception:
                # Never let logging break inference
                pass

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
            dist = torch.sum(self.softmax_o, dim=1)     # [batch_size*group_num, n_centroids]
            dist.masked_fill_(self.centroids_mask[layer_idx], self.DTYPE_MIN)
            self.cI[buffer_idx] = torch.topk(dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True)[1] # [batch_size*group_num, max_consider_cluster]
            self.cluster_ids[buffer_idx].copy_(self.cI[buffer_idx][..., :self.nprobe])
            # print ("layer ", layer_idx, "selected clusters:", self.cluster_ids[layer_idx])
            # end = time.perf_counter()
            # print (f"layer {layer_idx} select clusters: {(end-start) * 1000:.4f} ms")
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
        # estimation, next layer access sync
        if self.use_cluster_estimation and layer_idx > 1 and layer_idx < self.layer_num - 1:
            self.wave_buffer[layer_idx + 1].sync()
            self.wave_buffer[layer_idx].batch_update()

        # assemble the execution buffer
        # start = time.perf_counter()
        # current layer copy
        if layer_idx < 2 or not self.use_cluster_estimation:
            # print (self.list_keys[layer_idx].device, self.cache_keys[layer_idx].device, self.execution_buffer_keys.device)
            # print ("hit ", torch.sum (self.hit_unit_sizes[layer_idx], dim=1))
            # print ("miss ", torch.sum (self.miss_unit_sizes[layer_idx], dim=1))
            torch.cuda.nvtx.range_push("current_layer_copy")
            gather_copy_and_concat(self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys[buffer_idx],
                                self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values[buffer_idx],
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
    
    def save_collected_samples(self):
        """Save collected neutral-space samples to disk (call at end of inference)."""
        if self.collect_samples and self.sample_collector:
            self.sample_collector.save_all()

