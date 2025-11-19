import time
import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import gather_copy_and_concat
from weighted_flash_decoding import weighted_flash_decoding
import flashinfer

def profile_flashinfer(
    num_pages: int,
    selected_pages: int,
    page_size_full: int,
    page_size: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    num_pages *= batch_size

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    if device == 'cuda':
        keys = torch.randn((num_pages, page_size, 1, head_dim), dtype=dtype, device='cuda').contiguous()
        values = torch.randn((num_pages, page_size, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    else:
        keys = torch.randn((num_pages, page_size, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()
        values = torch.randn((num_pages, page_size, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()

    if page_size == 1:
        # Generate consecutive indices with random chunk sizes from [1, 15] that sum to selected_pages
        kv_indices_list = []
        kv_indptr_list = [0]

        for i in range(batch_size * kv_head):
            remaining = selected_pages
            current_indices = []

            while remaining > 0:
                # Random chunk size between 1 and min(2 * page_size_full - 1, remaining)
                chunk_size = torch.randint(1, min(2 * page_size_full, remaining + 1), (1,)).item()
                # Random starting position for this chunk
                start_idx = torch.randint(0, num_pages - chunk_size + 1, (1,)).item()
                # Add consecutive indices
                current_indices.extend(range(start_idx, start_idx + chunk_size))
                remaining -= chunk_size

            kv_indices_list.extend(current_indices)
            kv_indptr_list.append(len(kv_indices_list))

        kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32)
        kv_last_page_len = torch.full((batch_size * kv_head,), page_size, dtype=torch.int32)
    else:
        kv_indptr = torch.tensor(
            [i * selected_pages for i in range(batch_size * kv_head + 1)], dtype=torch.int32
        )
        kv_indices = torch.randperm(num_pages, dtype=torch.int32)[: batch_size * kv_head * selected_pages]
        kv_last_page_len = torch.full((batch_size * kv_head,), page_size, dtype=torch.int32)

    # warmup
    for it in range(5):
        decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            4,
            1,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper.run(
            queries,
            (keys, values),
            return_lse=True
        )
    
    torch.cuda.synchronize()

    start = time.perf_counter()
    # profile
    for it in range(10):
        decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            4,
            1,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper.run(
            queries,
            (keys, values),
            return_lse=True
        )
        torch.cuda.synchronize()
    end = time.perf_counter()
    total_time = end - start
    # print(f"Selected Pages: {selected_pages}, Page Size: {page_size}, Batch Size: {batch_size}")
    # print(f"FlashInfer Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")

    return total_time / 10 * 1000

def profile_flashinfer_mixed(
    num_pages: int,
    selected_pages_full: int,
    page_size_full: int,
    selected_pages_scatter: int,
    page_size_scatter: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    decode_wrapper_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    decode_wrapper_scatter = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    num_pages *= batch_size

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    if device == 'cuda':
        keys_full = torch.randn((num_pages, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()
        values_full = torch.randn((num_pages, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()
        keys_scatter = torch.randn((num_pages, page_size_scatter, 1, head_dim), dtype=dtype, device='cuda').contiguous()
        values_scatter = torch.randn((num_pages, page_size_scatter, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    else:
        keys_full = torch.randn((num_pages, page_size_full, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()
        values_full = torch.randn((num_pages, page_size_full, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()
        keys_scatter = torch.randn((num_pages, page_size_scatter, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()
        values_scatter = torch.randn((num_pages, page_size_scatter, 1, head_dim), dtype=dtype, pin_memory=True).contiguous()

    # full pages
    kv_indptr_full = torch.tensor(
        [i * selected_pages_full for i in range(batch_size * kv_head + 1)], dtype=torch.int32
    )
    kv_indices_full = torch.randperm(num_pages, dtype=torch.int32)[: batch_size * kv_head * selected_pages_full]
    kv_last_page_len_full = torch.full((batch_size * kv_head,), page_size_full, dtype=torch.int32)

    # scatter pages
    kv_indices_list = []
    kv_indptr_list = [0]

    for i in range(batch_size * kv_head):
        remaining = selected_pages_scatter
        current_indices = []

        while remaining > 0:
            # Random chunk size between 1 and min(2 * page_size_full - 1, remaining)
            chunk_size = torch.randint(1, min(2 * page_size_full, remaining + 1), (1,)).item()
            # Random starting position for this chunk
            start_idx = torch.randint(0, num_pages - chunk_size + 1, (1,)).item()
            # Add consecutive indices
            current_indices.extend(range(start_idx, start_idx + chunk_size))
            remaining -= chunk_size

        kv_indices_list.extend(current_indices)
        kv_indptr_list.append(len(kv_indices_list))

    kv_indptr_scatter = torch.tensor(kv_indptr_list, dtype=torch.int32)
    kv_indices_scatter = torch.tensor(kv_indices_list, dtype=torch.int32)
    kv_last_page_len_scatter = torch.full((batch_size * kv_head,), page_size_scatter, dtype=torch.int32)

    # warmup
    for it in range(5):
        decode_wrapper_full.plan(
            kv_indptr_full,
            kv_indices_full,
            kv_last_page_len_full,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        decode_wrapper_scatter.plan(
            kv_indptr_scatter,
            kv_indices_scatter,
            kv_last_page_len_scatter,
            4,
            1,
            head_dim,
            page_size_scatter,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper_full.run(
            queries,
            (keys_full, values_full),
            return_lse=True
        )

        attn_out_scatter, lse_out_scatter = decode_wrapper_scatter.run(
            queries,
            (keys_scatter, values_scatter),
            return_lse=True
        )

        flashinfer.cascade.merge_state_in_place(attn_out, lse_out, attn_out_scatter, lse_out_scatter)

    torch.cuda.synchronize()

    start = time.perf_counter()
    # profile
    for it in range(10):
        decode_wrapper_full.plan(
            kv_indptr_full,
            kv_indices_full,
            kv_last_page_len_full,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        decode_wrapper_scatter.plan(
            kv_indptr_scatter,
            kv_indices_scatter,
            kv_last_page_len_scatter,
            4,
            1,
            head_dim,
            page_size_scatter,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper_full.run(
            queries,
            (keys_full, values_full),
            return_lse=True
        )

        attn_out, lse_out = decode_wrapper_scatter.run(
            queries,
            (keys_scatter, values_scatter),
            return_lse=True
        )
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    total_time = end - start
    # print(f"Selected Pages Full: {selected_pages_full}, Page Size Full: {page_size_full}, "
    #       f"Selected Pages Scatter: {selected_pages_scatter}, Page Size Scatter: {page_size_scatter}, "
    #       f"Batch Size: {batch_size}")
    # print(f"FlashInfer Mixed Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000

def profile_gather_attention(
    n_centroids: int,
    nprobe: int,
    page_size: int,
    cache_cluster_num: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    input_length = 120000
    static_pattern_total = 4 + 64
    max_new_length = 1024
    THRESHOLD_LENGTH = 1024
    input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
    pages_per_cluster = 2
    buffer_size = max(int(nprobe * 4), 16) * pages_per_cluster
    static_stride = static_pattern_total + max_new_length
    cache_size = cache_cluster_num * pages_per_cluster
    batch_groups = batch_size * kv_head
    list_stride = input_length - static_pattern_total + input_length_new
    cache_stride = cache_size
    static_len = static_pattern_total

    steady_zone_keys = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")
    steady_zone_values = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")

    if device == 'cuda':
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
    else:
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()

    cache_keys = torch.zeros((batch_size, kv_head, cache_size, page_size, head_dim),
                                dtype=dtype, device="cuda").contiguous()
    cache_values = torch.zeros((batch_size, kv_head, cache_size, page_size, head_dim),
                                dtype=dtype, device="cuda").contiguous()

    hit_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
    
    # pin memory indices for missing clusters
    miss_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
                                    
    execution_buffer_keys = torch.zeros((batch_size*kv_head, buffer_size*page_size+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    execution_buffer_values = torch.zeros((batch_size*kv_head, buffer_size*page_size+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    valid_lengths = torch.zeros((batch_size*kv_head), dtype=torch.int32, device="cuda").contiguous()
    execution_stride = buffer_size * page_size + static_stride

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    # init missing units
    miss_unit_idices[:, :nprobe] = torch.randint(0, input_length - static_pattern_total + input_length_new,
                                                    (batch_groups, nprobe), dtype=torch.int32)
    if page_size == 1:
        # Generate random sizes from [1, 15] that sum to nprobe for each batch group
        for bg in range(batch_groups):
            remaining = nprobe
            num_units = 0
            while remaining > 0:
                # Random size between 1 and min(15, remaining)
                size = torch.randint(1, min(16, remaining + 1), (1,)).item()
                miss_unit_sizes[bg, num_units] = size
                remaining -= size
                num_units += 1
            miss_num_units[bg] = num_units
    else:
        miss_unit_sizes[:, :nprobe].fill_(page_size)
    
    miss_unit_sizes_cumsum[:, :nprobe] = torch.cumsum(miss_unit_sizes[:, :nprobe], dim=1, dtype=torch.int32)
    miss_num_units.fill_(nprobe)

    valid_lengths[:] = miss_unit_sizes_cumsum[:, nprobe - 1]

    # print(miss_unit_idices)
    # print(miss_unit_sizes)
    # print(miss_num_units)
    # print(hit_unit_idices)
    # print(hit_unit_sizes)
    # print(hit_num_units)
    # print(valid_lengths)

    # warmup
    for it in range(5):
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
    torch.cuda.synchronize()

    # profile
    gather_total = 0
    attention_total = 0
    start = time.perf_counter()
    for it in range(10):
        start_gather = time.perf_counter()
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        torch.cuda.synchronize()
        end_gather = time.perf_counter()
        gather_total += end_gather - start_gather

        start_attn = time.perf_counter()
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
        torch.cuda.synchronize()
        end_attn = time.perf_counter()
        attention_total += end_attn - start_attn
    end = time.perf_counter()
    total_time = end - start
    # print(f"Gather Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000, gather_total / 10 * 1000, attention_total /10 * 1000

def profile_gather_attention_mixed(
    n_centroids: int,
    nprobe: int,
    page_size_full: int,
    page_size_scatter: int,
    cache_cluster_num: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    input_length = 120000
    static_pattern_total = 4 + 64
    max_new_length = 1024
    THRESHOLD_LENGTH = 1024
    input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
    pages_per_cluster = 2
    buffer_size = max(int(nprobe * 4), 16) * pages_per_cluster
    static_stride = static_pattern_total + max_new_length
    cache_size = cache_cluster_num * pages_per_cluster
    batch_groups = batch_size * kv_head
    list_stride = input_length - static_pattern_total + input_length_new
    cache_stride = cache_size
    static_len = static_pattern_total

    steady_zone_keys = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")
    steady_zone_values = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")

    if device == 'cuda':
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
    else:
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()

    cache_keys = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()
    cache_values = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()

    hit_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
    
    # pin memory indices for missing clusters
    miss_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
                                    
    execution_buffer_keys = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    execution_buffer_values = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    valid_lengths = torch.zeros((batch_size*kv_head), dtype=torch.int32, device="cuda").contiguous()
    execution_stride = buffer_size * page_size_full + static_stride

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    # init missing units
    miss_unit_idices[:, :nprobe] = torch.randint(0, input_length - static_pattern_total + input_length_new,
                                                    (batch_groups, nprobe), dtype=torch.int32)
    # Generate sizes from normal distribution with mean=page_size_full, std=page_size_full/4
    # Clamp to be at least 1
    miss_unit_sizes[:, :nprobe] = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe)).to(torch.int32),
        min=1
    )
    miss_unit_sizes_cumsum[:, :nprobe] = torch.cumsum(miss_unit_sizes[:, :nprobe], dim=1, dtype=torch.int32)
    miss_num_units.fill_(nprobe)

    valid_lengths[:] = miss_unit_sizes_cumsum[:, nprobe - 1]

    # print(miss_unit_idices)
    # print(miss_unit_sizes)
    # print(miss_num_units)
    # print(hit_unit_idices)
    # print(hit_unit_sizes)
    # print(hit_num_units)
    # print(valid_lengths)

    # warmup
    for it in range(5):
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
    torch.cuda.synchronize()

    # profile
    gather_total = 0
    attention_total = 0
    start = time.perf_counter()
    for it in range(10):
        start_gather = time.perf_counter()
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        torch.cuda.synchronize()
        end_gather = time.perf_counter()
        gather_total += end_gather - start_gather

        start_attn = time.perf_counter()
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
        torch.cuda.synchronize()
        end_attn = time.perf_counter()
        attention_total += end_attn - start_attn
    end = time.perf_counter()
    total_time = end - start
    # print(f"Gather Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000, gather_total / 10 * 1000, attention_total /10 * 1000

def profile_gather_attention_mixed_cache(
    n_centroids: int,
    nprobe: int,
    cache_cluster_num: int,
    cache_hit_rate: float,
    num_pages: int,
    selected_pages_full: int,
    page_size_full: int,
    selected_pages_scatter: int,
    page_size_scatter: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    input_length = 120000
    static_pattern_total = 4 + 64
    max_new_length = 1024
    THRESHOLD_LENGTH = 1024
    input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
    pages_per_cluster = 2
    buffer_size = max(int(nprobe * 4), 16) * pages_per_cluster
    static_stride = static_pattern_total + max_new_length
    cache_size = cache_cluster_num * pages_per_cluster
    batch_groups = batch_size * kv_head
    list_stride = input_length - static_pattern_total + input_length_new
    cache_stride = cache_size
    static_len = static_pattern_total

    steady_zone_keys = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")
    steady_zone_values = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")

    if device == 'cuda':
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
    else:
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()

    cache_keys = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()
    cache_values = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()

    hit_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
    
    # pin memory indices for missing clusters
    miss_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
                                    
    execution_buffer_keys = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    execution_buffer_values = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    valid_lengths = torch.zeros((batch_size*kv_head), dtype=torch.int32, device="cuda").contiguous()
    execution_stride = buffer_size * page_size_full + static_stride

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    # init hit units
    nprobe_hit = int(nprobe * cache_hit_rate)
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_hit)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    hit_page_idx = 0
    for batch_idx in range(batch_groups):
        for cluster_idx in range(nprobe_hit):
            cluster_size = cluster_sizes_tokens[batch_idx, cluster_idx].item()
            num_pages = (cluster_size + page_size_full - 1) // page_size_full  # ceil division

            # Populate pages for this cluster
            for page_offset in range(num_pages):
                if hit_page_idx >= buffer_size:
                    break

                # Random page index from cache
                hit_unit_idices[batch_idx, hit_page_idx] = torch.randint(0, cache_size, (1,), dtype=torch.int32).item()

                # Determine page size (full or partial for last page)
                remaining_tokens = cluster_size - page_offset * page_size_full
                page_size = min(page_size_full, remaining_tokens)
                hit_unit_sizes[batch_idx, hit_page_idx] = page_size

                hit_page_idx += 1

        hit_num_units[batch_idx] = hit_page_idx
        hit_page_idx = 0  # reset for next batch

    hit_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(hit_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # init missing units
    nprobe_miss = nprobe - nprobe_hit
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens_miss = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_miss)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    miss_page_idx = 0
    max_list_idx = input_length - static_pattern_total + input_length_new
    for batch_idx in range(batch_groups):
        for cluster_idx in range(nprobe_miss):
            cluster_size = cluster_sizes_tokens_miss[batch_idx, cluster_idx].item()
            num_pages = (cluster_size + page_size_full - 1) // page_size_full  # ceil division

            # Populate pages for this cluster
            for page_offset in range(num_pages):
                if miss_page_idx >= buffer_size:
                    break

                # Random page index from list
                miss_unit_idices[batch_idx, miss_page_idx] = torch.randint(0, max_list_idx, (1,), dtype=torch.int32).item()

                # Determine page size (full or partial for last page)
                remaining_tokens = cluster_size - page_offset * page_size_full
                page_size = min(page_size_full, remaining_tokens)
                miss_unit_sizes[batch_idx, miss_page_idx] = page_size

                miss_page_idx += 1

        miss_num_units[batch_idx] = miss_page_idx
        miss_page_idx = 0  # reset for next batch

    miss_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(miss_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # Calculate valid lengths using actual number of units per batch group
    for batch_idx in range(batch_groups):
        hit_count = hit_num_units[batch_idx].item()
        miss_count = miss_num_units[batch_idx].item()
        hit_total = hit_unit_sizes_cumsum[batch_idx, hit_count - 1].item() if hit_count > 0 else 0
        miss_total = miss_unit_sizes_cumsum[batch_idx, miss_count - 1].item() if miss_count > 0 else 0
        valid_lengths[batch_idx] = hit_total + miss_total

    # print(miss_unit_idices)
    # print(miss_unit_sizes)
    # print(miss_num_units)
    # print(hit_unit_idices)
    # print(hit_unit_sizes)
    # print(hit_num_units)
    # print(valid_lengths)

    # warmup
    for it in range(5):
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
    torch.cuda.synchronize()

    # profile
    gather_total = 0
    attention_total = 0

    start = time.perf_counter()
    for it in range(10):
        start_gather = time.perf_counter()
        gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)
        torch.cuda.synchronize()
        end_gather = time.perf_counter()
        gather_total += end_gather - start_gather

        start_attention = time.perf_counter()
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=None,
            previous_lse=None,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
        torch.cuda.synchronize()
        end_attention = time.perf_counter()
        attention_total += end_attention - start_attention
    
    end = time.perf_counter()
    total_time = end - start
    # print(f"Gather Time: {gather_total / 10 * 1000:.2f} ms, Attention Time: {attention_total /10 * 1000:.2f} ms")
    # print(f"Gather Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000, gather_total / 10 * 1000, attention_total /10 * 1000

def profile_mixed_cache(
    n_centroids: int,
    nprobe: int,
    cache_cluster_num: int,
    cache_hit_rate: float,
    num_pages: int,
    selected_pages_full: int,
    page_size_full: int,
    selected_pages_scatter: int,
    page_size_scatter: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    # flashinfer setup
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    keys = torch.randn((cache_cluster_num, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    values = torch.randn((cache_cluster_num, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()

    input_length = 120000
    static_pattern_total = 4 + 64
    max_new_length = 1024
    THRESHOLD_LENGTH = 1024
    input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
    pages_per_cluster = 2
    buffer_size = max(int(nprobe * 4), 16) * pages_per_cluster
    static_stride = static_pattern_total + max_new_length
    cache_size = cache_cluster_num * pages_per_cluster
    batch_groups = batch_size * kv_head
    list_stride = input_length - static_pattern_total + input_length_new
    cache_stride = cache_size
    static_len = static_pattern_total

    steady_zone_keys = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")
    steady_zone_values = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")

    if device == 'cuda':
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
    else:
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()

    cache_keys = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()
    cache_values = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()

    hit_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
    
    # pin memory indices for missing clusters
    miss_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
                                    
    execution_buffer_keys = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    execution_buffer_values = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    valid_lengths = torch.zeros((batch_size*kv_head), dtype=torch.int32, device="cuda").contiguous()
    execution_stride = buffer_size * page_size_full + static_stride

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()
    
    # init hit units
    nprobe_hit = int(nprobe * cache_hit_rate)
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_hit)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    full_page_count = []
    hit_page_idx = 0
    for batch_idx in range(batch_groups):
        cur_full_page_count = 0
        for cluster_idx in range(nprobe_hit):
            cluster_size = cluster_sizes_tokens[batch_idx, cluster_idx].item()
            cur_full_page_count += cluster_size // page_size_full
            num_pages = (cluster_size % page_size_full + page_size_full - 1) // page_size_full  # ceil division

            # Populate pages for this cluster
            for page_offset in range(num_pages):
                if hit_page_idx >= buffer_size:
                    break

                # Random page index from cache
                hit_unit_idices[batch_idx, hit_page_idx] = torch.randint(0, cache_size, (1,), dtype=torch.int32).item()

                # Determine page size (full or partial for last page)
                remaining_tokens = cluster_size % page_size_full - page_offset * page_size_full
                page_size = min(page_size_full, remaining_tokens)
                hit_unit_sizes[batch_idx, hit_page_idx] = page_size

                hit_page_idx += 1

        hit_num_units[batch_idx] = hit_page_idx
        hit_page_idx = 0  # reset for next batch
        full_page_count.append(cur_full_page_count)

    hit_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(hit_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # init missing units
    nprobe_miss = nprobe - nprobe_hit
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens_miss = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_miss)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    miss_page_idx = 0
    max_list_idx = input_length - static_pattern_total + input_length_new
    for batch_idx in range(batch_groups):
        for cluster_idx in range(nprobe_miss):
            cluster_size = cluster_sizes_tokens_miss[batch_idx, cluster_idx].item()
            num_pages = (cluster_size + page_size_full - 1) // page_size_full  # ceil division

            # Populate pages for this cluster
            for page_offset in range(num_pages):
                if miss_page_idx >= buffer_size:
                    break

                # Random page index from list
                miss_unit_idices[batch_idx, miss_page_idx] = torch.randint(0, max_list_idx, (1,), dtype=torch.int32).item()

                # Determine page size (full or partial for last page)
                remaining_tokens = cluster_size - page_offset * page_size_full
                page_size = min(page_size_full, remaining_tokens)
                miss_unit_sizes[batch_idx, miss_page_idx] = page_size

                miss_page_idx += 1

        miss_num_units[batch_idx] = miss_page_idx
        miss_page_idx = 0  # reset for next batch

    miss_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(miss_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # Calculate valid lengths using actual number of units per batch group
    for batch_idx in range(batch_groups):
        hit_count = hit_num_units[batch_idx].item()
        miss_count = miss_num_units[batch_idx].item()
        hit_total = hit_unit_sizes_cumsum[batch_idx, hit_count - 1].item() if hit_count > 0 else 0
        miss_total = miss_unit_sizes_cumsum[batch_idx, miss_count - 1].item() if miss_count > 0 else 0
        valid_lengths[batch_idx] = hit_total + miss_total

    # flashinfer hit full pages
    full_page_count = torch.tensor(full_page_count, dtype=torch.int32)
    kv_indptr = torch.cumsum(torch.cat([torch.zeros((1,), dtype=torch.int32), full_page_count]), dim=0, dtype=torch.int32)
    kv_indices = torch.randint(0, cache_cluster_num, (kv_indptr[-1], ), dtype=torch.int32)
    kv_last_page_len = torch.full((batch_size * kv_head,), page_size_full, dtype=torch.int32)

    # print(miss_unit_idices)
    # print(miss_unit_sizes)
    # print(miss_num_units)
    # print(hit_unit_idices)
    # print(hit_unit_sizes)
    # print(hit_num_units)
    # print(valid_lengths)
    # print(full_page_count)

    gather_stream = torch.cuda.Stream()

    # warmup
    for it in range(5):
        with torch.cuda.stream(gather_stream):
            gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                                steady_zone_values, list_values, cache_values, execution_buffer_values,
                                miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                                hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                                valid_lengths, batch_groups, 
                                static_stride, list_stride, cache_stride,
                                execution_stride, buffer_size, static_len)

        # flashinfer full pages
        decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper.run(
            queries,
            (keys, values),
            return_lse=True
        )

        attn_out = attn_out.view(batch_groups, 1, 4, head_dim)
        lse_out = lse_out.view(batch_groups, 4, 1)
        
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=attn_out,
            previous_lse=lse_out,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
    torch.cuda.synchronize()

    # profile
    flashinfer_total = 0
    gather_total = 0
    gather_attention_total = 0

    start = time.perf_counter()
    for it in range(10):
        with torch.cuda.stream(gather_stream):
            gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)

        # start_gather = time.perf_counter()
        # gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
        #                     steady_zone_values, list_values, cache_values, execution_buffer_values,
        #                     miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
        #                     hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
        #                     valid_lengths, batch_groups, 
        #                     static_stride, list_stride, cache_stride,
        #                     execution_stride, buffer_size, static_len)
        # torch.cuda.current_stream().synchronize()
        # end_gather = time.perf_counter()
        # gather_total += end_gather - start_gather

        # flashinfer full pages
        start_flashinfer = time.perf_counter()
        decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper.run(
            queries,
            (keys, values),
            return_lse=True
        )
        torch.cuda.current_stream().synchronize()
        end_flashinfer = time.perf_counter()
        flashinfer_total += end_flashinfer - start_flashinfer

        attn_out = attn_out.view(batch_groups, 1, 4, head_dim)
        lse_out = lse_out.view(batch_groups, 4, 1)

        start_attention = time.perf_counter()
        gather_stream.synchronize()
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=attn_out,
            previous_lse=lse_out,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
        torch.cuda.synchronize()
        end_attention = time.perf_counter()
        gather_attention_total += end_attention - start_attention

    end = time.perf_counter()
    total_time = end - start
    # print(f"Gather + Attention Time: {gather_attention_total / 10 * 1000:.2f} ms, FlashInfer Time: {flashinfer_total /10 * 1000:.2f} ms")
    # print(f"Gather Time: {gather_total / 10 * 1000:.2f} ms")
    # print(f"Mixed Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000, gather_attention_total / 10 * 1000, flashinfer_total / 10 * 1000
    
def profile_mixed2_cache(
    n_centroids: int,
    nprobe: int,
    cache_cluster_num: int,
    cache_hit_rate: float,
    num_pages: int,
    selected_pages_full: int,
    page_size_full: int,
    selected_pages_scatter: int,
    page_size_scatter: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    # flashinfer setup
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    decode_wrapper_full = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    decode_wrapper_scatter = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    keys_full = torch.randn((cache_cluster_num * 2, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    values_full = torch.randn((cache_cluster_num * 2, page_size_full, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    keys_scatter = torch.randn((cache_cluster_num * 2 * page_size_full, page_size_scatter, 1, head_dim), dtype=dtype, device='cuda').contiguous()
    values_scatter = torch.randn((cache_cluster_num * 2 * page_size_full, page_size_scatter, 1, head_dim), dtype=dtype, device='cuda').contiguous()

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()

    # gather + attention setup
    input_length = 120000
    static_pattern_total = 4 + 64
    max_new_length = 1024
    THRESHOLD_LENGTH = 1024
    input_length_new = ((max_new_length-2) // THRESHOLD_LENGTH) * THRESHOLD_LENGTH
    pages_per_cluster = 2
    buffer_size = max(int(nprobe * 4), 16) * pages_per_cluster
    static_stride = static_pattern_total + max_new_length
    cache_size = cache_cluster_num * pages_per_cluster
    batch_groups = batch_size * kv_head
    list_stride = input_length - static_pattern_total + input_length_new
    cache_stride = cache_size
    static_len = static_pattern_total

    steady_zone_keys = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")
    steady_zone_values = torch.randn((batch_size, kv_head, static_pattern_total+max_new_length, head_dim), 
        dtype=dtype, device="cuda")

    if device == 'cuda':
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                            dtype=dtype, device="cuda").contiguous()
    else:
        list_keys = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()
        list_values = torch.randn((batch_size, kv_head, input_length-static_pattern_total+input_length_new, head_dim), 
                                dtype=dtype, pin_memory=True).contiguous()

    cache_keys = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()
    cache_values = torch.zeros((batch_size, kv_head, cache_size, page_size_full, head_dim),
                                dtype=dtype, device="cuda").contiguous()

    hit_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    hit_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
    
    # pin memory indices for missing clusters
    miss_unit_idices = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_unit_sizes_cumsum = torch.zeros((batch_size*kv_head, buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
    miss_num_units = torch.zeros((batch_size*kv_head), dtype=torch.int32, pin_memory=True).contiguous()
                                    
    execution_buffer_keys = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    execution_buffer_values = torch.zeros((batch_size*kv_head, buffer_size*page_size_full+static_stride, 1, head_dim), 
                                                dtype=dtype, device="cuda").contiguous()
    valid_lengths = torch.zeros((batch_size*kv_head), dtype=torch.int32, device="cuda").contiguous()
    execution_stride = buffer_size * page_size_full + static_stride

    queries = torch.randn((batch_size * kv_head, 4, head_dim), dtype=dtype, device='cuda').contiguous()
    
    # init hit units
    nprobe_hit = int(nprobe * cache_hit_rate)
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_hit)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    # Track full pages and scatter pages separately for FlashInfer
    full_page_indices = []  # List of lists for each batch
    scatter_page_indices = []  # List of lists for each batch
    scatter_page_lens = []  # List of lists for each batch

    hit_page_idx = 0
    for batch_idx in range(batch_groups):
        batch_full_indices = []
        batch_scatter_indices = []
        batch_scatter_lens = []

        for cluster_idx in range(nprobe_hit):
            cluster_size = cluster_sizes_tokens[batch_idx, cluster_idx].item()
            num_full_pages = cluster_size // page_size_full
            remaining_tokens = cluster_size % page_size_full

            # Populate full pages for this cluster
            for page_offset in range(num_full_pages):
                if hit_page_idx >= buffer_size:
                    break

                # Random page index from cache
                page_idx = torch.randint(0, cache_size, (1,), dtype=torch.int32).item()
                # hit_unit_idices[batch_idx, hit_page_idx] = page_idx
                # hit_unit_sizes[batch_idx, hit_page_idx] = page_size_full
                batch_full_indices.append(page_idx)

                hit_page_idx += 1

            # Populate scatter page if there are remaining tokens
            if remaining_tokens > 0 and hit_page_idx < buffer_size:
                page_idx = torch.randint(0, cache_size, (1,), dtype=torch.int32).item()
                # hit_unit_idices[batch_idx, hit_page_idx] = page_idx
                # hit_unit_sizes[batch_idx, hit_page_idx] = remaining_tokens
                batch_scatter_indices.append(page_idx)
                batch_scatter_lens.append(remaining_tokens)

                hit_page_idx += 1

        # hit_num_units[batch_idx] = hit_page_idx
        hit_page_idx = 0  # reset for next batch

        full_page_indices.append(batch_full_indices)
        scatter_page_indices.append(batch_scatter_indices)
        scatter_page_lens.append(batch_scatter_lens)

    # hit_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(hit_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # Initialize FlashInfer data structures for full pages
    full_page_counts = [len(indices) for indices in full_page_indices]
    kv_indptr_full = torch.cumsum(
        torch.cat([torch.zeros((1,), dtype=torch.int32), torch.tensor(full_page_counts, dtype=torch.int32)]),
        dim=0, dtype=torch.int32
    )
    kv_indices_full = torch.cat([torch.tensor(indices, dtype=torch.int32) for indices in full_page_indices]) if sum(full_page_counts) > 0 else torch.tensor([], dtype=torch.int32)
    kv_last_page_len_full = torch.full((batch_groups,), page_size_full, dtype=torch.int32)

    # Initialize FlashInfer data structures for scatter pages
    scatter_page_counts = [len(indices) for indices in scatter_page_indices]
    kv_indptr_scatter = torch.cumsum(
        torch.cat([torch.zeros((1,), dtype=torch.int32), torch.tensor(scatter_page_counts, dtype=torch.int32)]),
        dim=0, dtype=torch.int32
    )
    kv_indices_scatter = torch.cat([torch.tensor(indices, dtype=torch.int32) for indices in scatter_page_indices]) if sum(scatter_page_counts) > 0 else torch.tensor([], dtype=torch.int32)
    # Last page length for each batch group (the last scatter page's length)
    kv_last_page_len_scatter = torch.tensor(
        [lens[-1] if len(lens) > 0 else page_size_scatter for lens in scatter_page_lens],
        dtype=torch.int32
    )

    # init missing units
    nprobe_miss = nprobe - nprobe_hit
    # Generate clusters with mean size of 2 * page_size_full tokens
    cluster_sizes_tokens_miss = torch.clamp(
        torch.normal(mean=float(2 * page_size_full), std=float(2 * page_size_full) / 4.0,
                     size=(batch_groups, nprobe_miss)).to(torch.int32),
        min=1
    )

    # Convert cluster sizes to pages, populating indices and sizes
    miss_page_idx = 0
    max_list_idx = input_length - static_pattern_total + input_length_new
    for batch_idx in range(batch_groups):
        for cluster_idx in range(nprobe_miss):
            cluster_size = cluster_sizes_tokens_miss[batch_idx, cluster_idx].item()
            num_pages = (cluster_size + page_size_full - 1) // page_size_full  # ceil division

            # Populate pages for this cluster
            for page_offset in range(num_pages):
                if miss_page_idx >= buffer_size:
                    break

                # Random page index from list
                miss_unit_idices[batch_idx, miss_page_idx] = torch.randint(0, max_list_idx, (1,), dtype=torch.int32).item()

                # Determine page size (full or partial for last page)
                remaining_tokens = cluster_size - page_offset * page_size_full
                page_size = min(page_size_full, remaining_tokens)
                miss_unit_sizes[batch_idx, miss_page_idx] = page_size

                miss_page_idx += 1

        miss_num_units[batch_idx] = miss_page_idx
        miss_page_idx = 0  # reset for next batch

    miss_unit_sizes_cumsum[:, :buffer_size] = torch.cumsum(miss_unit_sizes[:, :buffer_size], dim=1, dtype=torch.int32)

    # Calculate valid lengths using actual number of units per batch group
    for batch_idx in range(batch_groups):
        hit_count = hit_num_units[batch_idx].item()
        miss_count = miss_num_units[batch_idx].item()
        hit_total = hit_unit_sizes_cumsum[batch_idx, hit_count - 1].item() if hit_count > 0 else 0
        miss_total = miss_unit_sizes_cumsum[batch_idx, miss_count - 1].item() if miss_count > 0 else 0
        valid_lengths[batch_idx] = hit_total + miss_total

    # print(miss_unit_idices)
    # print(miss_unit_sizes)
    # print(miss_num_units)
    # print(hit_unit_idices)
    # print(hit_unit_sizes)
    # print(hit_num_units)
    # print(valid_lengths)

    gather_stream = torch.cuda.Stream()

    # warmup
    for it in range(5):
        with torch.cuda.stream(gather_stream):
            gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                                steady_zone_values, list_values, cache_values, execution_buffer_values,
                                miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                                hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                                valid_lengths, batch_groups, 
                                static_stride, list_stride, cache_stride,
                                execution_stride, buffer_size, static_len)

        # flashinfer
        decode_wrapper_full.plan(
            kv_indptr_full,
            kv_indices_full,
            kv_last_page_len_full,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        decode_wrapper_scatter.plan(
            kv_indptr_scatter,
            kv_indices_scatter,
            kv_last_page_len_scatter,
            4,
            1,
            head_dim,
            page_size_scatter,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper_full.run(
            queries,
            (keys_full, values_full),
            return_lse=True
        )

        attn_out_scatter, lse_out_scatter = decode_wrapper_scatter.run(
            queries,
            (keys_scatter, values_scatter),
            return_lse=True
        )

        attn_out = attn_out.view(batch_groups, 1, 4, head_dim)
        lse_out = lse_out.view(batch_groups, 4, 1)
        
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=attn_out,
            previous_lse=lse_out,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
    torch.cuda.synchronize()

    # profile
    flashinfer_total = 0
    gather_total = 0
    gather_attention_total = 0

    start = time.perf_counter()
    for it in range(10):
        with torch.cuda.stream(gather_stream):
            gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
                            steady_zone_values, list_values, cache_values, execution_buffer_values,
                            miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
                            hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
                            valid_lengths, batch_groups, 
                            static_stride, list_stride, cache_stride,
                            execution_stride, buffer_size, static_len)

        # start_gather = time.perf_counter()
        # gather_copy_and_concat(steady_zone_keys, list_keys, cache_keys, execution_buffer_keys,
        #                     steady_zone_values, list_values, cache_values, execution_buffer_values,
        #                     miss_unit_idices, miss_unit_sizes, miss_unit_sizes_cumsum, miss_num_units,
        #                     hit_unit_idices, hit_unit_sizes, hit_unit_sizes_cumsum, hit_num_units,
        #                     valid_lengths, batch_groups, 
        #                     static_stride, list_stride, cache_stride,
        #                     execution_stride, buffer_size, static_len)
        # torch.cuda.current_stream().synchronize()
        # end_gather = time.perf_counter()
        # gather_total += end_gather - start_gather

        # flashinfer full pages
        start_flashinfer = time.perf_counter()
        decode_wrapper_full.plan(
            kv_indptr_full,
            kv_indices_full,
            kv_last_page_len_full,
            4,
            1,
            head_dim,
            page_size_full,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        decode_wrapper_scatter.plan(
            kv_indptr_scatter,
            kv_indices_scatter,
            kv_last_page_len_scatter,
            4,
            1,
            head_dim,
            page_size_scatter,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        attn_out, lse_out = decode_wrapper_full.run(
            queries,
            (keys_full, values_full),
            return_lse=True
        )

        attn_out_scatter, lse_out_scatter = decode_wrapper_scatter.run(
            queries,
            (keys_scatter, values_scatter),
            return_lse=True
        )
        torch.cuda.current_stream().synchronize()
        end_flashinfer = time.perf_counter()
        flashinfer_total += end_flashinfer - start_flashinfer

        attn_out = attn_out.view(batch_groups, 1, 4, head_dim)
        lse_out = lse_out.view(batch_groups, 4, 1)

        start_attention = time.perf_counter()
        gather_stream.synchronize()
        attn_out, lse_out = weighted_flash_decoding(
            queries.view(batch_groups, 1, 4, head_dim), 
            execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=attn_out,
            previous_lse=lse_out,
            cache_seqlens=valid_lengths,
            return_softmax_lse=True
        )
        torch.cuda.synchronize()
        end_attention = time.perf_counter()
        gather_attention_total += end_attention - start_attention

    end = time.perf_counter()
    total_time = end - start
    # print(f"Gather + Attention Time: {gather_attention_total / 10 * 1000:.2f} ms, FlashInfer Time: {flashinfer_total /10 * 1000:.2f} ms")
    # print(f"Gather Time: {gather_total / 10 * 1000:.2f} ms")
    # print(f"Mixed Attention Kernel Time: {total_time / 10 * 1000:.2f} ms")
    return total_time / 10 * 1000, gather_attention_total / 10 * 1000, flashinfer_total / 10 * 1000

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # full pages
    # avg_time = profile_flashinfer(
    #     num_pages=15300,
    #     selected_pages=175,
    #     page_size_full=8,
    #     page_size=8,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # print(f"FlashInfer Time: {avg_time:.2f} ms")
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # scatter tokens
    # avg_time = profile_flashinfer(
    #     num_pages=128000,
    #     selected_pages=512,
    #     page_size_full=1,
    #     page_size=1,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # print(f"FlashInfer Time: {avg_time:.2f} ms")
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # mixed pages
    # profile_flashinfer_mixed(
    #     num_pages=15300,
    #     selected_pages_full=175,
    #     page_size_full=8,
    #     selected_pages_scatter=510,
    #     page_size_scatter=1,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # gather attention - full pages
    # profile_gather_attention(
    #     n_centroids=7680,
    #     nprobe=175,
    #     page_size=8,
    #     cache_cluster_num=0,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # gather attention - scatter pages
    # profile_gather_attention(
    #     n_centroids=7680,
    #     nprobe=510,
    #     page_size=1,
    #     cache_cluster_num=0,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # gather attention - mixed pages
    # profile_gather_attention_mixed(
    #     n_centroids=7680,
    #     nprobe=138,
    #     page_size_full=8,
    #     page_size_scatter=1,
    #     cache_cluster_num=0,
    #     batch_size=1,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # gather attention - mixed pages with cache
    # profile_gather_attention_mixed_cache(
    #     n_centroids=7680,
    #     nprobe=138,
    #     cache_cluster_num=414,
    #     cache_hit_rate=0.9,
    #     num_pages=15300,
    #     selected_pages_full=175,
    #     page_size_full=8,
    #     selected_pages_scatter=510,
    #     page_size_scatter=1,
    #     batch_size=32,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # mixed attention
    # profile_mixed_cache(
    #     n_centroids=7680,
    #     nprobe=138,
    #     cache_cluster_num=414,
    #     cache_hit_rate=0.9,
    #     num_pages=15300,
    #     selected_pages_full=175,
    #     page_size_full=8,
    #     selected_pages_scatter=510,
    #     page_size_scatter=1,
    #     batch_size=8,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()

    # mixed2 attention
    # profile_mixed2_cache(
    #     n_centroids=7680,
    #     nprobe=138,
    #     cache_cluster_num=414,
    #     cache_hit_rate=0.9,
    #     num_pages=15300,
    #     selected_pages_full=175,
    #     page_size_full=8,
    #     selected_pages_scatter=510,
    #     page_size_scatter=1,
    #     batch_size=32,
    #     kv_head=8,
    #     head_dim=128,
    #     dtype=torch.bfloat16,
    #     device='cpu',
    # )
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
