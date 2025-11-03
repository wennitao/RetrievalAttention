import time
import torch
import flashinfer
from retroinfer_kernels import gather_copy_and_concat
from weighted_flash_decoding import weighted_flash_decoding
import matplotlib.pyplot as plt
import numpy as np

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
                # Random chunk size between 1 and min(15, remaining)
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
    torch.cuda.empty_cache()

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

        attn_out, lse_out = decode_wrapper_scatter.run(
            queries,
            (keys_scatter, values_scatter),
            return_lse=True
        )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

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
    torch.cuda.empty_cache()

    # profile - measure gather and attention separately
    gather_time_total = 0
    attn_time_total = 0

    for it in range(10):
        # Measure gather time
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
        gather_time_total += (end_gather - start_gather)

        # Measure attention time
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
        attn_time_total += (end_attn - start_attn)

    total_time = (gather_time_total + attn_time_total) / 10 * 1000
    gather_time = gather_time_total / 10 * 1000
    attn_time = attn_time_total / 10 * 1000

    return total_time, gather_time, attn_time

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
    torch.cuda.empty_cache()

    # profile - measure gather and attention separately
    gather_time_total = 0
    attn_time_total = 0

    # profile
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
        gather_time_total += (end_gather - start_gather)

        # Measure attention time
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
        attn_time_total += (end_attn - start_attn)
    
    total_time = (gather_time_total + attn_time_total) / 10 * 1000
    gather_time = gather_time_total / 10 * 1000
    attn_time = attn_time_total / 10 * 1000

    return total_time, gather_time, attn_time

def main():
    # FlashInfer Configuration 1: Full pages
    flashinfer_config1 = {
        'num_pages': 15300,
        'selected_pages': 175,
        'page_size': 8,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # FlashInfer Configuration 2: Scatter tokens
    flashinfer_config2 = {
        'num_pages': 130000,
        'selected_pages': 510,
        'page_size': 1,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # FlashInfer Configuration 3: mixed pages
    flashinfer_config3 = {
        'num_pages': 15300,
        'selected_pages_full': 175,
        'page_size_full': 8,
        'selected_pages_scatter': 510,
        'page_size_scatter': 1,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Gather Attention Configuration 1: Full pages
    gather_config1 = {
        'n_centroids': 7680,
        'nprobe': 175,
        'page_size': 8,
        'cache_cluster_num': 0,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Gather Attention Configuration 2: Scatter tokens
    gather_config2 = {
        'n_centroids': 7680,
        'nprobe': 510,
        'page_size': 1,
        'cache_cluster_num': 0,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Gather Attention Configuration 3: mixed pages
    gather_config3 = {
        'n_centroids': 7680,
        'nprobe': 138,
        'page_size_full': 8,
        'page_size_scatter': 1,
        'cache_cluster_num': 0,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Storage for results
    results = {
        'flashinfer_cpu_config1': [],
        'flashinfer_cpu_config2': [],
        'flashinfer_cpu_config3': [],
        'flashinfer_cuda_config1': [],
        'flashinfer_cuda_config2': [],
        'flashinfer_cuda_config3': [],
        'gather_cpu_config1': [],
        'gather_cpu_config2': [],
        'gather_cpu_config3': [],
        'gather_cuda_config1': [],
        'gather_cuda_config2': [],
        'gather_cuda_config3': [],
        # Breakdown for gather attention
        'gather_cpu_config1_gather': [],
        'gather_cpu_config1_attn': [],
        'gather_cpu_config2_gather': [],
        'gather_cpu_config2_attn': [],
        'gather_cpu_config3_gather': [],
        'gather_cpu_config3_attn': [],
        'gather_cuda_config1_gather': [],
        'gather_cuda_config1_attn': [],
        'gather_cuda_config2_gather': [],
        'gather_cuda_config2_attn': [],
        'gather_cuda_config3_gather': [],
        'gather_cuda_config3_attn': [],
    }

    print("Profiling FlashInfer and Gather Attention with different configurations...")
    print("=" * 80)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 80)

        # FlashInfer CPU Config 1
        try:
            print(f"  FlashInfer CPU - Config 1 (page_size={flashinfer_config1['page_size']})... ", end='', flush=True)
            time_val = profile_flashinfer(batch_size=batch_size, device='cpu', **flashinfer_config1)
            results['flashinfer_cpu_config1'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cpu_config1'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # FlashInfer CPU Config 2
        try:
            print(f"  FlashInfer CPU - Config 2 (page_size={flashinfer_config2['page_size']})... ", end='', flush=True)
            time_val = profile_flashinfer(batch_size=batch_size, device='cpu', **flashinfer_config2)
            results['flashinfer_cpu_config2'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cpu_config2'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # FlashInfer CUDA Config 1
        try:
            print(f"  FlashInfer CUDA - Config 1 (page_size={flashinfer_config1['page_size']})... ", end='', flush=True)
            time_val = profile_flashinfer(batch_size=batch_size, device='cuda', **flashinfer_config1)
            results['flashinfer_cuda_config1'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cuda_config1'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # FlashInfer CUDA Config 2
        try:
            print(f"  FlashInfer CUDA - Config 2 (page_size={flashinfer_config2['page_size']})... ", end='', flush=True)
            time_val = profile_flashinfer(batch_size=batch_size, device='cuda', **flashinfer_config2)
            results['flashinfer_cuda_config2'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cuda_config2'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CPU Config 1
        try:
            print(f"  Gather Attn CPU - Config 1 (page_size={gather_config1['page_size']})... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cpu', **gather_config1)
            results['gather_cpu_config1'].append(total_time)
            results['gather_cpu_config1_gather'].append(gather_time)
            results['gather_cpu_config1_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cpu_config1'].append(None)
            results['gather_cpu_config1_gather'].append(None)
            results['gather_cpu_config1_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CPU Config 2
        try:
            print(f"  Gather Attn CPU - Config 2 (page_size={gather_config2['page_size']})... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cpu', **gather_config2)
            results['gather_cpu_config2'].append(total_time)
            results['gather_cpu_config2_gather'].append(gather_time)
            results['gather_cpu_config2_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cpu_config2'].append(None)
            results['gather_cpu_config2_gather'].append(None)
            results['gather_cpu_config2_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CUDA Config 1
        try:
            print(f"  Gather Attn CUDA - Config 1 (page_size={gather_config1['page_size']})... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cuda', **gather_config1)
            results['gather_cuda_config1'].append(total_time)
            results['gather_cuda_config1_gather'].append(gather_time)
            results['gather_cuda_config1_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cuda_config1'].append(None)
            results['gather_cuda_config1_gather'].append(None)
            results['gather_cuda_config1_attn'].append(None)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CUDA Config 2
        try:
            print(f"  Gather Attn CUDA - Config 2 (page_size={gather_config2['page_size']})... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cuda', **gather_config2)
            results['gather_cuda_config2'].append(total_time)
            results['gather_cuda_config2_gather'].append(gather_time)
            results['gather_cuda_config2_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cuda_config2'].append(None)
            results['gather_cuda_config2_gather'].append(None)
            results['gather_cuda_config2_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # FlashInfer CPU Config 3
        try:
            print(f"  FlashInfer CPU - Config 3 (mixed)... ", end='', flush=True)
            time_val = profile_flashinfer_mixed(batch_size=batch_size, device='cpu', **flashinfer_config3)
            results['flashinfer_cpu_config3'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cpu_config3'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # FlashInfer CUDA Config 3
        try:
            print(f"  FlashInfer CUDA - Config 3 (mixed)... ", end='', flush=True)
            time_val = profile_flashinfer_mixed(batch_size=batch_size, device='cuda', **flashinfer_config3)
            results['flashinfer_cuda_config3'].append(time_val)
            print(f"{time_val:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cuda_config3'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CPU Config 3
        try:
            print(f"  Gather Attn CPU - Config 3 (mixed)... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention_mixed(batch_size=batch_size, device='cpu', **gather_config3)
            results['gather_cpu_config3'].append(total_time)
            results['gather_cpu_config3_gather'].append(gather_time)
            results['gather_cpu_config3_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cpu_config3'].append(None)
            results['gather_cpu_config3_gather'].append(None)
            results['gather_cpu_config3_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CUDA Config 3
        try:
            print(f"  Gather Attn CUDA - Config 3 (mixed)... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention_mixed(batch_size=batch_size, device='cuda', **gather_config3)
            results['gather_cuda_config3'].append(total_time)
            results['gather_cuda_config3_gather'].append(gather_time)
            results['gather_cuda_config3_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cuda_config3'].append(None)
            results['gather_cuda_config3_gather'].append(None)
            results['gather_cuda_config3_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Create 6 figures - one for each device/config combination
    # Each figure compares FlashInfer vs Gather Attention
    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)

    plot_configs = [
        # (flashinfer_key, gather_key, gather_gather_key, gather_attn_key, title, filename)
        ('flashinfer_cpu_config1', 'gather_cpu_config1', 'gather_cpu_config1_gather', 'gather_cpu_config1_attn',
         'CPU - Config 1 (page_size=8): FlashInfer vs Gather Attention', 'cpu_config1_comparison.png'),
        ('flashinfer_cpu_config2', 'gather_cpu_config2', 'gather_cpu_config2_gather', 'gather_cpu_config2_attn',
         'CPU - Config 2 (page_size=1): FlashInfer vs Gather Attention', 'cpu_config2_comparison.png'),
        ('flashinfer_cpu_config3', 'gather_cpu_config3', 'gather_cpu_config3_gather', 'gather_cpu_config3_attn',
         'CPU - Config 3 (mixed): FlashInfer vs Gather Attention', 'cpu_config3_comparison.png'),
        ('flashinfer_cuda_config1', 'gather_cuda_config1', 'gather_cuda_config1_gather', 'gather_cuda_config1_attn',
         'CUDA - Config 1 (page_size=8): FlashInfer vs Gather Attention', 'cuda_config1_comparison.png'),
        ('flashinfer_cuda_config2', 'gather_cuda_config2', 'gather_cuda_config2_gather', 'gather_cuda_config2_attn',
         'CUDA - Config 2 (page_size=1): FlashInfer vs Gather Attention', 'cuda_config2_comparison.png'),
        ('flashinfer_cuda_config3', 'gather_cuda_config3', 'gather_cuda_config3_gather', 'gather_cuda_config3_attn',
         'CUDA - Config 3 (mixed): FlashInfer vs Gather Attention', 'cuda_config3_comparison.png'),
    ]

    for fi_key, ga_key, ga_gather_key, ga_attn_key, title, filename in plot_configs:
        fig = plt.figure(figsize=(12, 7))

        # Plot FlashInfer
        fi_data = results[fi_key]
        valid_fi = [(bs, val) for bs, val in zip(batch_sizes, fi_data) if val is not None]
        if valid_fi:
            bs_vals, time_vals = zip(*valid_fi)
            plt.plot(bs_vals, time_vals, marker='o', linestyle='-', linewidth=3,
                    markersize=10, label='FlashInfer', color='C0')

        # Plot Gather Attention - Total
        ga_data = results[ga_key]
        valid_ga = [(bs, val) for bs, val in zip(batch_sizes, ga_data) if val is not None]
        if valid_ga:
            bs_vals, time_vals = zip(*valid_ga)
            plt.plot(bs_vals, time_vals, marker='s', linestyle='-', linewidth=3,
                    markersize=10, label='Gather Attn (Total)', color='C1')

        # Plot Gather Attention - Gather component
        ga_gather_data = results[ga_gather_key]
        valid_ga_gather = [(bs, val) for bs, val in zip(batch_sizes, ga_gather_data) if val is not None]
        if valid_ga_gather:
            bs_vals, time_vals = zip(*valid_ga_gather)
            plt.plot(bs_vals, time_vals, marker='s', linestyle='--', linewidth=2.5,
                    markersize=8, label='Gather Attn (Gather)', color='C2', alpha=0.8)

        # Plot Gather Attention - Attention component
        ga_attn_data = results[ga_attn_key]
        valid_ga_attn = [(bs, val) for bs, val in zip(batch_sizes, ga_attn_data) if val is not None]
        if valid_ga_attn:
            bs_vals, time_vals = zip(*valid_ga_attn)
            plt.plot(bs_vals, time_vals, marker='^', linestyle='--', linewidth=2.5,
                    markersize=8, label='Gather Attn (Attention)', color='C3', alpha=0.8)

        plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
        plt.ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(batch_sizes)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close(fig)

    print("=" * 80)
    print("All comparison plots saved!")

    # All configs for reference
    all_configs = [
        ('flashinfer_cpu_config1', 'FlashInfer CPU - Config 1 (page_size=8)', 'o', '-', 'C0'),
        ('flashinfer_cpu_config2', 'FlashInfer CPU - Config 2 (page_size=1)', 's', '-', 'C1'),
        ('flashinfer_cuda_config1', 'FlashInfer CUDA - Config 1 (page_size=8)', '^', '-', 'C2'),
        ('flashinfer_cuda_config2', 'FlashInfer CUDA - Config 2 (page_size=1)', 'd', '-', 'C3'),
        ('gather_cpu_config1', 'Gather Attn CPU - Config 1 (page_size=8)', 'o', '--', 'C4'),
        ('gather_cpu_config2', 'Gather Attn CPU - Config 2 (page_size=1)', 's', '--', 'C5'),
        ('gather_cuda_config1', 'Gather Attn CUDA - Config 1 (page_size=8)', '^', '--', 'C6'),
        ('gather_cuda_config2', 'Gather Attn CUDA - Config 2 (page_size=1)', 'd', '--', 'C7'),
    ]
    configs = all_configs

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table (all times in ms):")
    print("=" * 80)
    print(f"{'Batch':<6} {'FI-CPU1':<10} {'FI-CPU2':<10} {'FI-CUDA1':<10} {'FI-CUDA2':<10} "
          f"{'GA-CPU1':<10} {'GA-CPU2':<10} {'GA-CUDA1':<10} {'GA-CUDA2':<10}")
    print("-" * 80)
    for i, bs in enumerate(batch_sizes):
        row = f"{bs:<6} "
        for key, _, _, _, _ in configs:
            val = results[key][i]
            row += f"{val:<10.2f} " if val is not None else f"{'FAIL':<10} "
        print(row)
    print("=" * 80)
    print("Legend: FI=FlashInfer, GA=Gather Attention, Config1=page_size 8, Config2=page_size 1")


if __name__ == "__main__":
    main()
