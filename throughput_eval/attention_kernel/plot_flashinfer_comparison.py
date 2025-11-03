import time
import torch
import flashinfer
import matplotlib.pyplot as plt
import numpy as np

def profile_flashinfer(
    num_pages: int,
    selected_pages: int,
    page_size: int,
    batch_size: int,
    kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    """Profile FlashInfer attention kernel with given configuration."""
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
    total_time = (end - start) / 10 * 1000  # Average time in ms

    return total_time


def main():
    # Configuration 1: Full pages
    config1 = {
        'num_pages': 15300,
        'selected_pages': 175,
        'page_size': 8,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Configuration 2: Scatter tokens
    config2 = {
        'num_pages': 130000,
        'selected_pages': 510,
        'page_size': 1,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Storage for results
    results = {
        'cpu_config1': [],
        'cpu_config2': [],
        'cuda_config1': [],
        'cuda_config2': [],
    }

    print("Profiling FlashInfer with different configurations...")
    print("=" * 60)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 60)

        # CPU Config 1
        print(f"  CPU - Config 1 (page_size={config1['page_size']}, selected_pages={config1['selected_pages']})... ", end='', flush=True)
        time1 = profile_flashinfer(batch_size=batch_size, device='cpu', **config1)
        results['cpu_config1'].append(time1)
        print(f"{time1:.2f} ms")
        torch.cuda.empty_cache()

        # CPU Config 2
        print(f"  CPU - Config 2 (page_size={config2['page_size']}, selected_pages={config2['selected_pages']})... ", end='', flush=True)
        time2 = profile_flashinfer(batch_size=batch_size, device='cpu', **config2)
        results['cpu_config2'].append(time2)
        print(f"{time2:.2f} ms")
        torch.cuda.empty_cache()

        # CUDA Config 1
        print(f"  CUDA - Config 1 (page_size={config1['page_size']}, selected_pages={config1['selected_pages']})... ", end='', flush=True)
        time3 = profile_flashinfer(batch_size=batch_size, device='cuda', **config1)
        results['cuda_config1'].append(time3)
        print(f"{time3:.2f} ms")
        torch.cuda.empty_cache()

        # CUDA Config 2
        print(f"  CUDA - Config 2 (page_size={config2['page_size']}, selected_pages={config2['selected_pages']})... ", end='', flush=True)
        time4 = profile_flashinfer(batch_size=batch_size, device='cuda', **config2)
        results['cuda_config2'].append(time4)
        print(f"{time4:.2f} ms")
        torch.cuda.empty_cache()

    # Create plot
    plt.figure(figsize=(12, 8))

    plt.plot(batch_sizes, results['cpu_config1'], marker='o', linewidth=2, markersize=8,
             label=f"CPU - Config 1 (page_size={config1['page_size']}, selected_pages={config1['selected_pages']})")
    plt.plot(batch_sizes, results['cpu_config2'], marker='s', linewidth=2, markersize=8,
             label=f"CPU - Config 2 (page_size={config2['page_size']}, selected_pages={config2['selected_pages']})")
    plt.plot(batch_sizes, results['cuda_config1'], marker='^', linewidth=2, markersize=8,
             label=f"CUDA - Config 1 (page_size={config1['page_size']}, selected_pages={config1['selected_pages']})")
    plt.plot(batch_sizes, results['cuda_config2'], marker='d', linewidth=2, markersize=8,
             label=f"CUDA - Config 2 (page_size={config2['page_size']}, selected_pages={config2['selected_pages']})")

    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.title('FlashInfer Performance: CPU vs CUDA, Config 1 vs Config 2', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(batch_sizes)

    # Use log scale if values vary significantly
    if max(max(results.values(), key=max)) / min(min(results.values(), key=min)) > 10:
        plt.yscale('log')
        plt.ylabel('Execution Time (ms) - Log Scale', fontsize=14)

    plt.tight_layout()
    plt.savefig('flashinfer_comparison.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Plot saved as 'flashinfer_comparison.png'")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table:")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'CPU-Config1':<12} {'CPU-Config2':<12} {'CUDA-Config1':<12} {'CUDA-Config2':<12}")
    print("-" * 60)
    for i, bs in enumerate(batch_sizes):
        print(f"{bs:<12} {results['cpu_config1'][i]:<12.2f} {results['cpu_config2'][i]:<12.2f} "
              f"{results['cuda_config1'][i]:<12.2f} {results['cuda_config2'][i]:<12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
