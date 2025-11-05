import torch
import matplotlib.pyplot as plt
import numpy as np

# Import profiling functions from profiling.py
from profiling import (
    profile_flashinfer,
    profile_flashinfer_mixed,
    profile_gather_attention,
    profile_gather_attention_mixed,
    profile_gather_attention_mixed_cache,
    profile_mixed_cache,
    profile_mixed2_cache
)


def main():
    # FlashInfer Configuration 1: Full pages
    flashinfer_config1 = {
        'num_pages': 15300,
        'selected_pages': 175,
        'page_size_full': 8,
        'page_size': 8,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # FlashInfer Configuration 2: Scatter tokens
    flashinfer_config2 = {
        'num_pages': 130000,
        'selected_pages': 510,
        'page_size_full': 8,
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

    # FlashInfer Configuration 4: mixed pages with cache
    flashinfer_config4 = {
        'n_centroids': 7680,
        'nprobe': 138,
        'cache_cluster_num': 414,
        'cache_hit_rate': 0.9,
        'num_pages': 15300,
        'selected_pages_full': 175,
        'page_size_full': 8,
        'selected_pages_scatter': 510,
        'page_size_scatter': 1,
        'kv_head': 8,
        'head_dim': 128,
        'dtype': torch.bfloat16,
    }

    # Gather Attention Configuration 4: mixed pages with cache
    gather_config4 = {
        'n_centroids': 7680,
        'nprobe': 138,
        'cache_cluster_num': 414,
        'cache_hit_rate': 0.9,
        'num_pages': 15300,
        'selected_pages_full': 175,
        'page_size_full': 8,
        'selected_pages_scatter': 510,
        'page_size_scatter': 1,
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
        'flashinfer_cpu_config4': [],
        'flashinfer_cuda_config1': [],
        'flashinfer_cuda_config2': [],
        'flashinfer_cuda_config3': [],
        'flashinfer_cuda_config4': [],
        'gather_cpu_config1': [],
        'gather_cpu_config2': [],
        'gather_cpu_config3': [],
        'gather_cpu_config4': [],
        'gather_cuda_config1': [],
        'gather_cuda_config2': [],
        'gather_cuda_config3': [],
        'gather_cuda_config4': [],
        # Breakdown for gather attention
        'gather_cpu_config1_gather': [],
        'gather_cpu_config1_attn': [],
        'gather_cpu_config2_gather': [],
        'gather_cpu_config2_attn': [],
        'gather_cpu_config3_gather': [],
        'gather_cpu_config3_attn': [],
        'gather_cpu_config4_gather': [],
        'gather_cpu_config4_attn': [],
        'gather_cuda_config1_gather': [],
        'gather_cuda_config1_attn': [],
        'gather_cuda_config2_gather': [],
        'gather_cuda_config2_attn': [],
        'gather_cuda_config3_gather': [],
        'gather_cuda_config3_attn': [],
        'gather_cuda_config4_gather': [],
        'gather_cuda_config4_attn': [],
        # Breakdown for config4 mixed cache (includes both gather and flashinfer)
        'config4_cpu_gather': [],
        'config4_cpu_attn': [],
        'config4_cpu_flashinfer': [],
        # Breakdown for config4 mixed2 cache (includes both gather and flashinfer)
        'mixed2_cpu_config4': [],
        'config4_cpu_mixed2_gather': [],
        'config4_cpu_mixed2_attn': [],
        'config4_cpu_mixed2_flashinfer': [],
    }

    print("Profiling FlashInfer and Gather Attention with different configurations...")
    print("=" * 80)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 80)

        # # FlashInfer CPU Config 1
        # try:
        #     print(f"  FlashInfer CPU - Config 1 (page_size={flashinfer_config1['page_size']})... ", end='', flush=True)
        #     time_val = profile_flashinfer(batch_size=batch_size, device='cpu', **flashinfer_config1)
        #     results['flashinfer_cpu_config1'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cpu_config1'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # FlashInfer CPU Config 2
        # try:
        #     print(f"  FlashInfer CPU - Config 2 (page_size={flashinfer_config2['page_size']})... ", end='', flush=True)
        #     time_val = profile_flashinfer(batch_size=batch_size, device='cpu', **flashinfer_config2)
        #     results['flashinfer_cpu_config2'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cpu_config2'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # FlashInfer CUDA Config 1
        # try:
        #     print(f"  FlashInfer CUDA - Config 1 (page_size={flashinfer_config1['page_size']})... ", end='', flush=True)
        #     time_val = profile_flashinfer(batch_size=batch_size, device='cuda', **flashinfer_config1)
        #     results['flashinfer_cuda_config1'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cuda_config1'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # FlashInfer CUDA Config 2
        # try:
        #     print(f"  FlashInfer CUDA - Config 2 (page_size={flashinfer_config2['page_size']})... ", end='', flush=True)
        #     time_val = profile_flashinfer(batch_size=batch_size, device='cuda', **flashinfer_config2)
        #     results['flashinfer_cuda_config2'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cuda_config2'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CPU Config 1
        # try:
        #     print(f"  Gather Attn CPU - Config 1 (page_size={gather_config1['page_size']})... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cpu', **gather_config1)
        #     results['gather_cpu_config1'].append(total_time)
        #     results['gather_cpu_config1_gather'].append(gather_time)
        #     results['gather_cpu_config1_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cpu_config1'].append(None)
        #     results['gather_cpu_config1_gather'].append(None)
        #     results['gather_cpu_config1_attn'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CPU Config 2
        # try:
        #     print(f"  Gather Attn CPU - Config 2 (page_size={gather_config2['page_size']})... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cpu', **gather_config2)
        #     results['gather_cpu_config2'].append(total_time)
        #     results['gather_cpu_config2_gather'].append(gather_time)
        #     results['gather_cpu_config2_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cpu_config2'].append(None)
        #     results['gather_cpu_config2_gather'].append(None)
        #     results['gather_cpu_config2_attn'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CUDA Config 1
        # try:
        #     print(f"  Gather Attn CUDA - Config 1 (page_size={gather_config1['page_size']})... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cuda', **gather_config1)
        #     results['gather_cuda_config1'].append(total_time)
        #     results['gather_cuda_config1_gather'].append(gather_time)
        #     results['gather_cuda_config1_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cuda_config1'].append(None)
        #     results['gather_cuda_config1_gather'].append(None)
        #     results['gather_cuda_config1_attn'].append(None)

        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CUDA Config 2
        # try:
        #     print(f"  Gather Attn CUDA - Config 2 (page_size={gather_config2['page_size']})... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention(batch_size=batch_size, device='cuda', **gather_config2)
        #     results['gather_cuda_config2'].append(total_time)
        #     results['gather_cuda_config2_gather'].append(gather_time)
        #     results['gather_cuda_config2_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cuda_config2'].append(None)
        #     results['gather_cuda_config2_gather'].append(None)
        #     results['gather_cuda_config2_attn'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # FlashInfer CPU Config 3
        # try:
        #     print(f"  FlashInfer CPU - Config 3 (mixed)... ", end='', flush=True)
        #     time_val = profile_flashinfer_mixed(batch_size=batch_size, device='cpu', **flashinfer_config3)
        #     results['flashinfer_cpu_config3'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cpu_config3'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # FlashInfer CUDA Config 3
        # try:
        #     print(f"  FlashInfer CUDA - Config 3 (mixed)... ", end='', flush=True)
        #     time_val = profile_flashinfer_mixed(batch_size=batch_size, device='cuda', **flashinfer_config3)
        #     results['flashinfer_cuda_config3'].append(time_val)
        #     print(f"{time_val:.2f} ms")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['flashinfer_cuda_config3'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CPU Config 3
        # try:
        #     print(f"  Gather Attn CPU - Config 3 (mixed)... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention_mixed(batch_size=batch_size, device='cpu', **gather_config3)
        #     results['gather_cpu_config3'].append(total_time)
        #     results['gather_cpu_config3_gather'].append(gather_time)
        #     results['gather_cpu_config3_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cpu_config3'].append(None)
        #     results['gather_cpu_config3_gather'].append(None)
        #     results['gather_cpu_config3_attn'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # # Gather Attention CUDA Config 3
        # try:
        #     print(f"  Gather Attn CUDA - Config 3 (mixed)... ", end='', flush=True)
        #     total_time, gather_time, attn_time = profile_gather_attention_mixed(batch_size=batch_size, device='cuda', **gather_config3)
        #     results['gather_cuda_config3'].append(total_time)
        #     results['gather_cuda_config3_gather'].append(gather_time)
        #     results['gather_cuda_config3_attn'].append(attn_time)
        #     print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        # except Exception as e:
        #     print(f"FAILED: {e}")
        #     results['gather_cuda_config3'].append(None)
        #     results['gather_cuda_config3_gather'].append(None)
        #     results['gather_cuda_config3_attn'].append(None)
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # FlashInfer CPU Config 4 (cache - CPU only)
        try:
            print(f"  FlashInfer CPU - Config 4 (mixed+cache)... ", end='', flush=True)
            total_time, gather_attn_time, flashinfer_time = profile_mixed_cache(batch_size=batch_size, device='cpu', **flashinfer_config4)
            results['flashinfer_cpu_config4'].append(total_time)
            results['config4_cpu_gather'].append(gather_attn_time)
            results['config4_cpu_attn'].append(None)  # Combined in gather_attn_time
            results['config4_cpu_flashinfer'].append(flashinfer_time)
            print(f"{total_time:.2f} ms (gather+attn: {gather_attn_time:.2f}, flashinfer: {flashinfer_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['flashinfer_cpu_config4'].append(None)
            results['config4_cpu_gather'].append(None)
            results['config4_cpu_attn'].append(None)
            results['config4_cpu_flashinfer'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Mixed2 CPU Config 4 (cache - CPU only)
        try:
            print(f"  Mixed2 CPU - Config 4 (mixed+cache)... ", end='', flush=True)
            total_time, gather_attn_time, flashinfer_time = profile_mixed2_cache(batch_size=batch_size, device='cpu', **flashinfer_config4)
            results['mixed2_cpu_config4'].append(total_time)
            results['config4_cpu_mixed2_gather'].append(gather_attn_time)
            results['config4_cpu_mixed2_attn'].append(None)  # Combined in gather_attn_time
            results['config4_cpu_mixed2_flashinfer'].append(flashinfer_time)
            print(f"{total_time:.2f} ms (gather+attn: {gather_attn_time:.2f}, flashinfer: {flashinfer_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['mixed2_cpu_config4'].append(None)
            results['config4_cpu_mixed2_gather'].append(None)
            results['config4_cpu_mixed2_attn'].append(None)
            results['config4_cpu_mixed2_flashinfer'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Gather Attention CPU Config 4 (cache - CPU only)
        try:
            print(f"  Gather Attn CPU - Config 4 (mixed+cache)... ", end='', flush=True)
            total_time, gather_time, attn_time = profile_gather_attention_mixed_cache(batch_size=batch_size, device='cpu', **gather_config4)
            results['gather_cpu_config4'].append(total_time)
            results['gather_cpu_config4_gather'].append(gather_time)
            results['gather_cpu_config4_attn'].append(attn_time)
            print(f"{total_time:.2f} ms (gather: {gather_time:.2f}, attn: {attn_time:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results['gather_cpu_config4'].append(None)
            results['gather_cpu_config4_gather'].append(None)
            results['gather_cpu_config4_attn'].append(None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Create a single figure comparing all three cache baselines
    print("\n" + "=" * 80)
    print("Generating comparison plot...")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for stacked bar chart
    x = np.arange(len(batch_sizes))  # Label locations
    width = 0.25  # Width of bars

    # Extract data for all three baselines
    fi_gather = [results['config4_cpu_gather'][i] if results['config4_cpu_gather'][i] is not None else 0 for i in range(len(batch_sizes))]
    fi_flashinfer = [results['config4_cpu_flashinfer'][i] if results['config4_cpu_flashinfer'][i] is not None else 0 for i in range(len(batch_sizes))]

    mixed2_gather = [results['config4_cpu_mixed2_gather'][i] if results['config4_cpu_mixed2_gather'][i] is not None else 0 for i in range(len(batch_sizes))]
    mixed2_flashinfer = [results['config4_cpu_mixed2_flashinfer'][i] if results['config4_cpu_mixed2_flashinfer'][i] is not None else 0 for i in range(len(batch_sizes))]

    ga_gather = [results['gather_cpu_config4_gather'][i] if results['gather_cpu_config4_gather'][i] is not None else 0 for i in range(len(batch_sizes))]
    ga_attn = [results['gather_cpu_config4_attn'][i] if results['gather_cpu_config4_attn'][i] is not None else 0 for i in range(len(batch_sizes))]

    # Create stacked bars for each baseline
    # FlashInfer Mixed: Gather+Attn at bottom, FlashInfer on top
    ax.bar(x - width, fi_gather, width, label='FlashInfer Mixed (Gather+Attn)', color='#1f77b4', alpha=0.8)
    ax.bar(x - width, fi_flashinfer, width, bottom=fi_gather, label='FlashInfer Mixed (FlashInfer)', color='#aec7e8', alpha=0.8)

    # Mixed2: Gather+Attn at bottom, FlashInfer on top
    ax.bar(x, mixed2_gather, width, label='Mixed2 (Gather+Attn)', color='#ff7f0e', alpha=0.8)
    ax.bar(x, mixed2_flashinfer, width, bottom=mixed2_gather, label='Mixed2 (FlashInfer)', color='#ffbb78', alpha=0.8)

    # Gather Attention: Gather at bottom, Attention on top
    ax.bar(x + width, ga_gather, width, label='Gather Attn (Gather)', color='#2ca02c', alpha=0.8)
    ax.bar(x + width, ga_attn, width, bottom=ga_gather, label='Gather Attn (Attention)', color='#98df8a', alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('CPU - Config 4 (mixed+cache): All Three Cache Baselines Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig('cpu_config4_all_baselines_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: cpu_config4_all_baselines_comparison.png")
    plt.close(fig)

    print("=" * 80)
    print("Comparison plot saved!")

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
