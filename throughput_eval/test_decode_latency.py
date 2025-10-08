import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import time
import numpy as np
import argparse


class DecodeAttention(nn.Module):
    """Scaled dot-product attention module for decode phase"""

    def __init__(self, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, q, k, v):
        """
        Args:
            q: (batch, num_heads, 1, head_dim)
            k: (batch, num_heads, seq_len, head_dim)
            v: (batch, num_heads, seq_len, head_dim)
        Returns:
            output: (batch, num_heads, 1, head_dim)
            lse: log-sum-exp of attention scores (batch, num_heads, 1)
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, 1, seq_len)

        # Compute log-sum-exp for numerical stability tracking
        lse = torch.logsumexp(scores, dim=-1)  # (batch, num_heads, 1)

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, num_heads, 1, head_dim)

        return output, lse


# args
parser = argparse.ArgumentParser("Decode attention latency test", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="bfloat16",
    help="weight dtype",
)
parser.add_argument(
    "--kv-length", default=1024, type=int, help="KV cache length (context length)"
)
parser.add_argument(
    "--num-heads", default=32, type=int, help="number of attention heads"
)
parser.add_argument(
    "--num-kv-heads", default=8, type=int, help="number of KV heads (for GQA)"
)
parser.add_argument(
    "--head-dim", default=128, type=int, help="head dimension"
)
parser.add_argument(
    "--batch-size", default=1, type=int, help="batch size"
)
parser.add_argument(
    "--warmup-iters", default=3, type=int, help="number of warmup iterations"
)
parser.add_argument(
    "--measure-iters", default=10, type=int, help="number of measurement iterations"
)
args = parser.parse_args()
print(args)

# dtype
dtype = getattr(torch, args.dtype)
amp_enabled = args.dtype != "float32"

# Simulate decode attention: Q is (batch, 1, num_heads, head_dim), K/V are (batch, seq_len, num_kv_heads, head_dim)
batch_size = args.batch_size
seq_len = args.kv_length
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
head_dim = args.head_dim

print(f"\nTest configuration:")
print(f"  Batch size: {batch_size}")
print(f"  KV cache length: {seq_len}")
print(f"  Num heads: {num_heads}")
print(f"  Num KV heads: {num_kv_heads}")
print(f"  Head dim: {head_dim}")
print(f"  Dtype: {dtype}")

# Create attention module
attention_module = DecodeAttention(num_heads, num_kv_heads, head_dim)
attention_module.eval()
attention_module = ipex.optimize(attention_module, dtype=torch.bfloat16)
attention_module = torch.compile(attention_module, backend="ipex")

# Create tensors for decode phase (query length = 1)
q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=dtype)
k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype)
v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype)

# Expand KV for grouped query attention if needed
if num_kv_heads < num_heads:
    num_groups = num_heads // num_kv_heads
    k = k.repeat_interleave(num_groups, dim=2)
    v = v.repeat_interleave(num_groups, dim=2)

# Reshape for attention computation
# (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
q = q.transpose(1, 2)  # (batch, num_heads, 1, head_dim)
k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

# Warmup
print(f"\nWarming up with {args.warmup_iters} iterations...")
with torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
    for _ in range(args.warmup_iters):
        _ = attention_module(q, k, v)

# Measure
print(f"Measuring attention latency over {args.measure_iters} iterations...")
latencies = []

with torch.inference_mode(), torch.cpu.amp.autocast(enabled=amp_enabled):
    for i in range(args.measure_iters):
        start_time = time.time()
        output, lse = attention_module(q, k, v)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)

# Statistics
latencies = np.array(latencies)
print("\n" + "="*60)
print("DECODE ATTENTION LATENCY RESULTS (CPU with Intel Extension)")
print("="*60)
print(f"KV cache length: {seq_len}")
print(f"Batch size: {batch_size}")
print(f"Num heads: {num_heads}")
print(f"Num KV heads: {num_kv_heads}")
print(f"Head dim: {head_dim}")
print(f"Dtype: {dtype}")
print(f"\nAttention latency per layer:")
print(f"  Mean: {np.mean(latencies):.3f} ms")
print(f"  Median: {np.median(latencies):.3f} ms")
print(f"  Min: {np.min(latencies):.3f} ms")
print(f"  Max: {np.max(latencies):.3f} ms")
print(f"  Std: {np.std(latencies):.3f} ms")
print(f"  P95: {np.percentile(latencies, 95):.3f} ms")
print(f"  P99: {np.percentile(latencies, 99):.3f} ms")
print("="*60)
