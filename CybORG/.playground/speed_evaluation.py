import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

# Define dummy encoders with matching architectures
class DummyTransformer(torch.nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        cls_tokens = self.cls.expand(x.size(0), -1, -1)
        return self.transformer(torch.cat([cls_tokens, x], dim=1))[:, 0, :]

class DummyDeepSets(torch.nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.phi = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(torch.nn.Linear(d_model*2, d_model), torch.nn.ReLU())

    def forward(self, x):
        h_local = self.phi(x)
        h_global = torch.sum(h_local, dim=1, keepdim=True).expand(-1, x.size(1), -1)
        h_nodes = self.rho(torch.cat([h_local, h_global], dim=-1))
        return torch.max(h_nodes, dim=1)[0]


def run_single_device_benchmark(device: torch.device, N_sizes, n_iters=100) -> dict:
    """Runs the benchmark on the specified device and returns the ms timings."""
    print(f"--- Benchmarking on {device.type.upper()} ---")
    
    transformer = DummyTransformer().to(device).eval()
    deepsets = DummyDeepSets().to(device).eval()
    
    trans_ms = []
    ds_ms = []
    
    for n in N_sizes:
        x = torch.randn(1, n, 128, device=device) # Batch size 1
        
        # Warmup
        with torch.no_grad():
            for _ in range(10): 
                transformer(x)
                deepsets(x)
                
        # Transformer Timing
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_iters): transformer(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        trans_ms.append(((time.time() - t0) / n_iters) * 1000)
        
        # DeepSets Timing
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_iters): deepsets(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        ds_ms.append(((time.time() - t0) / n_iters) * 1000)
        
        print(f"N={n:<4} | Trans: {trans_ms[-1]:.2f}ms | DS: {ds_ms[-1]:.2f}ms")
        
    return {"trans": trans_ms, "ds": ds_ms}


def plot_combined_results(cpu_res, gpu_res, N_sizes):
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 1. Plot GPU Results (Solid Lines, filled markers)
    if gpu_res:
        ax.plot(N_sizes, gpu_res["trans"], label="Transformer (GPU)", 
                color="black", linestyle="-", marker="o", markersize=7)
        ax.plot(N_sizes, gpu_res["ds"], label="Deep Sets (GPU)", 
                color="gray", linestyle="-", marker="s", markersize=7)
                
    # 2. Plot CPU Results (Dashed Lines, open markers)
    if cpu_res:
        ax.plot(N_sizes, cpu_res["trans"], label="Transformer (CPU)", 
                color="black", linestyle="--", marker="o", markerfacecolor="white", markersize=7)
        ax.plot(N_sizes, cpu_res["ds"], label="Deep Sets (CPU)", 
                color="gray", linestyle="--", marker="s", markerfacecolor="white", markersize=7)
    
    # 3. Reference Line
    ax.axhline(y=10, color='red', linestyle=':', label='10ms Real-Time Threshold')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Hosts (N)")
    ax.set_ylabel("Forward Pass Latency (ms)")
    ax.set_title("Inference Latency: CPU vs GPU Scaling")
    
    ax.set_xticks([10, 50, 100, 500, 1000, 5000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Move legend outside the plot if it gets too crowded
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig("cpu_gpu_latency_scaling.png", dpi=300, bbox_inches='tight')
    print("Saved plot to cpu_gpu_latency_scaling.png")
    plt.show()

if __name__ == "__main__":
    N_sizes = [10, 50, 100, 250, 500, 1000, 2000, 5000]
    
    # Run CPU Benchmark
    cpu_device = torch.device("cpu")
    cpu_results = run_single_device_benchmark(cpu_device, N_sizes)
    
    # Run GPU Benchmark (if available)
    gpu_results = None
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_results = run_single_device_benchmark(gpu_device, N_sizes)
    else:
        print("Warning: No CUDA GPU detected. Only plotting CPU results.")
        
    plot_combined_results(cpu_results, gpu_results, N_sizes)
