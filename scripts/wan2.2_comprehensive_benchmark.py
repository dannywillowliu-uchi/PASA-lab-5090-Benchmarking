#!/usr/bin/env python3
"""
WAN 2.2 Comprehensive Benchmark Suite
Runs WAN2.2 14B benchmarks across different step counts and GPU configurations.
Tests: 10 steps vs 20 steps, and 1-8 GPU configurations.
"""

import subprocess
import time
import json
import os
import argparse
from datetime import datetime

def run_single_benchmark(steps, gpu_count, run_id):
    """Run a single benchmark configuration"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "/workspace/results"
    
    # Construct GPU specification for Docker
    if gpu_count == 1:
        gpu_spec = "device=0"
        docker_gpu_args = ["--gpus", "device=0"]
    else:
        gpu_spec = "all"  # Use all available GPUs
        docker_gpu_args = ["--gpus", "all"]
    
    # Docker command for the benchmark
    cmd = [
        "sudo", "docker", "run", "--rm"
    ] + docker_gpu_args + [
        "-v", "/root/gpu_benchmarking/models/Wan2.2-T2V-A14B:/workspace/models",
        "-v", "/root/gpu_benchmarking/results:/workspace/results",
        "-v", "/root/gpu_benchmarking/scripts/wan2.2_benchmark.py:/workspace/wan2.2_benchmark.py",
        "wan2.2-benchmark:simplified",
        "python3", "/workspace/wan2.2_benchmark.py",
        "--sample_steps", str(steps)
    ]
    
    # Add multi-GPU parameters if needed
    if gpu_count > 1:
        cmd.extend([
            "--multi_gpu",
            "--nproc_per_node", str(gpu_count),
            "--gpu_id", "0"
        ])
    else:
        cmd.extend([
            "--offload_model",  # Use model offloading for single GPU
            "--gpu_id", "0"
        ])
    
    print(f"\n[INFO] Running benchmark {run_id}: {steps} steps, {gpu_count} GPU(s)")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        total_time = end_time - start_time
        success = True
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        total_time = end_time - start_time
        success = False
        stdout = e.stdout if hasattr(e, 'stdout') else ''
        stderr = e.stderr if hasattr(e, 'stderr') else str(e)
    
    # Comprehensive result
    result_data = {
        "benchmark_id": run_id,
        "model": "wan2.2-t2v-a14b-14B",
        "steps": steps,
        "gpu_count": gpu_count,
        "gpu_spec": gpu_spec,
        "resolution": "480*832",
        "frame_count": "default",
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "success": success,
        "execution_time_seconds": round(total_time, 2),
        "timestamp": timestamp,
        "docker_stdout": stdout,
        "docker_stderr": stderr
    }
    
    # Save individual result
    result_file = f"/root/gpu_benchmarking/results/comprehensive_wan22_run_{run_id}_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"[INFO] Run {run_id} completed in {total_time:.2f}s - Success: {success}")
    
    # Try to extract detailed metrics from the individual result file
    try:
        # Parse the JSON output from the benchmark to get detailed metrics
        import re
        docker_output = stdout
        if 'Results saved to:' in docker_output:
            # Extract the result file path
            result_match = re.search(r'Results saved to: (/workspace/results/[^\s]+\.json)', docker_output)
            if result_match:
                result_path = result_match.group(1).replace('/workspace/results/', '/root/gpu_benchmarking/results/')
                if os.path.exists(result_path):
                    with open(result_path, 'r') as f:
                        detailed_results = json.load(f)
                    
                    # Extract key metrics
                    result_data.update({
                        "pure_generation_time_seconds": detailed_results.get("pure_generation_time_seconds"),
                        "model_loading_time_seconds": detailed_results.get("model_loading_time_seconds"),
                        "memory_analysis": detailed_results.get("memory_analysis", {}),
                        "memory_samples": detailed_results.get("memory_monitoring_samples", 0)
                    })
                    
                    print(f"[METRICS] Generation: {detailed_results.get('pure_generation_time_seconds', 'N/A')}s, "
                          f"Loading: {detailed_results.get('model_loading_time_seconds', 'N/A')}s")
    except Exception as e:
        print(f"[WARNING] Could not extract detailed metrics: {e}")
    
    return result_data

def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    
    print("=" * 80)
    print("WAN 2.2 14B Comprehensive Benchmark Suite")
    print("=" * 80)
    
    # Test configurations - 1 GPU vs 8 GPU only
    configs = [
        # Step count comparisons (single GPU)
        {"steps": 10, "gpu_count": 1, "description": "10 steps, 1 GPU"},
        {"steps": 20, "gpu_count": 1, "description": "20 steps, 1 GPU"},
        
        # GPU scaling tests (8 GPUs)
        {"steps": 10, "gpu_count": 8, "description": "10 steps, 8 GPUs"},
        {"steps": 20, "gpu_count": 8, "description": "20 steps, 8 GPUs"},
    ]
    
    results = []
    total_start = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n[BENCHMARK {i}/{len(configs)}] {config['description']}")
        
        result = run_single_benchmark(
            steps=config["steps"],
            gpu_count=config["gpu_count"], 
            run_id=f"{i:02d}_{config['steps']}steps_{config['gpu_count']}gpu"
        )
        results.append(result)
        
        # Brief pause between runs to clear GPU memory
        print("[INFO] Waiting 30 seconds before next benchmark...")
        time.sleep(30)
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    # Generate summary report
    summary = {
        "suite": "WAN2.2-14B-Comprehensive-Benchmark",
        "total_duration_seconds": round(total_duration, 2),
        "total_runs": len(configs),
        "successful_runs": sum(1 for r in results if r["success"]),
        "failed_runs": sum(1 for r in results if not r["success"]),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results": results
    }
    
    # Save summary
    summary_file = f"/root/gpu_benchmarking/results/wan22_comprehensive_summary_{summary['timestamp']}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)
    print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Successful Runs: {summary['successful_runs']}/{len(configs)}")
    print(f"Summary saved to: {summary_file}")
    
    # Print performance comparison
    print("\nPERFORMANCE SUMMARY:")
    print("-" * 50)
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['benchmark_id']}: {result['execution_time_seconds']}s")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN 2.2 Comprehensive Benchmark Suite")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - Configurations that would be tested:")
        configs = [
            {"steps": 10, "gpu_count": 1, "description": "10 steps, 1 GPU"},
            {"steps": 20, "gpu_count": 1, "description": "20 steps, 1 GPU"},
            {"steps": 10, "gpu_count": 8, "description": "10 steps, 8 GPUs"},
            {"steps": 20, "gpu_count": 8, "description": "20 steps, 8 GPUs"},
        ]
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config['description']}")
    else:
        run_comprehensive_benchmark()
