
#!/usr/bin/env python3
"""
WAN 2.2 Video Generation Benchmark Script (Standardized)
Benchmarks the WAN 2.2 T2V model with validated parameters and standardized output.
"""

import subprocess
import time
import json
import os
import argparse
import threading
import re
from datetime import datetime

def monitor_gpu_memory(memory_stats, monitoring_active, nproc_per_node=1):
    """Monitor GPU memory usage during benchmark"""
    memory_readings = []
    
    while monitoring_active.is_set():
        try:
            # Get memory info for all GPUs being used
            for gpu_id in range(nproc_per_node):
                result = subprocess.run(['nvidia-smi', f'--id={gpu_id}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, check=True)
                memory_used, memory_total = map(int, result.stdout.strip().split(', '))
                memory_readings.append({
                    'timestamp': time.time(),
                    'gpu_id': gpu_id,
                    'memory_used_mb': memory_used,
                    'memory_total_mb': memory_total,
                    'memory_utilization_pct': round((memory_used / memory_total) * 100, 2)
                })
            time.sleep(0.5)  # Sample every 500ms
        except Exception as e:
            print(f"[WARNING] GPU memory monitoring error: {e}")
            break
    
    memory_stats['readings'] = memory_readings
    return memory_readings

def parse_generation_timing(stdout):
    """Extract generation timing from WAN2.2 logs"""
    timings = {}
    
    # Look for key timing markers in logs
    lines = stdout.split('\n')
    for line in lines:
        if 'Creating WanT2V pipeline' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['pipeline_start'] = match.group(1)
        elif 'loading' in line and 'models_t5' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['t5_load_start'] = match.group(1)
        elif 'loading' in line and 'VAE' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['vae_load_start'] = match.group(1)
        elif 'Creating WanModel from' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['model_load_start'] = match.group(1)
        elif 'Generating video' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['generation_start'] = match.group(1)
        elif 'Saving generated video' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['generation_end'] = match.group(1)
        elif 'Finished' in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                timings['finished'] = match.group(1)
    
    return timings

def calculate_timing_metrics(timings):
    """Calculate timing metrics from parsed timestamps"""
    metrics = {}
    
    def time_diff(start_key, end_key):
        if start_key in timings and end_key in timings:
            start = datetime.strptime(timings[start_key], '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(timings[end_key], '%Y-%m-%d %H:%M:%S')
            return (end - start).total_seconds()
        return None
    
    metrics['model_loading_time'] = time_diff('pipeline_start', 'generation_start')
    metrics['pure_generation_time'] = time_diff('generation_start', 'generation_end')
    metrics['saving_time'] = time_diff('generation_end', 'finished')
    metrics['total_inference_time'] = time_diff('generation_start', 'finished')
    
    return metrics

def run_benchmark(
    model_path,
    prompt,
    sample_steps,
    size,
    output_dir,
    offload_model,
    gpu_id,
    multi_gpu=False,
    nproc_per_node=1
):
    # Let WAN2.2 use its default frame count (no frame_num parameter)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(output_dir, f"wan22_test_{timestamp}.gif")
    results_file = os.path.join(output_dir, f"wan22_results_{timestamp}.json")

    if multi_gpu and nproc_per_node > 1:
        # Multi-GPU distributed command with torchrun
        cmd = [
            "torchrun", f"--nproc_per_node={nproc_per_node}",
            "/workspace/wan2.2/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--sample_steps", str(sample_steps),
            "--ckpt_dir", model_path,
            "--dit_fsdp",
            "--t5_fsdp", 
            "--ulysses_size", str(nproc_per_node),
            "--prompt", prompt,
            "--save_file", video_file
        ]
    else:
        # Single GPU command
        cmd = [
            "python3", "/workspace/wan2.2/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--offload_model", str(offload_model),
            "--sample_steps", str(sample_steps),
            "--ckpt_dir", model_path,
            "--convert_model_dtype",
            "--prompt", prompt,
            "--save_file", video_file
        ]

    print(f"[INFO] Running WAN2.2 benchmark: {cmd}")
    
    # Start GPU memory monitoring
    memory_stats = {}
    monitoring_active = threading.Event()
    monitoring_active.set()
    memory_thread = threading.Thread(target=monitor_gpu_memory, args=(memory_stats, monitoring_active, nproc_per_node))
    memory_thread.start()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        total_time = end_time - start_time
        success = True
        stdout = result.stdout
        error = result.stderr
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        total_time = end_time - start_time
        success = False
        stdout = e.stdout if hasattr(e, 'stdout') else ''
        error = e.stderr if hasattr(e, 'stderr') else str(e)
    finally:
        # Stop memory monitoring
        monitoring_active.clear()
        memory_thread.join(timeout=5)
    
    # Parse timing information from logs
    timing_info = parse_generation_timing(stdout)
    timing_metrics = calculate_timing_metrics(timing_info)
    
    # Calculate memory statistics
    memory_readings = memory_stats.get('readings', [])
    memory_analysis = {}
    if memory_readings:
        # Group by GPU
        for gpu_id in range(nproc_per_node):
            gpu_readings = [r for r in memory_readings if r['gpu_id'] == gpu_id]
            if gpu_readings:
                memory_used = [r['memory_used_mb'] for r in gpu_readings]
                memory_analysis[f'gpu_{gpu_id}'] = {
                    'peak_memory_mb': max(memory_used),
                    'average_memory_mb': round(sum(memory_used) / len(memory_used), 2),
                    'min_memory_mb': min(memory_used),
                    'peak_utilization_pct': max([r['memory_utilization_pct'] for r in gpu_readings]),
                    'average_utilization_pct': round(sum([r['memory_utilization_pct'] for r in gpu_readings]) / len(gpu_readings), 2)
                }
        
        # Overall statistics across all GPUs
        all_memory_used = [r['memory_used_mb'] for r in memory_readings]
        memory_analysis['overall'] = {
            'total_peak_memory_mb': sum([memory_analysis[f'gpu_{i}']['peak_memory_mb'] for i in range(nproc_per_node) if f'gpu_{i}' in memory_analysis]),
            'total_average_memory_mb': sum([memory_analysis[f'gpu_{i}']['average_memory_mb'] for i in range(nproc_per_node) if f'gpu_{i}' in memory_analysis]),
            'memory_readings_count': len(memory_readings)
        }

    # Comprehensive result format with detailed metrics
    results = {
        "model": "wan2.2-t2v-a14b",
        "prompt": prompt,
        "task": "t2v-A14B",
        "resolution": size,
        "steps": sample_steps,
        "precision": "FP16",
        "gpu_configuration": f"{nproc_per_node}_gpu" if multi_gpu else "single_gpu",
        "nproc_per_node": nproc_per_node,
        "distributed": multi_gpu,
        "frame_count": "default",
        "output_file": video_file,
        "success": success,
        
        # Timing metrics
        "total_execution_time_seconds": round(total_time, 2),
        "model_loading_time_seconds": timing_metrics.get('model_loading_time'),
        "pure_generation_time_seconds": timing_metrics.get('pure_generation_time'),
        "saving_time_seconds": timing_metrics.get('saving_time'),
        "total_inference_time_seconds": timing_metrics.get('total_inference_time'),
        
        # Memory metrics
        "memory_analysis": memory_analysis,
        "memory_monitoring_samples": len(memory_readings),
        
        # Raw timing data
        "timing_markers": timing_info,
        
        "stdout": stdout,
        "stderr": error,
        "timestamp": timestamp
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {results_file}")
    if success:
        print(f"[INFO] Video saved to: {video_file}")
    else:
        print(f"[ERROR] Benchmark failed. See stderr in results file.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN 2.2 Video Generation Benchmark (Standardized)")
    parser.add_argument("--model_path", default="/workspace/models", help="Path to WAN 2.2 model checkpoints")
    parser.add_argument("--prompt", default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.", help="Text prompt")
    parser.add_argument("--sample_steps", type=int, default=10, help="Number of sampling steps")
    parser.add_argument("--size", default="480*832", help="Video resolution")
    parser.add_argument("--output_dir", default="/workspace/results", help="Output directory")
    parser.add_argument("--offload_model", action="store_true", help="Enable model offloading")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multi-GPU distributed training")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node for multi-GPU")

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model_path,
        prompt=args.prompt,
        sample_steps=args.sample_steps,
        size=args.size,
        output_dir=args.output_dir,
        offload_model=args.offload_model,
        gpu_id=args.gpu_id,
        multi_gpu=args.multi_gpu,
        nproc_per_node=args.nproc_per_node
    )
