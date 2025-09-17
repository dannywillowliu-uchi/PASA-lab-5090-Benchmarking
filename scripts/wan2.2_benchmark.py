
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
from datetime import datetime

def run_benchmark(
    model_path,
    prompt,
    sample_steps,
    size,
    frame_num,
    output_dir,
    offload_model,
    gpu_id
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(output_dir, f"wan22_test_{timestamp}.gif")
    results_file = os.path.join(output_dir, f"wan22_results_{timestamp}.json")

    cmd = [
        "python3", "/workspace/wan2.2/generate.py",
        "--task", "t2v-A14B",
        "--size", size,
        "--offload_model", str(offload_model),
        "--sample_steps", str(sample_steps),
        "--ckpt_dir", model_path,
        "--convert_model_dtype",
        "--prompt", prompt,
        "--save_file", video_file,
        "--frame_num", str(frame_num)
    ]

    print(f"[INFO] Running WAN2.2 benchmark: {cmd}")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        total_time = end_time - start_time
        output = result.stdout
        error = result.stderr
        success = True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        total_time = end_time - start_time
        output = e.stdout if hasattr(e, 'stdout') else ''
        error = e.stderr if hasattr(e, 'stderr') else str(e)
        success = False

    # Standardized result format
    results = {
        "model": "wan2.2-t2v-a14b",
        "prompt": prompt,
        "task": "t2v-A14B",
        "resolution": size,
        "steps": sample_steps,
        "precision": "FP16",
        "gpu_id": gpu_id,
        "frame_num": frame_num,
        "output_file": video_file,
        "success": success,
        "total_time_seconds": round(total_time, 2),
        "stdout": output,
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
    parser.add_argument("--sample_steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--size", default="480*832", help="Video resolution")
    parser.add_argument("--frame_num", type=int, default=25, help="Number of frames")
    parser.add_argument("--output_dir", default="/workspace/results", help="Output directory")
    parser.add_argument("--offload_model", action="store_true", help="Enable model offloading")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model_path,
        prompt=args.prompt,
        sample_steps=args.sample_steps,
        size=args.size,
        frame_num=args.frame_num,
        output_dir=args.output_dir,
        offload_model=args.offload_model,
        gpu_id=args.gpu_id
    )
