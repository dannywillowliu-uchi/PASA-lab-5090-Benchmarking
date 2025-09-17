
# GPU Benchmarking Infrastructure

A comprehensive benchmarking system for evaluating text-to-image and text-to-video models across multiple GPU configurations, with support for quantized models using GGUF format.

## Project Overview

This project provides automated benchmarking infrastructure for:
- **FLUX.1-dev** text-to-image generation (quantized: Q8_0, Q4_0, Q2_K)
- **WAN 2.2 5B** text-to-image generation (quantized: Q8_0, Q4_0, Q2_K) 
- **WAN 2.2 14B** text-to-video generation (quantized and full precision)

### Hardware Support
- 8x NVIDIA GeForce RTX 5090 (32.6GB VRAM each)
- Multi-GPU parallel processing
- Docker containerization for reproducible environments

## Project Structure

```
gpu_benchmarking/
├── benchmarks/           # Benchmark configurations and scripts
│   ├── configs/         # JSON configuration files
│   ├── data/           # Test data and prompts
│   ├── reports/        # Generated benchmark reports  
│   └── scripts/        # Benchmark execution scripts
├── docker/             # Docker infrastructure
│   ├── configs/        # Docker-specific configs
│   ├── dockerfiles/    # Container definitions
│   ├── images/         # Docker build scripts
│   └── backups/        # Critical image backups
├── models/             # Model storage (gitignored - too large)
│   └── quantized/      # GGUF quantized models
├── results/            # Benchmark outputs (gitignored)
├── shared/             # Shared utilities
│   ├── monitoring/     # GPU monitoring tools
│   └── utils/          # Common utilities
└── scripts/            # Main execution scripts
```

## Setup Instructions

### Prerequisites
- Docker with NVIDIA container runtime
- NVIDIA drivers and CUDA toolkit
- Python 3.8+
- Hugging Face CLI (for model downloads)

### Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd gpu_benchmarking
```

2. **Set up Docker infrastructure**
```bash
# Make scripts executable
chmod +x docker/*.sh

# Label critical images for protection
./docker/label_critical_images.sh

# Create backup of current images
./docker/backup_images.sh
```

3. **Download quantized models** (these are not in git due to size)
```bash
# FLUX models
huggingface-cli download city96/FLUX.1-dev-gguf flux1-dev-Q8_0.gguf --local-dir models/quantized/flux.1-dev-gguf/Q8_0/
huggingface-cli download city96/FLUX.1-dev-gguf flux1-dev-Q4_0.gguf --local-dir models/quantized/flux.1-dev-gguf/Q4_0/
huggingface-cli download city96/FLUX.1-dev-gguf flux1-dev-Q2_K.gguf --local-dir models/quantized/flux.1-dev-gguf/Q2_K/

# WAN models
huggingface-cli download QuantStack/Wan2.2-TI2V-5B-GGUF --local-dir models/quantized/wan2.2-ti2v-5b-gguf/
huggingface-cli download QuantStack/Wan2.2-T2V-A14B-GGUF --local-dir models/quantized/wan2.2-t2v-a14b-gguf/
```

4. **Build Docker images**
```bash
# Quick rebuild from existing working image
./docker/quick_rebuild.sh

# Or full rebuild if needed
./docker/rebuild_all.sh
```

## 🐳 Docker Infrastructure

### Critical Images
- `wan2.2-benchmark:latest` - Main benchmarking environment (11GB)
- `flux1dev-texttoimage:simple` - FLUX text-to-image with quantization support
- `wan22_5b:gpu` - WAN 5B with GGUF support
- `wan22_14b:gpu` - WAN 14B with GGUF support

### Protection Scripts
- `backup_images.sh` - Creates backups before maintenance
- `safe_cleanup.sh` - Safely clean Docker without destroying critical images
- `label_critical_images.sh` - Tags important images for protection

## 📊 Running Benchmarks

### Single GPU Benchmarks
```bash
# FLUX benchmarks
python3 flux1dev_benchmark_1gpu.py --steps 30 --quantization Q4_0

# WAN 5B benchmarks  
python3 wan2_2_5b_benchmark.py --steps 20 --quantization Q8_0

# WAN 14B benchmarks
python3 wan2_2_benchmark_single_gpu.py --gpu_id 0 --sample_steps 10
```

### Multi-GPU Benchmarks
```bash
# 8-GPU WAN 14B benchmark
python3 wan2_2_benchmark_8gpu.py --model_path /root/Wan2.2-T2V-A14B --gpu_count 8 --sample_steps 20
```

### Quantized Model Benchmarks
```bash
# Run comprehensive quantized benchmark
./scripts/run_quantized_benchmark.sh
```

## 📈 Benchmark Configurations

The project includes GenEval-compatible benchmark configurations:

### FLUX.1-dev Quantized
- **Q8_0**: 10, 30, 50 inference steps
- **Q4_0**: 10, 30, 50 inference steps  
- **Q2_K**: 10, 30, 50 inference steps

### WAN 2.2 5B Quantized
- **Q8_0**: 10, 20 inference steps
- **Q4_0**: 10, 20 inference steps
- **Q2_K**: 10, 20 inference steps

### WAN 2.2 14B Quantized
- **Q8_0**: 10, 20 inference steps
- **Q4_0**: 10, 20 inference steps
- **Q2_K**: 10, 20 inference steps

## 🛡️ Infrastructure Protection

### Before System Maintenance
Always run the backup script:
```bash
./docker/backup_images.sh
```

### Safe Docker Cleanup
Use the protected cleanup script:
```bash
./docker/safe_cleanup.sh
```

### Recovery
If images are lost, use the quick rebuild:
```bash
./docker/quick_rebuild.sh
```

## 📝 Configuration Files

Key configuration files:
- `benchmarks/configs/` - Benchmark parameters
- `docker/dockerfiles/` - Container definitions
- `DOCKER_PROTECTION_PROTOCOL.md` - Infrastructure protection guidelines

## 🔍 Monitoring

GPU monitoring utilities in `shared/monitoring/`:
- Real-time GPU utilization tracking
- Memory usage monitoring
- Temperature and power monitoring

## 🚨 Important Notes

1. **Large Files**: Models and results are excluded from git (see `.gitignore`)
2. **Docker Protection**: Always backup images before system maintenance
3. **GPU Access**: Ensure NVIDIA container runtime is properly configured
4. **Model Downloads**: Use Hugging Face CLI for quantized model downloads
5. **Recovery**: Keep `quick_rebuild.sh` and Dockerfiles updated

## 📋 Troubleshooting

### NVIDIA Runtime Issues
```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Test GPU access
sudo docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Missing Images
```bash
# List available images
sudo docker images

# Rebuild from existing working image
./docker/quick_rebuild.sh
```

### Model Download Issues
```bash
# Login to Hugging Face
huggingface-cli login

# Check available models
huggingface-cli list city96/FLUX.1-dev-gguf
```

## 🤝 Contributing

1. Always test Docker builds before committing
2. Update documentation for new benchmark configurations
3. Use protection scripts before major changes
4. Keep Dockerfiles and build scripts updated

## 📄 License

[Add your license information here]

---

**⚠️ Critical Infrastructure**: This repository contains critical benchmarking infrastructure. Always use protection scripts before system maintenance operations.
