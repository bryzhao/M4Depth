# M4Depth - RBE 577 Final Project

This repository is a fork of the original [M4Depth](https://github.com/michael-fonder/M4Depth) implementation, extended for the **Machine Learning for Robotics** final project at WPI.

The project reproduces the paper's results on the MidAir synthetic dataset and explores sim-to-real transfer by fine-tuning on the UseGeo real-world drone dataset.

---

## Original Work Attribution

This codebase builds upon the work of Fonder et al. The original README content and paper citation are preserved below, as their work forms the foundation of this project.

> **Parallax Inference for Robust Temporal Monocular Depth Estimation in Unstructured Environments**
>
> [Michaël Fonder](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u225873), [Damien Ernst](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u030242) and [Marc Van Droogenbroeck](https://www.uliege.be/cms/c_9054334/fr/repertoire?uid=u182591)
>
> [PDF](https://www.mdpi.com/1424-8220/22/23/9374/pdf) | [Sensors Journal](https://www.mdpi.com/1424-8220/22/23/9374)

If you use this work, please cite the original paper:
```bibtex
@article{Fonder2022Parallax,
  title     = {Parallax Inference for Robust Temporal Monocular Depth Estimation in Unstructured Environments},
  author    = {Fonder, Michael and Ernst, Damien and Van Droogenbroeck, Marc},
  journal   = {Sensors},
  volume    = {22},
  number    = {23},
  pages     = {1-22},
  month     = {November},
  year      = {2022},
  doi       = {10.3390/s22239374}
}
```

---

## Project Overview

M4Depth estimates depth from RGB image sequences using camera motion (from GPS/IMU). Unlike methods that learn scene-specific depth cues, M4Depth learns the physics of **motion parallax**—how objects at different distances move differently as the camera moves. This makes it more robust in unfamiliar environments like forests.

### What I Added

This fork extends the original repository with:

| Addition | Purpose |
|----------|---------|
| `train.sh`, `eval.sh` | Wrapper scripts for training and evaluation with proper environment setup |
| `dataloaders/usegeo.py` | Custom dataloader for the UseGeo real-world dataset |
| `train_usegeo.py` | Fine-tuning script for sim-to-real transfer learning |
| `visualize_predictions.py` | Generate side-by-side depth comparisons for MidAir |
| `visualize_usegeo.py` | Generate depth comparisons for UseGeo |
| `plot_*.py` | Scripts to generate training curves and metric comparisons |
| `usegeo_csv_generator.py` | Generate train/test splits for UseGeo |

---

## Results Summary

### Phase 1: MidAir (Synthetic Data)

I trained the network from scratch on MidAir and achieved results comparable to the paper:

| Metric | My Results | Paper | Notes |
|--------|-----------|-------|-------|
| Abs Rel ↓ | **0.102** | 0.105 | 3% better |
| Sq Rel ↓ | **3.23** | 3.454 | 6% better |
| RMSE ↓ | 7.24 | 7.043 | Similar |
| RMSE Log ↓ | 0.190 | 0.186 | Similar |
| δ < 1.25 ↑ | 0.917 | 0.919 | Similar |
| δ < 1.25² ↑ | 0.953 | 0.953 | Identical |
| δ < 1.25³ ↑ | 0.969 | 0.969 | Identical |

### Phase 2: UseGeo (Real-World Data)

Fine-tuned the MidAir-pretrained model on real drone imagery from UseGeo. See `usegeo_comparison.png` for a comparison of fine-tuned vs. scratch-trained models.

---

## Setup Instructions

### Prerequisites

- Ubuntu 22.04 (tested)
- NVIDIA GPU with 12GB+ VRAM
- CUDA 12.x
- Python 3.10

### Environment Setup

I used a Python virtual environment instead of conda to avoid conflicts with system packages (ROS2, etc.). The key is explicitly setting PYTHONPATH to isolate from system Python.

```bash
# Create virtual environment
python3 -m venv m4depth_env
source m4depth_env/bin/activate

# Install dependencies
pip install tensorflow[and-cuda]==2.20 numpy pandas h5py pyquaternion matplotlib tifffile

# Verify GPU detection
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Important: Environment Activation

Before running any script, always activate the environment with explicit PYTHONPATH:

```bash
source /path/to/m4depth_env/bin/activate
export PYTHONPATH="/path/to/m4depth_env/lib/python3.10/site-packages"
```

The wrapper scripts (`train.sh`, `eval.sh`, etc.) handle this automatically.

### Dataset Setup

1. **MidAir Dataset** (~317 GB)
   - Download from [midair.ulg.ac.be](https://midair.ulg.ac.be/download.html)
   - Select "Left RGB" and "Stereo Disparity"
   - Extract and update `datasets_location.json`

2. **UseGeo Dataset**
   - Download from the UseGeo project page
   - Run `python usegeo_csv_generator.py` to create train/test splits

3. **Configure paths** in `datasets_location.json`:
   ```json
   {
     "midair": "/path/to/MidAir",
     "usegeo": "/path/to/UseGeo"
   }
   ```

---

## Usage

### Training on MidAir (Phase 1)

```bash
./train.sh
```

This runs training with the paper's configuration:
- 4-frame sequences
- 6-level pyramid architecture
- Batch size 3
- ~220k iterations total

Monitor training progress:
```bash
tensorboard --logdir=./checkpoints/summaries
```

### Evaluation

```bash
./eval.sh
```

Results are saved to `checkpoints/perfs-midair.txt`.

### Generate Visualizations

```bash
./visualize.sh --num_samples=10
```

Creates side-by-side comparisons in `visualizations/`.

### Fine-tuning on UseGeo (Phase 2)

```bash
./train_usegeo.sh
```

This loads the MidAir-pretrained weights and fine-tunes on UseGeo with:
- Lower learning rate (1e-5)
- Batch size 2
- 30 epochs

### Generate Training Plots

```bash
python plot_from_tensorboard.py --output=training_curves.png
python plot_usegeo_training.py
python plot_usegeo_comparison.py
```

---

## Repository Structure

```
M4Depth/
├── main.py                    # Original training/eval script
├── m4depth_network.py         # Network architecture
├── m4depth_options.py         # Command-line options
├── dataloaders/
│   ├── generic.py             # Base dataloader class
│   ├── midair.py              # MidAir dataloader (original)
│   ├── usegeo.py              # UseGeo dataloader (added)
│   └── ...
├── data/midair/               # Train/test CSV splits
│
│ # Scripts I Added:
├── train.sh                   # MidAir training wrapper
├── eval.sh                    # Evaluation wrapper
├── visualize.sh               # Visualization wrapper
├── train_usegeo.py            # UseGeo fine-tuning script
├── train_usegeo.sh            # UseGeo training wrapper
├── visualize_predictions.py   # MidAir visualization
├── visualize_usegeo.py        # UseGeo visualization
├── visualize_usegeo.sh        # UseGeo viz wrapper
├── plot_from_tensorboard.py   # Training curve plots
├── plot_usegeo_training.py    # UseGeo training plots
├── plot_usegeo_comparison.py  # Compare fine-tuned vs scratch
├── usegeo_csv_generator.py    # Generate UseGeo data splits
│
│ # Generated Outputs:
├── training_curves.png        # MidAir training loss plot
├── eval_comparison.png        # Metrics comparison chart
├── usegeo_training_curves.png # UseGeo training loss plot
├── usegeo_comparison.png      # Fine-tuned vs scratch comparison
└── checkpoints/
    └── perfs-midair.txt       # Evaluation metrics
```

---

## Key Findings: Sim-to-Real Transfer

One goal of this project was to investigate whether pretraining on synthetic data helps with real-world depth estimation.

**Observations:**
- The MidAir-pretrained model provides a useful starting point
- Fine-tuning on UseGeo improves performance on real images
- The domain gap is significant—synthetic images are cleaner and more uniform
- UseGeo lacks camera pose data, so M4Depth operates in single-frame mode rather than using motion parallax

**Challenge:** UseGeo doesn't provide GPS/IMU pose data, which M4Depth normally uses to compute parallax. The dataloader uses identity poses (no motion), so the network falls back to single-frame depth estimation. Despite this limitation, the pretrained encoder features still help.

---

## Troubleshooting

### GPU not detected
```bash
# Verify CUDA installation
nvidia-smi

# Check TensorFlow sees GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Import errors / wrong packages
System Python packages (from ROS2, etc.) can leak into the venv. Always use:
```bash
export PYTHONPATH="/path/to/m4depth_env/lib/python3.10/site-packages"
```

### NaN during training
This happens occasionally due to numerical instability. Resume from the latest checkpoint—training auto-saves.

### Out of memory
Reduce batch size:
```bash
python main.py --batch_size=2 ...
```

---

## Original README Content

The sections below are preserved from the original M4Depth repository for reference.

---

### Overview (Original)

M4Depth is deep neural network designed to estimate depth from RGB image sequences acquired in unknown environments by a camera moving with 6 degrees of freedom (DoF), and is:

* **Lightweight**: M4Depth only requires 500MB of VRAM to run;
* **Real-time**: M4Depth has a fast inference time that makes it compatible for real-time applications on most GPUs;
* **State-of-the-art**: M4Depth is state of the art on the Mid-Air dataset and outperforms existing methods in a generalization setup on the TartanAir dataset, while having good performances on the KITTI dataset.

This network is the result of two major contributions:
* We define a notion of visual parallax between two frames from a generic six-degree-of-freedom (6-DoF) camera motion, and present a way to build cost volumes with this parallax;
* We use these cost volumes to build a novel lightweight multi-level architecture;

### Dependencies (Original)

The original authors recommend Anaconda:
```shell
conda install -c conda-forge tensorflow-gpu=2.7 numpy pandas
```

**My setup:** I used pip with TensorFlow 2.20 for CUDA 12.x compatibility:
```shell
pip install tensorflow[and-cuda] numpy pandas h5py pyquaternion matplotlib
```

### Datasets (Original)

Our set of experiments relies on three different datasets. Our code refers to the `datasets_location.json` configuration file to know where the data are located.

#### Mid-Air
Download from [midair.ulg.ac.be](https://midair.ulg.ac.be/download.html). Select "Left RGB" and "Stereo Disparity" image types.

#### KITTI
```shell
bash scripts/0b-get_kitti.sh path/to/desired/dataset/location
```

#### TartanAir
```shell
bash scripts/0c-get_tartanair.sh path/to/desired/dataset/location
```

### Training from Scratch (Original)

```shell
bash scripts/1a-train-midair.sh path/to/desired/weights/location
```

### Evaluation (Original)

```shell
bash scripts/2-evaluate.sh dataset path/to/weights/location
```

### Running on Custom Data (Original)

To test M4Depth on a custom dataset:
1. Generate CSV files mapping frame locations and camera motion
2. Write a corresponding dataloader (inherit from `DataLoaderGeneric`)
3. Add your dataset as a choice in the `--dataset` argument

---

## References

```bibtex
@inproceedings{Fonder2019MidAir,
  author    = {Fonder, Michael and Van Droogenbroeck, Marc},
  title     = {Mid-Air: A multi-modal dataset for extremely low altitude drone flights},
  booktitle = {IEEE CVPRW},
  year      = {2019}
}

@inproceedings{Geiger2012AreWe,
  title     = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  author    = {Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle = {IEEE CVPR},
  year      = {2012}
}
```

---

## License

The original code is licensed under AGPLv3. See [LICENSE](LICENSE) for details.

The CUDA backprojection implementation is licensed under BSD 3-Clause. See [cuda_backproject/LICENSE](cuda_backproject/LICENSE).
