# RadioDiff-Inverse

Diffusion posterior sampling for radio-map style inpainting / inverse problems.

This repository applies the FPS-SMC diffusion posterior sampling pipeline to radio map reconstruction tasks, using building masks as conditioning signals and the [RadioMapSeer](https://radiomapseer.github.io/) dataset.

---

## 1. Environment

```bash
conda create -n radiodiff python=3.8 -y
conda activate radiodiff
pip install -r requirements.txt
```

A CUDA-compatible PyTorch build is strongly recommended. The code was tested with PyTorch 2.0.0+cu118.

---

## 2. Dataset: RadioMapSeer

This project uses the **RadioMapSeer** dataset for radio map reconstruction experiments.

- **Homepage**: https://radiomapseer.github.io/
- **Download**: Follow the instructions on the dataset homepage to obtain the data.

After downloading, organize the data under `data/` as follows:

```text
data/
├── samples/           # radio map images (used as ground truth)
├── buildings_complete/# building footprint masks
├── antennas/          # antenna position maps
└── val_images/        # validation split images
```

See [data/README.md](data/README.md) for more details.

---

## 3. Pretrained Checkpoints

Download the pretrained score estimation models and place them in `models/`.

| File | Dataset | Source |
|------|---------|--------|
| `ffhq_10m.pt` | FFHQ | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) (from [DPS2022](https://github.com/DPS2022/diffusion-posterior-sampling)) |
| `imagenet256.pt` | ImageNet | [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) (from [DPS2022](https://github.com/DPS2022/diffusion-posterior-sampling)) |
| `256x256_diffusion_uncond.pt` | ImageNet (alt.) | [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |

```text
models/
├── ffhq_10m.pt
└── imagenet256.pt      # or 256x256_diffusion_uncond.pt
```

See [models/README.md](models/README.md) for details.

---

## 4. Run Experiments

All scripts default to repository-relative output paths under `results/`. You can override the output directory by passing it as the first argument.

### Random mask — with building-mask conditioning

```bash
./run_random_cond.sh
# or specify output directory:
./run_random_cond.sh results/my_run
```

### Random mask — unconditional

```bash
./run_random_uncond.sh
```

### Sensor-rectangle mask — with building-mask conditioning

```bash
./run_sensor_cond.sh
```

### Sensor-rectangle mask — unconditional

```bash
./run_sensor_uncond.sh
```

### Run directly with Python

```bash
python3 sample_condition.py \
    --task_config configs/inpainting_config_random_cond.yaml \
    --save_dir results/my_run \
    --gpu 0 \
    --num_images 10 \
    --mask_ratios 0.9 \
    --noise_levels 0.05
```

### Interpolation baselines (no diffusion)

```bash
python3 sample_interpolation.py \
    --task_config configs/inpainting_config_random_cond.yaml \
    --save_dir results/interp \
    --gpu 0
```

---

## 5. Compute Metrics

Evaluate all experiments under `results/`:

```bash
python3 advanced_metric_calculator.py --base_dir results --gpu 0
```

Evaluate a single experiment directory:

```bash
python3 advanced_metric_calculator.py --exp_dir results/random_cond --gpu 0
```

Outputs per experiment: `metrics.json`, `per_sample_metrics.json`
Summary outputs: `all_result.json`, `summary_metrics_vMMDD_HHMM.json`

---

## 6. Project Structure

```text
.
├── sample_condition.py        # main conditional pipeline
├── sample_condition_uncond.py # main unconditional pipeline
├── sample_interpolation.py    # interpolation baselines (RBF, Linear, etc.)
├── advanced_metric_calculator.py
├── configs/                   # task / model / diffusion YAML configs
├── data/                      # datasets (git-ignored except README)
├── models/                    # checkpoints (git-ignored except README)
├── results/                   # experiment outputs (git-ignored except README)
├── guided_diffusion/          # diffusion model and sampler internals
├── util/                      # metrics, image utilities, logging
├── ImageNet/                  # ImageNet-specific entry scripts
├── scripts/                   # utility scripts
├── run_random_cond.sh
├── run_random_uncond.sh
├── run_sensor_cond.sh
└── run_sensor_uncond.sh
```

---

## 7. Experimental Components

The road-conditioned path is kept for reference but is not part of the default reproducible pipeline:

- `run_cond_road.sh` is intentionally disabled
- `sample_road.py` requires additional dataset wiring before use

---

## Acknowledgements

This codebase builds on the diffusion posterior sampling / FPS-SMC framework. Pretrained checkpoints are from [DPS2022](https://github.com/DPS2022/diffusion-posterior-sampling). Radio map data is from the [RadioMapSeer](https://radiomapseer.github.io/) dataset.
