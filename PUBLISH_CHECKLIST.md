# Public GitHub Release Checklist

## 1. Initialize repository (if not initialized)

```bash
git init
git add .
git status
```

## 2. Confirm large files are excluded

```bash
find . -type f -size +20M
```

Only local files under `models/`, `data/`, and `results/` should be large.

## 3. Review tracked file scope

Recommended code-first structure:

- keep: `configs/`, `guided_diffusion/`, `data/dataloader.py`, `util/`, `scripts/`, entry scripts
- keep as references: `old_scripts/`, `sample_road.py` (experimental)
- do not track: checkpoints, datasets, generated outputs

## 4. Optional structure simplification before first release

- Move legacy scripts from `old_scripts/` into `archive/legacy/`
- Keep only one metrics workflow (`advanced_metric_calculator.py`)
- Keep one default run path in README (conditional + unconditional)

## 5. Final sanity checks

```bash
bash -n run_random_cond.sh run_random_uncond.sh run_sensor_cond.sh run_sensor_uncond.sh
python3 -m py_compile advanced_metric_calculator.py sample_condition.py sample_condition_uncond.py
```

## 6. First commit and push

```bash
git add .
git commit -m "Prepare public release: cleanup, ignore rules, docs, portable scripts"
# replace with your repo URL
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```
