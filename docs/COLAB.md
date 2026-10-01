# Running nvs3d on Colab / Kaggle

> `gdown` (used by `scripts/download_lego.py`) is an **optional** dependency.
> Install it locally with `pip install -e ".[data]"` or just `pip install gdown`.
> On Colab/Kaggle it is installed automatically via `requirements-colab.txt`.

## Google Colab

### One-time setup

1. **Upload Lego data to Drive.**
   - Download `nerf_synthetic.zip` from the [official NeRF data folder](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).
   - Extract the `lego/` folder and upload it to your Drive, e.g.
     `My Drive/datasets/nerf_synthetic/lego/`.

2. **Open the notebook.**
   - Upload `notebooks/colab_runner.ipynb` to Colab, or open it from GitHub.
   - Select **Runtime → Change runtime type → T4 GPU**.

3. **Edit placeholders.**
   - Set `REPO_URL` to your fork's URL.
   - Set `DRIVE_LEGO` to the path where you uploaded `lego/` on Drive.

### Training

Run cells 1–5 in order. Output (checkpoints, metrics, renders) is written to
`/content/drive/MyDrive/nvs3d_runs/` so a session disconnect does not lose
progress.

### Resuming after disconnect

Re-run cells 1–4 (mount, clone, setup, link data), then run cell 6 (resume).
The resume cell finds the latest checkpoint in the Drive runs directory.

### CLI overrides

The same code runs locally and on Colab without edits, using CLI overrides:

```bash
python scripts/train.py \
    --config configs/nerf_blender.yaml \
    --data_dir /content/nvs3d/datasets/nerf_synthetic/lego \
    --output_dir /content/drive/MyDrive/nvs3d_runs \
    --device cuda
```

---

## Kaggle

### Setup

1. **Add data.**
   - Go to your Kaggle notebook → **Add data** → search for "nerf synthetic"
     or upload `nerf_synthetic.zip` as a private dataset.
   - The data will be mounted at `/kaggle/input/<dataset-name>/lego/`.

2. **Clone the repo and install.**
   ```bash
   !git clone https://github.com/YOUR_USER/nvs3d.git /kaggle/working/nvs3d
   %cd /kaggle/working/nvs3d
   !bash scripts/colab_setup.sh
   ```

3. **Link or copy data.**
   ```python
   from pathlib import Path
   src = Path("/kaggle/input/nerf-synthetic/lego")   # adjust to your dataset name
   dst = Path("/kaggle/working/nvs3d/datasets/nerf_synthetic/lego")
   dst.parent.mkdir(parents=True, exist_ok=True)
   if not dst.exists():
       dst.symlink_to(src)
   ```

4. **Train.**
   ```bash
   !python scripts/train.py \
       --config configs/nerf_blender.yaml \
       --data_dir /kaggle/working/nvs3d/datasets/nerf_synthetic/lego \
       --output_dir /kaggle/working/nvs3d_runs \
       --device cuda
   ```

### Notes

- Kaggle provides **30 hours/week** of GPU (T4 or P100).
- Outputs go to `/kaggle/working/` which persists across saves.
- `/kaggle/input/` is read-only — always write outputs to `/kaggle/working/`.
- Save your notebook version periodically to preserve outputs.
