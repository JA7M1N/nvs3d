# Per-phase prompts (P0–P13)

Paste one at a time. Review the diff and tests before sending the next.
Every prompt ends with the same closing block (shown once below as CLOSE).

**CLOSE:**
> Stop when the phase's "Done" criteria pass. Then summarize: the exact commands you ran, pass/fail counts, any tolerance you chose and why, every (verify) item you hit, and anything you are unsure about. Do not touch earlier-phase code without telling me first. If a test fails, do not loosen the tolerance; explain why.

---

## P0 — Scaffolding
> Read `AGENTS.md` and `docs/blueprint.md` Parts 1, 2, 6 and Part 14 step 1. Implement only Phase P0. Create the repo structure from Part 6 (empty modules are fine), `pyproject.toml`, config/logging/seed/checkpoint utilities, and a pytest setup. Write a short plan first and wait for my approval before writing code. Done = pytest runs and a dummy run writes a run dir with `config.yaml` and `metrics.jsonl`. Do not start P1. + CLOSE

## P1 — Geometry core + data loaders
> Implement Phase P1. Read `docs/blueprint.md` N1, N2, Part 7 (P1), Part 9, and Part 14 steps 2–5. Build `core/transforms.py`, `core/cameras.py`, `core/rays.py`, `data/blender_io.py`, `data/scene.py`. Tests first: SE3/Sim3 inverse and composition, Umeyama recovers a known Sim3, project/unproject round trip, ray through the principal point is the optical axis, half-pixel offset, Blender->OpenCV flip only inside the loader. Add a frustum plot script; check that Lego cameras form a hemisphere facing the origin and that rays hit the object. + CLOSE

## P2 — NeRF components
> Implement Phase P2. Read N3 (only the sampler interfaces), N4, N5, N6, Part 7 (P2), Part 9. Build `nerf/encoding.py` (positional), `nerf/field.py`, `nerf/volume.py`, `nerf/sampler.py` (stratified only). Tests first: analytic slab transmittance, weights sum to 1 - T_final, exclusive-cumprod off-by-one test, float64 gradcheck of volume rendering, random-field render is finite. Flag every (verify) item (e.g. PE frequencies, MLP widths/skip layers). + CLOSE

## P3 — Coarse NeRF trainer on synthetic data
> Implement Phase P3. Read N3, N7, N8, N10, Part 7 (P3), Part 9. Build `train/losses.py`, PSNR in `eval/metrics.py`, `train/nerf_trainer.py` (coarse network only), `nerf/render.py`. First overfit 1 image, then Lego at 100×100, then 200×200. Check white-background compositing and near/far. Report real PSNR numbers. If no GPU, write the code and CPU smoke tests only and tell me the exact command to run on Colab/Kaggle. + CLOSE

## P4 — Hierarchical NeRF + evaluation
> Implement Phase P4. Read N3 (inverse-CDF), N8, Part 7 (P4), Part 9, Part 14 step 12. Add the fine network, inverse-CDF sampling (no gradient through sampled t, guard NaN in the pdf), `core/ssim.py`, an LPIPS wrapper, and `eval/evaluate.py` that writes `report.md`. Tests first: inverse-CDF samples match a known 1D distribution statistically; SSIM(x,x)=1 and matches a reference value; gradcheck SSIM. + CLOSE

## P5 — Video preprocessing + SfM
> Implement Phase P5. Read `docs/blueprint.md` Part 5, Part 7 (P5), Part 9, Part 11, Part 14 steps 13–15. Build `preprocess/{extract_frames,select,run_sfm,diagnose_sfm,normalize}.py` and `data/colmap_io.py`. Tests first: blur scorer ranks synthetically blurred frames correctly; redundancy filter; Sim3 normalization round trip. Synthetic validation: render a Blender-style video with known poses, run SfM, report Umeyama-aligned ATE. Do not reimplement distortion; use COLMAP/OpenCV undistortion. + CLOSE

## P6 — Hash-grid NeRF on an object capture
> Implement Phase P6. Read N9, Part 7 (P6), Part 9, Part 14 step 16. Implement the hash grid in `nerf/encoding.py` and `HashNeRF` behind the `RadianceField` protocol. Tests first: trilinear interpolation at grid vertices equals the stored feature; gradients reach only the touched entries; gradcheck in fp64. Quote every (verify) hyperparameter back to me. Benchmark it/s. + CLOSE

## P7 — 3DGS math
> Implement Phase P7. Read G1, G3, G4, Part 7 (P7), Part 9, Part 14 steps 17–19. Build `core/quaternion.py`, `core/sh.py`, `gs/model.py` (covariance), `gs/project.py` (EWA). Order of evidence for each: analytic test (isotropic Gaussian projection, SH DC term), numerical Jacobian, fp64 gradcheck, then oracle comparison against gsplat (means2d, conics, radii, rgb) in `tests/render/`. Check quaternion order and SH sign convention explicitly. + CLOSE

## P8 — Naive rasterizer + overfit
> Implement Phase P8. Read G5, G6, Part 7 (P8), Part 9, Part 14 step 20. Build `gs/raster/base.py`, `gs/raster/naive.py` (dense [N,P] front-to-back compositing on tiny images), `gs/render.py`. Tests first: single Gaussian at a known pixel, two opaque Gaussians obey depth order, behind-camera culled, fp64 gradcheck of compositing. Then (a) 2D warm-up: 2k 2D Gaussians fitting a 128×128 image, (b) single-view then multi-view 3D fit on 1k Gaussians. Report real PSNR. + CLOSE

## P9 — Tiled rasterizer
> Implement Phase P9. Read G7, G11, Part 7 (P9), Part 9, Part 10, Part 14 step 21. Build `gs/raster/tiled.py` (pairs -> stable sort -> padded tiles -> compositing, cap K, chunking, crop/patch support). Tests first: equals naive backend (< 1e-4) when K is uncapped, empty tiles and tile-edge off-by-one cases, gradcheck, oracle comparison with gsplat including gradients w.r.t. μ, s, q, o, SH. Produce a benchmark table (it/s, peak VRAM, tile overflow %). Target: 256×256 crop at 200k Gaussians within 8 GB. + CLOSE

## P10 — 3DGS training without densification
> Implement Phase P10. Read G2, G8, Part 7 (P10), Part 14 steps 22–23. Build `gs/init.py` (filter, kNN scale, random fallback) and `train/gs_trainer.py`: param groups, position LR scaled by scene extent, LR decay, L1 + D-SSIM, SH degree progression, patch training, checkpointing. No densification. Tests first: init scale from kNN on a known point grid; LR scaling uses `extent`; checkpoint resume is bit-exact for a few steps. Report loss curves on Lego. Quote every (verify) hyperparameter. + CLOSE

## P11 — Density control
> Implement Phase P11. Read G9, Part 7 (P11), Part 9, Part 14 steps 24–25. Build `gs/density.py`: gradient accumulation with the NDC scaling trap handled, clone/split/prune/opacity-reset, optimizer-state surgery (Adam m and v kept in sync with parameters), Gaussian cap. Tests first (G9 tests): after clone/split/prune every param tensor and every optimizer state tensor have matching shapes and values for survivors; split preserves total mass as specified; reset lowers opacity as specified. Then train Lego 400×400 and report actual PSNR and the N-Gaussians curve. + CLOSE

## P12 — Full 3DGS on the room + oracle comparison
> Implement Phase P12. Read Part 7 (P12), Part 9, Part 11, Part 14 step 26, Part 13 (E-items referenced by P12). Train on the phone-room SceneData with our renderer, then run the same SceneData through gsplat or Splatfacto as the oracle. Build `eval/benchmark.py` producing a table of PSNR/SSIM/LPIPS/N/time. If the gap is larger than ~1.5 dB, diagnose convention mismatches first (c2w, dilation, SH direction) and report; do not tune hyperparameters to hide it. Use the held-out protocol in Part 9 (every 8th frame, ±1 neighbors excluded). + CLOSE

## P13 — Export + navigation
> Implement Phase P13. Read G10, Part 7 (P13), Part 14 steps 27–28. Build `export/{ply,splat,video}.py` and `viewer/viser_app.py` using our own renderer. Tests first: PLY round trip preserves μ, scale (log), quaternion order, opacity (logit), SH layout; the Sim3 `norm` record maps exports back to COLMAP space. Tell me exactly how to verify the PLY in SuperSplat; I will do that check myself. + CLOSE

---

### Always-on rules (already in AGENTS.md, repeat if the agent drifts)
> Quote every `(verify)` item back to me instead of hard-coding it. If a test fails, don't loosen the tolerance; explain why. No "looks right" sign-off; numeric tests only.
