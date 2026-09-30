# NeRF / 3D Gaussian Splatting From Scratch — Technical Blueprint

Phone video → navigable 3D scene, rendering pipeline implemented in PyTorch. Date: 30 Sep 2026. Items tagged (verify) come from memory of papers/repos and must be checked against the current source before being hard-coded.

---

## Part 0 — Verdict on the Gemini research

**Kept:** the pipeline stages, the 3DGS formulas and reference hyperparameters, the NeRF recipe, the "poses are the real bottleneck" thesis, the memory arithmetic (params + grads + Adam m + v = 4×).

**Corrected / challenged:**

| # | Gemini claim | Problem | Decision |
|---|---|---|---|
| 1 | Distortion formula has k2 r | Typo; radial term is k1 r² + k2 r⁴ + k3 r⁶ | Undistort with COLMAP/OpenCV, never re-implement distortion in the renderer |
| 2 | Use sequential matching for video | For ≤ ~300 frames, exhaustive matching is cheap (~45k pairs) and gives loop closure for free; sequential needs a vocabulary tree for loops | Exhaustive by default, sequential only for >400 frames |
| 3 | Level 1 = vanilla NeRF on own phone video | Vanilla NeRF needs many hours, and a phone capture of a room is unbounded (needs contraction/proposal nets) | NeRF is validated on synthetic data first; on phone data only for bounded object captures, using a hash-grid variant. Rooms are 3DGS-only |
| 4 | Absolute PSNR targets (mid-20s dB etc.) | Meaningless across scenes/resolutions | Success is relative to an oracle (gsplat trained on the same data) plus numeric equivalence tests |
| 5 | Custom web viewer as a milestone | Not learning-critical for the renderer | MVP viewer = viser (server-side rendering with own renderer) + export to existing web viewers. Custom WebGL is optional Phase 15 |
| 6 | "Pure PyTorch rasterizer is prohibitive" | Overstated: with tile-sorted, capped, chunked compositing it works up to ~10⁵–10⁶ Gaussians at ≤ ~1 MP | Adopt a backend-swappable rasterizer: naive (tests) → tiled PyTorch (own, main) → gsplat (oracle/optional) → Triton (optional) |
| 7 | Chamfer ≲ 0.8 mm, LERF mIoU >55% targets | Unrealistic for from-scratch | Dropped |
| 8 | SLAM, feed-forward (AnySplat), PhysGaussian, Mip-NeRF 360, compression details in the core | Not on the learning path | Moved to extensions or excluded |
| 9 | Depth-prior, exposure, anti-aliasing all "essential-ish" | Each adds a variable that hides bugs | Only after the baseline matches the oracle |

**Additions Gemini missed:** float64 gradcheck-able implementation, a synthetic round-trip test (recover a known Gaussian scene), patch-based training for pure-PyTorch memory, the NDC-gradient scaling trap in densification, a Sim(3) normalization record so exports can return to COLMAP/metric space, and a pose-quality experiment.

---

## Part 1 — Project definition

**What it does:** takes a monocular video of a static scene, recovers camera poses (external SfM), and trains a radiance field with a custom PyTorch renderer. It renders novel views and lets a user navigate the scene.

**Primary input:** one phone video (.mp4/.mov, 1080p, 30 fps, 30–120 s). Secondary: Blender-synthetic datasets (transforms.json) for validation.

**Primary output:** a trained scene (checkpoint) + Inria-format .ply + flythrough video + evaluation report (PSNR/SSIM/LPIPS on held-out frames) + interactive navigation.

**Core objective:** understand and correctly implement differentiable rendering for both NeRF (volumetric, ray-marched) and 3DGS (explicit, rasterized), verified numerically against references.

**Non-goals:** writing an SfM system; dynamic scenes; relighting; metric accuracy guarantees; large-scale/outdoor scenes; production capture app; custom CUDA in the first pass.

**Assumptions:** static scene; single lens (no ultra-wide switch); CUDA NVIDIA GPU ≥ 8 GB; Python 3.11, PyTorch 2.x; COLMAP registers ≥ 90% of frames.

**Constraints:** VRAM limits resolution (MVP ≤ ~960 px long side, patch training); pure-PyTorch rasterizer is several× to tens of × slower than CUDA (verify by benchmarking in Phase 9).

**"From scratch" means:**

- **Custom:** camera/ray math, transforms, quaternions, positional encoding, MLPs (using nn.Linear), hash grid, samplers, volume rendering, SH, covariance/projection, tile-sort + compositing rasterizer, SSIM/PSNR, densification/pruning, param-group/optimizer-state management, trainers, PLY export, frame selection/blur logic, diagnostics.
- **Allowed PyTorch primitives:** autograd, nn.Linear, Adam, conv2d, searchsorted, sort.
- **External:** SfM (COLMAP), image I/O, LPIPS, viewer frameworks, pretrained models.
- **Test-oracle only:** gsplat, Inria rasterizer.

**MVP:** object-centric video → frame selection → pycolmap poses → own NeRF (vanilla + hierarchical, synthetic + object) → own 3DGS (tiled PyTorch, full density control, SH ≤ 3) → metrics, video, PLY, viser viewer.

**Full system:** MVP + fast rasterizer backends (gsplat / Triton), custom autograd for compositing, depth/appearance regularization, VGGT pose front-end comparison, meshing, metric scaling, .splat/.spz export, custom web viewer.

---

## Part 2 — System architecture

```
                        ┌──────────────────────────────┐
  phone.mp4 ──► L2 DATA │ extract → blur/dup filter →  │──► frames/
                        │ pycolmap SfM → undistort →   │──► sparse/ (K, poses, points)
                        │ normalize (Sim3)             │──► SceneData
                        └──────────────┬───────────────┘
                                       ▼
  L1 GEOMETRY CORE: Camera, SE3/Sim3, rays, quaternion, SH, SSIM  (shared)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   L3a NeRF: encoding, MLP/hashgrid                       L3b GaussianModel (μ,s,q,o,SH)
            │                                                     │
   L4a Sampler + VolumeRenderer                     L4b Rasterizer backends
            │                                       (naive|tiled_torch|gsplat|triton)
            └───────────────┬─────────────────────────────────────┘
                            ▼
   L5 TRAINING: losses, optimizer groups, schedulers, densifier (3DGS), checkpointing
                            ▼
   L6 EVAL / EXPORT / VIEWER: metrics, spiral video, PLY/.splat, viser, web viewer
   L0 INFRA (all layers): config, logging, seeds, checkpoint IO, profiling
```

**Dependency rule:** L1 imports nothing from above. L3/L4 depend on L1 only. L5 depends on L3+L4 through two protocols (RadianceField, Rasterizer). L2 outputs SceneData and nothing else; trainers never touch COLMAP files.

**Data flow:** video → frames (uint8) → SfM (COLMAP model) → SceneData (undistorted images, Camera list, points, split, normalization) → trainer samples rays (NeRF) or full/patch views (3DGS) → render → loss → backward → optimizer (+ densify) → checkpoint → eval/export.

### Component map

| Component | Module | Custom / External | Responsibility |
|---|---|---|---|
| Video input | preprocess/extract_frames.py | Ext (OpenCV/ffmpeg) | Decode, downscale, timestamp |
| Frame selection, blur, redundancy | preprocess/select.py | Custom | Sharpest-per-window + baseline check |
| Camera calibration | preprocess/run_sfm.py | Ext (COLMAP self-calib) | EXIF init, SIMPLE_RADIAL, refine |
| Feature extraction / matching | pycolmap | Ext | SIFT + exhaustive matching + RANSAC |
| Pose estimation / SfM / BA | pycolmap | Ext | K, [R\|t], sparse points |
| SLAM / feed-forward poses | preprocess/vggt_frontend.py | Ext, optional | Alternative front-end (Phase 14) |
| SfM diagnostics | preprocess/diagnose_sfm.py | Custom | Registered %, reproj error, track length, trajectory sanity |
| Undistortion | pycolmap/OpenCV | Ext | Produce PINHOLE images |
| Normalization | preprocess/normalize.py | Custom | Sim3 recenter/scale/up; scene extent |
| Point-cloud init | gs/init.py | Custom | Filter points, kNN scale, random fallback |
| Scene representation | data/scene.py | Custom | SceneData, splits |
| Camera + ray gen | core/cameras.py, core/rays.py | Custom | Pinhole, c2w/w2c, rays |
| Positional encoding, MLP, hash grid | nerf/ | Custom | Field F(x,d)→(σ,c) |
| Ray sampling | nerf/sampler.py | Custom | Stratified, inverse-CDF, ray/bounds intersect |
| Volume rendering | nerf/volume.py | Custom | α, T, weights, depth, acc |
| 3DGS representation | gs/model.py | Custom | Parameters + activations |
| Projection / visibility / cull | gs/project.py | Custom | EWA, radii, frustum |
| Sort + rasterize + composite | gs/raster_*.py | Custom (gsplat oracle) | Backends |
| Densify / prune / reset | gs/density.py | Custom | Clone/split/prune + Adam state surgery |
| Losses | train/losses.py, core/ssim.py | Custom | MSE, L1, D-SSIM |
| Trainers, schedulers | train/ | Custom | Loops, LR schedules, ckpt |
| Metrics | eval/metrics.py | Custom PSNR/SSIM; Ext LPIPS | Held-out eval |
| Export | export/ | Custom | PLY (Inria), .splat, video |
| Viewer | viewer/ | Ext (viser) → optional custom | Navigation |
| GPU/perf | utils/profiling.py | Custom | Timers, VRAM, benchmarks |
| Config / logging | utils/ | Ext (dataclass+YAML, TensorBoard) | Reproducibility |

---

## Part 3 — NeRF from scratch

**Conventions (global):** internal camera = OpenCV (x right, y down, z forward); poses stored as c2w 4×4 float32. Blender/NeRF transforms.json (OpenGL: y up, −z forward) is converted at the loader: `c2w[:3, 1:3] *= -1`. No other module handles conventions.

### N1 Camera model and coordinates

- **Math:** pixel (u,v) with centers at (i+0.5, j+0.5); K = [[fx,0,cx],[0,fy,cy],[0,0,1]]. Camera-space ray d_c = ((u−cx)/fx, (v−cy)/fy, 1). x_c = R_w2c x_w + t; c2w = w2c⁻¹; camera center o = c2w[:3,3].
- **I/O:** Camera(K, c2w, W, H) → projection/unprojection functions.
- **Edge cases:** c2w vs w2c confusion; half-pixel offset; non-square pixels; images resized (scale K by the resize factor, including the +0.5 shift).
- **Tests:** project(unproject(p)) = p to 1e-5; random-rotation orthonormality; known cube corners project to known pixels; loader flips reproduce a hand-computed pose.

### N2 Ray generation

- **Math:** d_w = R_c2w d_c, normalized so ‖d_w‖=1, then t is Euclidean distance. o_w = c2w[:3,3].
- **Algorithm:** precompute per-image pixel grid; sample rays from a flat index (img, y, x) on GPU; no per-ray Python loops.
- **I/O:** (cam_idx, pix_idx) → rays_o [B,3], rays_d [B,3], RGB target [B,3].
- **Edge cases:** rays through image corners for wide FOV; RGBA compositing to white background for synthetic data.
- **Tests:** center-pixel ray = optical axis; ray through a known 3D point hits it (distance < 1e-5); all ‖d‖=1.

### N3 Bounds and sampling

- **Math:** scene normalized so the object sits in a sphere of radius R (or AABB). t_near, t_far from ray–sphere intersection (|o+td|²=R²); rays that miss get a zero-weight flag. Synthetic Lego: fixed near=2, far=6.
- **Stratified:** t_i = t_n + (i + U_i)/N · (t_f − t_n), U_i ∼ U[0,1); deterministic midpoints at eval.
- **Hierarchical (inverse-CDF):** from coarse weights w_i, pdf_i = (w_i + ε)/Σ(w+ε) over bins, cdf = cumsum, draw u, t = searchsorted(cdf,u) + linear interpolation inside the bin; detach the samples; merge with the coarse samples and sort.
- **Edge cases:** all-zero weights (add ε); searchsorted bounds (clamp indices); rays missing the bounds.
- **Tests:** sampling a known pdf reproduces its histogram (χ² test); samples sorted and within [t_n,t_f]; no gradient through samples.

### N4 Positional encoding

- **Math:** γ(p) = [p, sin(2⁰πp), cos(2⁰πp), …, sin(2^{L−1}πp), cos(2^{L−1}πp)]; L=10 (position), L=4 (direction). Output dim 3+3·2L. Counters spectral bias (the MLP learns low frequencies first).
- **Tests:** shape/dim; γ(0) known; frequency ordering; gradient finite.
- **Extension:** hash grid (N9), IPE (later).

### N5 MLP architecture

- **Structure (original):** 8 layers × 256, ReLU, skip concat of γ(x) at layer 5 → σ head (1) + 256-d feature; concat γ(d) → 128-unit layer → RGB (3, sigmoid).
- **Density:** σ = softplus(raw + b) (or ReLU + noise σ~N(0,1) during training on real data). Density depends on x only; color on (x, d), which enforces multi-view-consistent geometry.
- **Init:** Kaiming/Xavier; bias so initial σ is small (avoid opaque start).
- **Edge cases:** dead ReLU (σ≡0 for every sample → zero gradients: fix with softplus/init); NaNs from exp.
- **Tests:** output shapes; σ ≥ 0; RGB ∈ (0,1); σ invariant to d; overfit test: 1 image, 1 minute → PSNR > 20.

### N6 Volume rendering

- **Continuous:** C(r)=∫ T(t)σ(t)c(t)dt, T(t)=exp(−∫σ).
- **Discrete:** δ_i = t_{i+1}−t_i (last δ = 1e10 or handle background), α_i = 1−exp(−σ_iδ_i), T_i = Π_{j<i}(1−α_j), w_i = T_iα_i, Ĉ = Σ w_i c_i, depth = Σ w_i t_i, acc = Σ w_i. Compute T with an exclusive cumprod (or exp(exclusive_cumsum(−σδ)), more stable). Background: Ĉ + (1−acc)·bg.
- **Note:** dependence on δ makes σ scale-aware; use ray-length-normalized t consistently (unit-norm d).
- **Tests (analytic):** constant σ over a slab of thickness L → acc = 1−exp(−σL) (1e-5); σ=0 → color = bg; one opaque sample → color = its color, weights sum to 1; Σw ≤ 1 always; float64 gradcheck.

### N7 Loss

L = Σ_rays ‖Ĉ_c − C‖² + ‖Ĉ_f − C‖² (coarse and fine). PSNR = −10 log₁₀ MSE. Optional later: distortion loss, depth loss. **Tests:** loss zero on identical inputs; PSNR of a known-noise image matches the analytic value.

### N8 Training loop, optimizer, batching

Adam (β=(0.9,0.999), ε=1e-7 … 1e-8), lr 5e-4 → 5e-5 exponential decay over the run, 4096 rays/batch (fewer if VRAM-limited; chunk MLP in chunks of 2¹⁶ points), 100k–300k iters for full quality (recipe, verify). Rays sampled uniformly over all training pixels (all images cached on GPU as uint8, ≈100 MB at 200 images × 400²).

**AMP:** run MLP in bf16 if desired; volume rendering in fp32.

Validation every N iters on 3–5 held-out views (full-image rendering in ray chunks, no_grad).

**Checkpoint:** model + optimizer + step + RNG + config + normalization transform. Resume must be bit-reasonable (loss continues smoothly).

**Edge cases:** NaN guard (skip step, log); memory from [B,N,3] intermediates; validation view accidentally in train split.

**Tests:** overfit 1 image; overfit 10 rays to ~0 loss; resume test (train 200, save, resume 100 ≈ train 300 within tolerance); determinism under a fixed seed.

### N9 Hash-grid variant (for phone object data)

- **Math:** per level l, res_l = ⌊N_min·b^l⌋, b=exp((ln N_max − ln N_min)/(L−1)); for the 8 voxel corners, index = (x⊕y·2654435761⊕z·805459861) & (T−1) (int64 wraparound is fine with the mask), or dense indexing where (res+1)³ ≤ T; trilinear interpolate F=2 features; concatenate the 16 levels; small MLP (density: 1 hidden ×64; color: 2 hidden ×64, verify) + SH/PE view-direction encoding.
- Init table U(−1e-4,1e-4); Adam lr 1e-2, ε=1e-15. Occupancy grid to skip empty space (optional).
- **Tests:** on a level without collisions, a trilinear lookup of a planted linear function is exact; index range valid; gradients hit only touched entries.
- **Pure PyTorch cost:** ~33M gathers/iter at 262k samples×16 levels×8 corners — workable, slower than tiny-cuda-nn (benchmark in Phase 6).

### N10 Novel-view rendering

Spiral / interpolated path: pose interpolation with slerp for rotation, linear for translation; render in ray chunks, no_grad; save mp4 + depth video. **Tests:** rendering a training pose reproduces the training PSNR.

---

## Part 4 — 3D Gaussian Splatting from scratch

**Parameters per Gaussian:** μ∈R³, log s∈R³, q∈R⁴ (normalized on use), logit o∈R, SH [16,3] (DC + rest) → 59 floats. Store as separate nn.Parameters (separate LR groups).

### G1 Covariance

R(q) from unit quaternion (w,x,y,z); Σ = R S Sᵀ Rᵀ, S=diag(exp(log s)) (PSD by construction). **Tests:** R orthonormal, det=+1; Σ symmetric PSD; eigenvalues = s²; float64 gradcheck; normalization gradient at q≈0 guarded.

### G2 Initialization

Points from SfM after filtering (track length ≥ 3, reprojection error < 2 px, drop points beyond ~5× median camera distance). μ=point, DC = (rgb−0.5)/0.28209479, higher SH = 0, q=(1,0,0,0), o=0.1 (stored as logit), isotropic s = sqrt(mean of squared distances to 3 nearest neighbors) (clamp ≥ 1e-7), stored log. kNN via torch.cdist in chunks or scipy.cKDTree. Random-init fallback: uniform in the camera-extent bounding box, rgb random (a documented weaker baseline for the init experiment).

### G3 Camera projection and projected covariance (EWA)

t = W μ + T (camera space); cull t_z < 0.2; clamp t_x/t_z, t_y/t_z to ±1.3·tan(FOV/2) for the Jacobian only.

μ' = (fx t_x/t_z + cx, fy t_y/t_z + cy).

J = [[fx/t_z, 0, −fx t_x/t_z²],[0, fy/t_z, −fy t_y/t_z²]]; Σ' = (J W) Σ (J W)ᵀ (2×2); Σ̃ = Σ' + 0.3·I (dilation).

Conic = Σ̃⁻¹ = [[c,−b],[−b,a]]/det for Σ̃=[[a,b],[b,c]]; radius = 3·sqrt(λ_max), λ = mid ± sqrt(max(0.1, mid²−det)), mid=(a+c)/2. Skip det ≤ 0.

**Tests:** a Gaussian on the optical axis, isotropic → circular with σ' = f·s/z; compare against a numerical Jacobian (finite differences in float64); dilation adds exactly 0.3; behind-camera Gaussians are culled; gsplat oracle comparison of means2d, conics, radii.

### G4 View-dependent color (SH)

c(d) = 0.5 + Σ_{l≤L}Σ_m k_l^m Y_l^m(d), clamped ≥ 0. d = normalize(μ − camera_center) (camera → Gaussian, verify against the oracle; odd bands flip sign if reversed). Degree activated progressively: +1 every 1000 iters to 3. **Tests:** DC-only = constant 0.28209479·k; orthonormality of Y_lm by numerical integration on the sphere; comparison against gsplat's SH; flipping the direction convention makes the oracle test fail (proves the test is sensitive).

### G5 Sorting and visibility

Depth = view-space t_z of the center (an approximation: no per-pixel sorting, so popping is possible). Global argsort(depth), then stable-sort the (tile, gaussian) pairs by tile, preserving depth order inside tiles. Tile = 16×16 px; a Gaussian covers tiles overlapped by its [μ'−r, μ'+r] box.

### G6 Alpha compositing (per pixel, front-to-back)

α_i = min(0.99, o_i·exp(−½ Δᵀ Σ̃⁻¹ Δ)), Δ = p − μ'_i; contributions with α<1/255 are skipped; C = Σ c_iα_iΠ_{j<i}(1−α_j) + T_final·bg. (The reference stops when T<1e-4.) This is the same quadrature as NeRF (α = 1−exp(−σδ)) — with Gaussian opacity replacing density integration. Depth output: Σ w_i z_i (alpha-weighted) and acc.

### G7 Pure-PyTorch tiled rasterizer (algorithm)

1. Project all Gaussians (vectorized) → means2d, conics, radii, depth, colors, opacity, valid.
2. Sort valid Gaussians by depth.
3. Compute tile rectangle per Gaussian; n_i = area; repeat_interleave to create (tile_id, gaussian_id) pairs; stable-sort by tile.
4. Per-tile offsets from bincount/cumsum; cap K (e.g. 256–512) nearest Gaussians per tile (log the overflow fraction — a documented approximation); scatter into a padded [T,K] index tensor (sentinel = zero-opacity dummy).
5. For a chunk of tiles: gather params → α [Tc,256,K] → logT = exclusive_cumsum(log(1−α)) → w = α·exp(logT) → rgb = einsum(w, c), plus bg·exp(logT_last+log(1−α_last)).
6. Stitch tiles into the image. Chunk over tiles with torch.utils.checkpoint if VRAM-bound.

**Memory:** dense per-tile tensors ≈ Tc·256·K·4 B each; e.g. 256 tiles (256×256 image), K=256 → 67 MB per intermediate.

**Naive reference backend:** [N,P] dense (tiny scenes only, unit tests). N=10⁵ × P=65k px = 26 GB for one tensor — hence tiling.

**Patch training:** render a random tile-aligned 256×256 crop per iteration (cull Gaussians by crop box) — bounds memory regardless of full image size. Densification statistics come only from visible Gaussians in that crop.

### G8 Optimization

**Loss:** 0.8·L1 + 0.2·(1−SSIM) (11×11 Gaussian window, σ=1.5; own implementation).

Adam with per-group LR (reference, verify): position 1.6e-4→1.6e-6 (exp decay, × scene extent), SH DC 2.5e-3, SH rest 1.25e-4, opacity 0.025, scale 5e-3, rotation 1e-3; ε=1e-15. 30k iterations reference; MVP 15–20k with a Gaussian cap.

**Screen-space gradient:** means2d.retain_grad(); accumulate ‖∇_{means2d}L‖ per visible Gaussian and a visibility count. Trap: the reference gradient is with respect to NDC. If you differentiate w.r.t. pixels, multiply x by W/2 and y by H/2 before thresholding at 2e-4, or the threshold is meaningless (verify against gsplat's grad2d convention).

### G9 Density control (every 100 iters, iter 500–15000)

g = accumulated_grad / count.

- **Clone** if g ≥ 2e-4 and max(s) ≤ 0.01·extent: duplicate the Gaussian.
- **Split** if g ≥ 2e-4 and max(s) > 0.01·extent: replace with 2 Gaussians sampled μ + R S ε, ε∼N(0,I), scale /1.6.
- **Prune:** o < 0.005; after the first reset also max screen radius > 20 px or max(s) > 0.1·extent.
- **Opacity reset** every 3000 iters (until 15000): o ← min(o, 0.01).

**Optimizer state surgery:** for each param group, concatenate/mask exp_avg, exp_avg_sq; new clones get zero state. This is where subtle bugs live.

**Tests:** counts before/after match hand calculation on a toy set; optimizer state shapes track parameter shapes; splitting preserves the rendered image approximately (PSNR > 40 dB between pre/post-split renders).

**Budget cap (MVP):** if N > N_max, keep only the top-k by gradient among candidates.

### G10 Exports

Inria PLY: x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3 (raw logit/log/unnormalized quaternion; SH rest is channel-major, verify layout). **Test:** round-trip write→read; load in SuperSplat/gsplat.

### G11 Pure PyTorch vs custom kernels

| Component | Pure PyTorch (MVP) | Kernel later |
|---|---|---|
| quat→R, Σ, projection, SH | Yes, permanent | No |
| Tile pair generation + sort | sort (OK) | Radix-sort CUDA |
| Per-pixel compositing | Padded tensors, autograd | Fused fwd/bwd (recompute, no [T,256,K] storage) — main speedup |
| Densification | Yes | No |
| SSIM | Yes | Fused optional |

**Learning ladder for the kernel step:** (1) custom torch.autograd.Function with hand-written backward for the compositor in PyTorch, verified by gradcheck; (2) Triton kernel; (3) optional CUDA. Not required for MVP.

---

## Part 5 — Phone video pipeline

**Capture conditions:** 1080p 30 fps; lock exposure/focus/WB if possible; stabilization off; single lens, no zoom; slow steady motion (≲ 0.3 m/s); ≥ 70% overlap between kept frames; translate (parallax), don't only rotate in place; object: 2–3 orbit rings at different elevations, 360°; room: walk the perimeter facing inward and outward, plus some cross-room passes; even lighting, no moving people, avoid mirrors, glass, blank white walls.

**Frame selection (preprocess/select.py):**

1. Decode at ≤ 1080p; convert to grayscale at ~½ res for scoring.
2. Blur score = variance of the Laplacian; compare within a sliding window (~0.3–0.5 s) and keep the sharpest; reject frames below the 20th percentile of the whole video.
3. Redundancy: keep a frame only if median optical-flow / feature displacement to the last kept frame exceeds a threshold (≈ 3–5% of image width) or the time gap exceeds a maximum.
4. Target 150–300 frames (object), 250–500 (room).
5. Log a selection report (histograms, kept indices).

**Calibration/poses:** pycolmap: SIFT → exhaustive matching → incremental mapper, SIMPLE_RADIAL camera (single camera for the whole video; EXIF focal as initialization) → image_undistorter → PINHOLE images and cameras. GLOMAP/global mapper only if runtime is a problem (verify current status in COLMAP).

**Scale ambiguity:** SfM units are arbitrary. Training doesn't need metric scale; normalization sets extent. Metric scale (extension) = known distance between two triangulated points/ArUco marker.

**Coordinate handling:** COLMAP world → apply Sim3 (center = mean camera position or least-squares closest point of optical axes for object captures; scale so max camera distance ≈ fixed; optional gravity alignment via PCA of camera up vectors) → store (s, R, t) in SceneData and checkpoints. extent = 1.1·max‖c_i − mean(c)‖ computed on cameras only.

**Failure cases → diagnostics (diagnose_sfm.py):**

| Symptom | Metric | Action |
|---|---|---|
| Tracking fails / multiple models | registered < 90%; >1 model | More frames, exhaustive matching, remove blur, recapture with more overlap |
| Drift / warped trajectory | reproj error > 1.5 px mean; trajectory not smooth/closed | Loop closure, recapture, check intrinsics vs EXIF |
| Wrong intrinsics | focal far from EXIF/expected FOV; distortion k1 huge | Fix lens; avoid lens switching |
| Sparse points | < ~2k points; mean track length < 3 | Textureless scene; add texture/light |
| Train PSNR high, test PSNR low | gap > 5 dB | Pose error or overfitting; check held-out blocks |
| Floaters near cameras | high opacity, small depth | Poses/blur; near-plane; opacity-reset schedule |
| Smeared surfaces | uniformly blurry renders | Blur in inputs, poor poses, resolution too low |

**Poses "failed" fallback ladder:** re-select frames → exhaustive matching with larger feature count → masks for dynamic regions → VGGT pose front-end (Phase 14) → recapture.

**Pose-vs-model bug separation:** always run the same COLMAP data through your renderer and through gsplat. If gsplat also fails, the data is bad; if only yours fails, the renderer is buggy.

---

## Part 6 — Software structure

```
nvs3d/
├── pyproject.toml
├── configs/
│   ├── base.yaml                # seeds, logging, device
│   ├── nerf_blender.yaml
│   ├── nerf_hash_object.yaml
│   ├── gs_blender.yaml
│   └── gs_phone_room.yaml
├── src/nvs3d/
│   ├── core/
│   │   ├── cameras.py           # Camera, projection, conventions
│   │   ├── transforms.py        # SE3, Sim3, slerp, Umeyama
│   │   ├── rays.py
│   │   ├── quaternion.py
│   │   ├── sh.py
│   │   └── ssim.py
│   ├── data/
│   │   ├── scene.py             # SceneData, Split
│   │   ├── blender_io.py
│   │   ├── colmap_io.py
│   │   └── dataset.py           # RayDataset, ViewDataset
│   ├── preprocess/
│   │   ├── extract_frames.py
│   │   ├── select.py            # blur + redundancy
│   │   ├── run_sfm.py           # pycolmap wrapper
│   │   ├── diagnose_sfm.py
│   │   └── normalize.py
│   ├── nerf/
│   │   ├── encoding.py          # positional, hashgrid
│   │   ├── field.py             # NeRFMLP, HashNeRF (RadianceField protocol)
│   │   ├── sampler.py
│   │   ├── volume.py
│   │   └── render.py
│   ├── gs/
│   │   ├── model.py             # GaussianModel
│   │   ├── init.py
│   │   ├── project.py
│   │   ├── raster/{base.py, naive.py, tiled.py, gsplat_backend.py, triton_backend.py}
│   │   ├── density.py           # Densifier + optimizer surgery
│   │   └── render.py
│   ├── train/{nerf_trainer.py, gs_trainer.py, losses.py, schedulers.py}
│   ├── eval/{metrics.py, evaluate.py, benchmark.py}
│   ├── export/{ply.py, splat.py, video.py}
│   ├── viewer/{viser_app.py}
│   └── utils/{config.py, logging.py, checkpoint.py, seed.py, profiling.py}
├── tests/{unit, math, render, gradcheck, integration, regression}/
├── scripts/{prepare_video.py, train.py, evaluate.py, render_path.py, export.py, bench.py}
├── web/viewer/                  # optional custom viewer
└── runs/                        # gitignored
```

**Key interfaces (contracts):**

```python
@dataclass
class Camera: K: Tensor[3,3]; c2w: Tensor[4,4]; W: int; H: int   # OpenCV convention

@dataclass
class SceneData:
    images: Tensor|list[Path]; cameras: list[Camera]
    points_xyz: Tensor|None; points_rgb: Tensor|None
    split: dict[str, list[int]]          # "train", "test"
    extent: float; norm: Sim3            # COLMAP-world -> training-world

class RadianceField(Protocol):
    def forward(self, x: Tensor[N,3], d: Tensor[N,3]) -> tuple[Tensor[N], Tensor[N,3]]  # sigma, rgb

class Rasterizer(Protocol):
    def __call__(self, g: GaussianParams, cam: Camera, crop: Box|None, cfg) -> RenderOut
# RenderOut: rgb[H,W,3], depth, alpha, aux{means2d, radii, visible_mask, n_pairs, tile_overflow}
```

**Config:** typed dataclasses + YAML + CLI overrides (tyro or OmegaConf); every run writes the resolved config. Training config (3DGS): iters, sh_degree_max, lr groups, densify_{from,until,interval,grad_thr,percent_dense}, opacity_reset_interval, lambda_ssim, backend, tile K, patch size, max_gaussians, image scale. Rendering config: backend, near, bg color, output size, chunk sizes.

**Checkpoint format:** torch.save({version, kind:"nerf"|"gs", config_yaml, step, state_dict, optim_state, rng_state, scene_meta:{extent, norm(Sim3), sh_degree, K/size}, git_hash}).

**Logging structure:** runs/&lt;exp&gt;/&lt;timestamp&gt;/{config.yaml, metrics.jsonl, tb/, renders/, ckpt/, report.md}; scalars: loss, PSNR, N_gaussians, VRAM, it/s, tile overflow %, mean opacity.

**Experiment tracking:** TensorBoard (or wandb optional); one config = one run; seeds in config.

**Testing structure:** unit (shapes, conventions), math (analytic), gradcheck (float64), render (vs oracle), integration (tiny end-to-end), regression (frozen PSNR baselines within tolerance).

---

## Part 7 — Build roadmap

Each phase: Goal / Tasks / Math / Tests / Output / Done / Deps / Failure modes.

**P0 — Scaffolding.** Repo, env, configs, logging, seed, checkpoint IO, pytest, CI-lite. **Done:** pytest runs; a dummy run writes a run dir. **Deps:** none. **Failures:** CUDA/torch mismatch.

**P1 — Geometry core + data loaders.** Camera, SE3/Sim3, rays, Blender + COLMAP loaders, convention flip. **Math:** pinhole, homogeneous coords, c2w/w2c. **Tests:** N1/N2 tests; render the loaded poses as a camera-frustum plot (viser/matplotlib) — Lego cameras form a hemisphere facing the center. **Done:** ray visualization hits the object. **Failures:** flipped y/z, transposed rotation, half-pixel.

**P2 — NeRF components.** PE, MLP, sampler, volume renderer. **Tests:** N4–N6 (analytic slab, gradcheck). **Done:** all pass; render of a random field is finite. **Failures:** exclusive cumprod off by one.

**P3 — NeRF trainer (coarse only), synthetic.** Overfit one image then Lego at 100×100→200×200. **Done:** Lego 200×200 PSNR ≥ ~26 dB after ~30–50k iters (heuristic); spiral video shows plausible 3D. **Failures:** white background not composited, dead ReLU, wrong near/far.

**P4 — Hierarchical NeRF + evaluation.** Fine network, inverse-CDF, metrics (PSNR/SSIM own; LPIPS external). **Done:** fine > coarse PSNR by ≥ 1 dB; eval script produces report.md. **Failures:** gradients flowing through sampled t; NaN in pdf.

**P5 — Video preprocessing + SfM.** Extract, blur/redundancy selection, pycolmap, undistort, diagnostics, normalization, SceneData. **Tests:** synthetic video made by rendering a Blender scene → poses recovered within tolerance (Umeyama-aligned ATE); unit tests of the blur scorer on synthetic blur. **Done:** `prepare_video.py phone.mp4` yields a valid SceneData + diagnostics report. **Failures:** multi-lens, wrong sequence, <90% registered.

**P6 — NeRF on own object capture (hash-grid).** Implement N9; benchmark it/s. Masking optional (SAM, external). **Done:** held-out PSNR sane; spiral video plausible. **Failures:** background fog; bounds misfit.

**P7 — 3DGS math.** Quaternion/covariance, projection, SH, SSIM. **Tests:** G1, G3, G4, numeric Jacobian, oracle comparison. **Done:** forward-only comparisons agree (~1e-5 in fp64). **Failures:** quaternion order, SH sign.

**P8 — Naive rasterizer + 2D/3D overfit.** Dense [N,P] compositing on tiny images. (a) 2D warm-up: fit a 128×128 image with 2k 2D Gaussians; (b) fit a single 3D view then multi-view on 1k Gaussians. **Done:** (a) PSNR > 28 dB; (b) view-fit works. **Failures:** sort direction, α clamp, gradients through masks.

**P9 — Tiled rasterizer.** G7 with cap K, chunking, patch training. **Tests:** equal to the naive backend (< 1e-4) when K is uncapped; vs gsplat; gradcheck; memory/it-s benchmark table. **Done:** 256×256 crop at 200k Gaussians fits in 8 GB. **Failures:** tile overflow, off-by-one tile edges, empty tiles.

**P10 — 3DGS training without densification.** SfM init, param groups, LR schedule, L1+SSIM, SH progression. **Done:** on Lego (init from random points/synthetic cloud) loss decreases; own scene converges to blurry-but-correct. **Failures:** LR not scaled by extent, opacity saturation.

**P11 — Density control.** G9 with optimizer surgery, gradient accumulation (NDC scaling), cap. **Tests:** G9 tests. **Done:** Lego 400×400 PSNR ≥ ~30 dB (heuristic; reference ≈ 33 at 800×800); N grows then plateaus. **Failures:** Gaussian sprawl, OOM, optimizer state desync.

**P12 — Full 3DGS on own room + oracle comparison.** Same SceneData through gsplat/Nerfstudio Splatfacto; compare PSNR/SSIM/LPIPS/N/time. **Done:** within ~1.5 dB of the oracle (heuristic). **Failures:** convention mismatch between backends (check c2w, dilation, SH direction).

**P13 — Export + navigation.** PLY, .splat, viser viewer with own renderer, spiral/hero video; load the PLY in SuperSplat. **Done:** navigable scene, PLY loads in a third-party viewer identical to your renders. **Failures:** SH layout, quaternion order/scale activation in PLY.

**P14 — Performance & quality upgrades (optional, each gated by an experiment).** Custom autograd.Function compositor; Triton kernel; gsplat backend switch; depth prior; per-image appearance affine; VGGT pose front-end comparison. **Done:** each has a before/after table.

**P15 — Optional custom web viewer** (WebGL2 instancing + worker counting sort with 65,536 depth buckets).

---

## Part 8 — Learning roadmap (only what the project needs)

| Before | Understand | Self-check |
|---|---|---|
| P1 | Homogeneous coords, SE(3), rotation matrix properties, quaternion→R, pinhole K, c2w vs w2c, OpenCV vs OpenGL axes | Project a cube by hand; explain what c2w[:3,3] is |
| P2 | Beer–Lambert / volume rendering derivation (T(t)=exp(−∫σ)), quadrature, why α=1−exp(−σδ); Monte-Carlo/stratified sampling; inverse-transform sampling; spectral bias, Fourier features | Derive Σ w_i ≤ 1; derive the CDF sampler |
| P3–4 | MSE/PSNR relation, Adam mechanics (moments, bias correction), exponential LR decay, overfitting/holdout, autograd basics, gradcheck | Write Adam by hand for one parameter |
| P5 | Epipolar geometry (E/F, RANSAC), triangulation, bundle adjustment/reprojection error (concept only), SIFT at concept level, intrinsics/distortion, scale ambiguity, Umeyama alignment | Explain why monocular scale is unrecoverable |
| P6 | Multiresolution hashing, trilinear interpolation, collisions | Implement trilinear by hand for a 2D grid |
| P7 | Multivariate Gaussian; affine transform Σ→AΣAᵀ; PSD; Jacobian linearization (first-order Taylor); EWA splatting idea; SH basis (degrees 0–3 only) | Derive Σ'=JWΣWᵀJᵀ; compute J numerically |
| P8–9 | Alpha compositing ≡ discretized volume rendering; ordering/depth sort; tile-based binning; memory analysis of tensors; autograd graph memory; retain_grad | Compute VRAM for [T,256,K] |
| P11 | Gradient as a "needs more capacity" signal; densification logic; optimizer-state bookkeeping; SSIM definition | Explain why clone vs split |
| P14 | Custom autograd.Function, recompute-in-backward, Triton programming model, mixed precision | Derive the backward of front-to-back compositing |

**Omit:** general ML theory, CNN architectures, generative models, SLAM theory.

---

## Part 9 — Testing and validation

| Layer | Test | Proves |
|---|---|---|
| Unit | shapes, dtypes, device, determinism | plumbing |
| Math | slab transmittance; isotropic Gaussian projection; SH DC; rotation properties; Umeyama recovers known Sim3 | formulas |
| Gradcheck | float64 torch.autograd.gradcheck for covariance, projection, SH, compositing, volume render (with α-threshold disabled to avoid discontinuities) | differentiability |
| Oracle | your project, SH, rasterizer vs gsplat: means2d, conics, radii, rgb, depth, alpha, and gradients wrt μ, s, q, o, SH | correct vs reference (dev-only dependency) |
| Rendering | single Gaussian at known location gives expected pixel; two overlapping opaque Gaussians obey depth order; camera translation moves it correctly; behind-camera culled | geometry |
| Synthetic scenes | (a) planted 3DGS scene → render → fit from random init → recover; (b) Blender Lego/known cameras; (c) a cube with 4 cameras at known poses | end-to-end |
| Known cameras | Lego poses; ring of cameras around origin: check look-at | conventions |
| Overfit | 1 image, 10 rays, 1 view | optimizer/loss |
| Reconstruction | PSNR/SSIM/LPIPS on held-out; ATE/RPE after Umeyama alignment for poses; depth error on synthetic | quality |
| Visual | side-by-side GT/render/error-map/depth; spiral videos | artifacts |
| Performance | it/s, VRAM peak, Gaussians, FPS vs resolution & N; tile overflow % | scaling |
| Regression | frozen small scene; PSNR ≥ baseline − 0.3 dB; N_gaussians within 10% | no silent drift |

**Held-out protocol on phone video:** hold out every 8th selected frame and exclude its ±1 neighbors from training (avoid near-duplicate leakage).

**How to know a component is correct:** it passes an analytic test, a gradcheck, and an oracle test, in that order. A component with only "the picture looks right" is not done.

---

## Part 10 — Performance and hardware

| | MVP | Advanced |
|---|---|---|
| GPU | NVIDIA 8–12 GB (Colab/Kaggle T4 16 GB works) | 24 GB (3090/4090-class) |
| CPU / RAM | 8 cores / 16 GB (COLMAP CPU SIFT is slower) | 16 cores / 32–64 GB |
| Storage | ~20 GB | 100+ GB |
| Resolution | ≤ ~960 px long side; 256² patches | 1080p, full-frame |
| Gaussians | 100k–500k | 1–5M |
| NeRF | vanilla 200–400 px, hours; hash-grid ~20–60 min (estimate) | Zip-NeRF-style, out of scope |
| 3DGS train | pure PyTorch: expect 1–5 it/s (estimate; benchmark P9); 15–20k iters ≈ 1–5 h | gsplat/Triton: tens of it/s, ~10–30 min |
| Render | own tiled: sub-second per frame | 100+ FPS via CUDA |

**Memory:** 59 floats × 4 B ≈ 236 B/Gaussian; training ≈ 4× (params+grads+m+v) → 1M ≈ 0.94 GB, plus rasterizer intermediates (dominant). Levers: patch training, downscale 2–4×, lower SH degree, cap N, tile chunking + checkpointing, bf16 only for non-covariance math (keep projection fp32). Avoid CPU↔GPU transfers in the loop: cache images on GPU as uint8, convert on the fly; avoid .item() per iteration.

---

## Part 11 — External dependencies

| Component | Build ourselves? | Existing option | Reason | Recommended approach |
|---|---|---|---|---|
| Camera/ray/coordinate math | Yes | PyTorch3D, kornia | Bug source #1, core learning | Custom |
| PE, MLP, volume rendering, sampling | Yes | nerfstudio | Core learning | Custom |
| Hash grid | Yes (pure PyTorch) | tiny-cuda-nn | Learn the idea; accept slowness | Custom; tcnn only as benchmark |
| Quaternion/covariance/projection/SH | Yes | gsplat | Core learning | Custom; gsplat as oracle |
| Tile sort + compositing | Yes (tiled PyTorch) | gsplat, Inria CUDA | Central learning | Custom + gsplat backend option |
| Densification/pruning | Yes | gsplat strategies | Central learning | Custom |
| SSIM, PSNR | Yes | pytorch-msssim | Small | Custom, validate against skimage |
| LPIPS | No | lpips | Pretrained | Use |
| SfM/matching/BA | No | COLMAP/pycolmap, GLOMAP | Multi-year engineering | Use |
| Feed-forward poses | No | VGGT, MASt3R | Optional comparison | Use (P14) |
| Frame decode | No | OpenCV/ffmpeg | Infra | Use |
| Blur/redundancy selection | Yes | — | Small, high value | Custom |
| Undistortion | No | COLMAP/OpenCV | Infra | Use |
| Monocular depth, SAM | No | Depth Anything V2, SAM 2 | Pretrained | Use in extensions |
| Meshing | No | Open3D, marching cubes | Tooling | Use |
| Viewer (MVP) | No | viser, SuperSplat | Not core | Use; custom optional |
| PLY IO | Yes | plyfile | Trivial, format learning | Custom (or plyfile) |
| Config/logging | No | dataclass+YAML, TensorBoard | Infra | Use |

---

## Part 12 — Advanced capability roadmap

| Extension | Foundation | New components | Difficulty | Dependencies | Where |
|---|---|---|---|---|---|
| Semantic 3D | Trained 3DGS | Lift SAM/CLIP/DINO features to per-Gaussian features; feature rendering channel | A | Rasterizer supports N-channel output | Extension |
| Object-level | Semantics | Mask→Gaussian selection; per-object export | A | Semantics | Extension |
| Measurements | Metric scale | Scale reference; point picking; depth/mesh raycast | I–A | Depth rendering | Small core add-on |
| Floor plan | Metric + planes | RANSAC plane fit; wall/floor extraction | A | Mesh/points | Extension |
| Scene understanding / AI querying | Semantics | Language feature field or scene graph + VLM | A–R | Semantics | Extension |
| Editing / insertion / removal | Object-level | Gaussian transforms (rotate SH), inpainting | A | Masks | Extension |
| Relighting | Normals + BRDF | Inverse rendering | R | Geometry | Separate |
| Multi-session | Poses | Registration (Sim3 align, ICP), merged model | A–R | Global poses | Extension |
| Temporal/change detection | Multi-session | Render/depth compare | A–R | Registration | Extension |
| AR/VR | Exports | WebXR viewer, compression | A | Web viewer | Separate |
| Digital twins | Metric+semantic+updates | All above | R | — | Separate |
| Real-time rendering | Tiled backend | Triton/CUDA rasterizer, LOD | A | P14 | Core (Phase 14) |
| Large-scale | Hierarchical/chunking | Scene partition | R | Real-time | Separate |

**Foundation guarantees (so nothing is rebuilt):** (1) rasterizer output channels are generic (extra_features slot); (2) Sim3 normalization is stored; (3) Rasterizer and RadianceField are protocols; (4) exports keep raw parameters.

---

## Part 13 — Experiments

| # | Hypothesis | Variables | Method | Metrics | Interpretation |
|---|---|---|---|---|---|
| E1 | 3DGS beats NeRF at fixed wall-clock | representation | Same scene, equal time budget | PSNR/LPIPS/time/VRAM/FPS | 3DGS wins on render speed; check quality parity |
| E2 | Stratified + hierarchical beats uniform | samples/ray, hierarchical on/off | Lego | PSNR vs samples | Hierarchical wins per-sample; diminishing returns |
| E3 | Higher PE frequency improves detail until noise | L ∈ {4,6,8,10,12}; hash vs PE | Lego 200 px | PSNR, sharpness | Too high L → noise/overfit |
| E4 | SfM init beats random | init: SfM / random / dense | Same scene, 3 seeds | PSNR, N, convergence curve | Bigger gap on textureless scenes |
| E5 | Pose noise degrades quality super-linearly | σ_rot, σ_trans perturbation on GT poses | Lego (GT known) | PSNR vs noise, ATE | Set tolerance for capture quality; test learnable pose deltas |
| E6 | Resolution vs quality vs cost | scale ∈ {¼,½,1}; progressive | Fixed iters | PSNR/time/VRAM | Diminishing returns; progressive saves time |
| E7 | Densification thresholds control N and quality | grad_thr, percent_dense, interval, cap | Grid | PSNR vs N | Pareto curve; sprawl at low thresholds |
| E8 | SH degree improves view-dependent effects | deg 0–3 | Specular scene | PSNR, size | Gains small on diffuse rooms; overfit risk |
| E9 | Patch size / tile cap K affect fidelity/speed | K ∈ {64,128,256,512}, patch | Own rasterizer vs oracle | Error vs oracle, it/s, overflow% | Choose K where overflow < 1% |
| E10 | Capture conditions | blur, fps, overlap, motion speed | Same room, 4 captures | Registered %, PSNR, floaters | Rank what matters most |
| E11 | COLMAP vs VGGT poses | pose front-end | Same frames | ATE, PSNR, time | VGGT faster; check accuracy (verify) |
| E12 | Depth-prior regularization | λ_depth | Room | PSNR, depth quality | Helps textureless walls; hurts with bad scale alignment |

Every run: fixed seeds (≥ 3 for small effects), same held-out protocol, config logged.

---

## Part 14 — Master execution order

1. pyproject.toml, src/nvs3d/utils/{config,logging,seed,checkpoint}.py, tests/. Run pytest.
2. core/transforms.py (SE3, Sim3, Umeyama, slerp) + tests.
3. core/cameras.py (Camera, project/unproject, OpenGL→OpenCV) + tests.
4. core/rays.py + ray tests.
5. data/blender_io.py, data/scene.py; frustum visualization script; check Lego cameras.
6. nerf/encoding.py (PE) + tests.
7. nerf/field.py (NeRF MLP) + tests.
8. nerf/volume.py + analytic tests + gradcheck.
9. nerf/sampler.py (stratified, ray-sphere bounds, inverse-CDF) + tests.
10. train/losses.py, eval/metrics.py (PSNR); train/nerf_trainer.py (coarse); overfit 1 image.
11. Train coarse NeRF on Lego 100 px; then 200 px; then add fine network; nerf/render.py spiral video.
12. core/ssim.py; eval/metrics.py (SSIM, LPIPS wrapper); eval/evaluate.py → report.md.
13. preprocess/extract_frames.py, select.py with tests on synthetic blur.
14. preprocess/run_sfm.py (pycolmap → undistort), data/colmap_io.py, preprocess/diagnose_sfm.py, preprocess/normalize.py. Validate on a rendered synthetic video with known poses (ATE).
15. scripts/prepare_video.py on a real phone capture → SceneData.
16. nerf/encoding.py hash grid + tests; HashNeRF; train on the object capture.
17. core/quaternion.py, gs/model.py (covariance) + tests.
18. core/sh.py + oracle tests.
19. gs/project.py + numerical-Jacobian and oracle tests.
20. gs/raster/naive.py + gs/render.py; 2D warm-up; single-view fit.
21. gs/raster/tiled.py (pairs → stable sort → padded tiles → compositing) + equality with naive + oracle + gradcheck; benchmark it/s, VRAM.
22. gs/init.py (filter, kNN scale, random fallback).
23. train/gs_trainer.py: param groups, LR decay, L1+SSIM, SH schedule, patch training; no densification; checkpoint.
24. gs/density.py: gradient accumulation (NDC scaling), clone/split/prune/reset, optimizer surgery, cap; unit tests.
25. Train on Lego 400 px → PSNR ~30 dB (heuristic). Debug against gsplat on the same input.
26. Train on the phone room; same data through gsplat/Splatfacto; compare with eval/benchmark.py.
27. export/ply.py, export/splat.py, export/video.py; open PLY in SuperSplat.
28. viewer/viser_app.py (own renderer, navigable).
29. Experiments E1–E12 in priority order: E4, E7, E9, E5, E1, E8, E10.
30. Optional: custom autograd.Function compositor → Triton kernel → gsplat backend switch → depth prior → appearance affine → VGGT front-end.
31. Optional: web viewer; extensions from Part 12.

---

## Part 15 — Architectural review

**Issues found and fixes**

| Issue | Category | Fix |
|---|---|---|
| Gemini: vanilla NeRF on phone video | Unrealistic | NeRF restricted to synthetic + bounded objects with hash grid |
| Pure-PyTorch rasterizer memory | Bottleneck | Tile cap K, chunking, checkpointing, patch training, backend abstraction |
| Per-tile depth sort popping and K cap | Correctness | Documented approximation, overflow metric, uncapped mode for tests |
| NDC vs pixel gradient | Missing math | Explicit scaling before threshold, oracle check |
| Optimizer state on densify | Hidden bug source | Dedicated surgery + tests |
| Convention drift (OpenGL/OpenCV, c2w/w2c) | Bug source | Single conversion point at loaders; one canonical form |
| No metric scale | Assumption | Sim3 stored, scale as extension |
| Oracle mismatch (dilation, near plane, SH direction) | Hidden | Oracle config that matches; sensitivity test |
| Viewer as major work | Complexity | viser + external viewers first |
| Pose bottleneck hidden | Missing | Diagnostics + same-data-into-oracle test |
| Circular deps | Check | Layer rule: L1 → L3/L4 → L5 → L6; L2 outputs SceneData only; no import nerf in gs and vice versa |
| Over-engineering | Simplify | One Rasterizer protocol, one RadianceField protocol; no plugin system; no distributed training |

**Where the risk is highest:** P9 (tiled rasterizer correctness/perf) and P11 (density control). Gate each with oracle equivalence, not visual appearance.

**Scalability:** ≤ ~1M Gaussians / ~1 MP per view in pure PyTorch; beyond that requires fused kernels (P14) or gsplat. Large-scale scenes are out of scope.

**Corrected final architecture (summary):**

- Shared L1 geometry core with a single convention (OpenCV, c2w) and a stored Sim3.
- L2 external SfM (pycolmap), custom frame selection + diagnostics, output SceneData.
- Two representation tracks: NeRF (vanilla → hierarchical → hash grid for objects) and 3DGS (main path for rooms).
- Rasterizer as a swappable backend: naive (tests) → tiled PyTorch (own, main) → gsplat (oracle/optional) → Triton (optional).
- Every component gated by analytic test + gradcheck + oracle comparison.
- Viewer: viser + third-party web viewers; custom viewer optional.
- Extensions attach through the generic-feature channel, exportable parameters, and stored normalization.
