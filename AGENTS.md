# AGENTS.md — nvs3d (NeRF / 3DGS from scratch)

Full spec: `docs/blueprint.md`. This file is the short rulebook. Read it fully every session.

## Goal
Phone video -> camera poses (external SfM) -> trained radiance field using OUR OWN PyTorch renderer -> novel views, PLY export, viser viewer. Core objective: understand and correctly implement differentiable rendering for NeRF and 3DGS, verified numerically.

## Non-goals
Writing an SfM system, dynamic scenes, relighting, metric accuracy, outdoor/large scenes, capture app, custom CUDA in the first pass.

## Workflow (strict)
1. Work on ONE phase only (P0..P15, blueprint Part 7). Never start the next phase.
2. Before coding: read the blueprint sections relevant to the phase (Part 7 entry, the matching Nx/Gx sections, Part 9). Then write a short plan and wait for approval if asked.
3. Write the phase's tests BEFORE the implementation.
4. Stop when the phase's "Done" criteria pass. Then report: commands run, test output summary, open doubts.
5. Do not modify code from earlier phases without telling me first, and say why.

## Conventions (never violate)
- Camera model: OpenCV convention (x right, y down, z forward).
- Poses are camera-to-world (c2w), 4x4. `w2c` is derived, never stored.
- Blender/OpenGL (x right, y up, -z forward) and COLMAP (w2c) conventions are converted ONLY inside the loaders (`data/blender_io.py`, `data/colmap_io.py`). No conversion anywhere else.
- Pixel centers at +0.5. Quaternions: (w, x, y, z), normalized before use.
- Shapes are documented in docstrings; dtype float32 in training, float64 in gradcheck.
- Key contracts: `Camera(K, c2w, W, H)`, `SceneData`, `RadianceField` protocol, `Rasterizer` protocol (Part 6 of the blueprint). Do not change them silently.

## Dependency rule
- L1 `core/` imports nothing from higher layers.
- L3/L4 (`nerf/`, `gs/`) depend on `core/` only.
- L5 `train/` talks to L3/L4 only through the `RadianceField` and `Rasterizer` protocols.
- L2 `data/` + `preprocess/` outputs `SceneData` and nothing else; trainers never read COLMAP files.
- Layers: L0 utils, L1 core, L2 data/preprocess, L3 models, L4 renderers, L5 train, L6 eval/export/viewer.

## "From scratch" rule
- Write ourselves: camera/ray math, transforms, quaternions, positional encoding, MLPs (nn.Linear allowed), hash grid, samplers, volume rendering, SH, covariance/projection, tile-sort + compositing rasterizer, SSIM/PSNR, densification/pruning, optimizer-state handling, trainers, PLY export.
- Allowed PyTorch primitives: autograd, nn.Linear, Adam, conv2d, searchsorted, sort.
- External allowed: COLMAP/pycolmap, image I/O, LPIPS, viser, pretrained models.
- gsplat and Inria rasterizer code: TEST ORACLE ONLY, in `tests/` or an explicit `oracle` backend. Never import or copy them into the main path. Never paste code from those repos.

## Definition of done (per component, in this order)
1. Analytic test (closed-form expected value).
2. float64 `torch.autograd.gradcheck` (disable discontinuities such as the alpha threshold).
3. Oracle test against gsplat / reference (dev-only dependency), where applicable.
"Looks right" is NOT a sign-off. Only numeric tests count. Visual outputs (frustum plots, spiral videos) are supplementary evidence.

## Test rules
- If a test fails, DO NOT loosen the tolerance, skip it, or xfail it. Explain why it fails and propose a fix.
- Never hard-code a tolerance to make something pass; derive it (fp64 ~1e-10 for gradcheck, ~1e-5 for forward comparisons, unless the blueprint says otherwise).
- Tests live in `tests/{unit,math,render,gradcheck,integration,regression}/`. Fixed seeds always.
- PSNR figures marked "heuristic" in the blueprint are guides, not guarantees. Report actual numbers honestly.

## (verify) items
The blueprint tags facts from memory with `(verify)`. Quote each one back to me instead of hard-coding it. Also flag any constant, hyperparameter, or formula you are recalling from memory (paper or repo) rather than deriving or reading. Put them in a `VERIFY.md` list with the file and line where they would be used.

## Code style
- Python 3.11, PyTorch 2.x, type hints, dataclasses + YAML configs (tyro or OmegaConf). Every run writes its resolved config.
- Small pure functions, no hidden global state, explicit devices and dtypes.
- Checkpoint format and run-dir layout: blueprint Part 6. Do not invent alternatives.

## Hardware
CUDA is needed for real training. If no GPU is present, do code and CPU tests only, say so, and do not fake training results.

## Do not
- Add dependencies not listed in blueprint Part 11 without asking.
- Refactor unrelated code, rename public APIs, or reformat untouched files.
- Claim something works without showing the command and output that proves it.
- Guess when the blueprint is ambiguous: ask.
- Never commit or push unless explicitly asked.
