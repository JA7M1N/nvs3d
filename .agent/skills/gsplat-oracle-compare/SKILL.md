---
name: gsplat-oracle-compare
description: Compare our 3DGS projection, SH, or rasterizer against gsplat as a test oracle, including gradients. Use for the oracle step of the definition of done.
---

# Steps
1. gsplat is a dev-only dependency. Import it only inside `tests/render/`, guarded with `pytest.importorskip("gsplat")` and a CUDA check. Never import it from `src/`.
2. Build one fixed-seed scene (random Gaussians in front of a known Camera) and convert our parameters to gsplat's input layout in a helper inside the test file. Check conventions explicitly: quaternion order, c2w vs w2c (viewmat), scale activation, SH layout and direction, pixel-center offset, dilation/blur.
3. Compare forward outputs: means2d, conics, radii, rgb, depth, alpha. Use the same dtype; start with atol ~1e-5 (fp64) or ~1e-4 (fp32) and report the actual max error.
4. Compare gradients of a scalar loss (e.g. sum of rgb * fixed random weights) with respect to means, scales, quats, opacities, SH.
5. On mismatch, diagnose conventions first. Never loosen tolerances or tune the scene to hide a gap.
6. Report: max abs/rel error per quantity, scene size, and any convention difference found.
