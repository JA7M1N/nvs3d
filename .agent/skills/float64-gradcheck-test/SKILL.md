---
name: float64-gradcheck-test
description: Write a float64 torch.autograd.gradcheck test for a differentiable function (covariance, projection, SH, compositing, volume rendering). Use whenever a component needs its gradcheck step.
---

# Steps
1. Put the test in `tests/gradcheck/test_<component>.py`.
2. Build tiny inputs: `torch.randn(..., dtype=torch.float64, requires_grad=True)` with a fixed seed. Keep sizes small (e.g. 4-8 Gaussians, 8x8 pixels) so gradcheck is fast.
3. Wrap the component in a function taking only tensors that need gradients; close over everything else.
4. Disable discontinuities: alpha threshold, radius culling, clamps, top-K caps. Add a test flag or config option for this; do not change production defaults.
5. Keep inputs away from kinks (e.g. opacities in 0.2-0.8, Gaussians well in front of the camera).
6. Call `torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)`. Do not loosen these to pass; if it fails, report which input fails and why.
7. Also assert outputs are finite and gradients are not all zero.
