# Architecture and "from scratch" rule
- L1 core/ imports nothing from higher layers. L3/L4 (nerf/, gs/) depend on core/ only.
- train/ talks to models and renderers only via the RadianceField and Rasterizer protocols.
- data/ and preprocess/ output SceneData only; trainers never read COLMAP files.
- Write ourselves: camera/ray math, PE, hash grid, samplers, volume rendering, SH, covariance/projection, tiled rasterizer, SSIM/PSNR, densification, optimizer surgery, PLY export.
- Allowed primitives: autograd, nn.Linear, Adam, conv2d, searchsorted, sort.
- gsplat and Inria code: test oracle only (tests/ or an explicit oracle backend). Never copy into the main path.
