# Conventions
- Camera: OpenCV convention (x right, y down, z forward). Poses are c2w, 4x4. w2c is derived, never stored.
- Blender/OpenGL and COLMAP (w2c) conventions are converted ONLY in data/blender_io.py and data/colmap_io.py.
- Pixel centers at +0.5. Quaternions are (w, x, y, z) and normalized before use.
- float32 for training, float64 for gradcheck. Document tensor shapes in docstrings.
- Do not change the Camera, SceneData, RadianceField or Rasterizer contracts (blueprint Part 6) silently.
