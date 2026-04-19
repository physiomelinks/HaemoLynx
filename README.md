# ImageLynx

Converts raw microscopy images of the microvasculature into computational haemodynamics models for hypothesis testing, experimental design, and more.

# Testing

to run tests do 

`pytest -s`

from the repo root directory

# Allowable input mask formats

tif

h5 

# GPU Acceleration (Optional)

ImageLynx supports hardware acceleration for 3D image preprocessing and skeletonization using NVIDIA GPUs. 

To enable this, install the optional dependencies:
```bash
pip install -r requirements-gpu.txt
```
The pipeline will automatically detect your GPU and offload the heaviest matrix operations. If no compatible GPU is found, it will gracefully fall back to CPU-bound processing.



