# grasp_world

Utilities and playground scripts for working with mesh SDFs and visor-based visualizations.

## Environment Snapshot

The current development environment was inspected with:

- Python 3.10.0 (conda-forge build)
- PyTorch 2.8.0+cu128, TorchVision 0.23.0+cu128, Torchaudio 2.8.0+cu128
- Kaolin 0.18.0
- Viser 1.0.10
- Trimesh 4.8.2
- Matplotlib 3.10.6
- scikit-image 0.25.2


## Setup

1. Create and activate the environment:
   ```bash
   conda create -n grasp_world python=3.10
   conda activate grasp_world
   pip install --upgrade pip
   ```

2. Install the core packages (adjust the Torch build if you have a different CUDA version):
   ```bash
   pip install \
       torch==2.8.0+cu128 \
       torchvision==0.23.0+cu128 \
       torchaudio==2.8.0+cu128 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

3. Install Kaolin (choose the wheel that matches your Torch/CUDA pair, or build from source):
   ```bash
   pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu128.html
   ```
   If the prebuilt wheel fails, follow [Kaolin’s build-from-source guide](https://kaolin.readthedocs.io).

4. Install CPU/visualization dependencies used by the playground scripts:
   ```bash
   pip install trimesh "pyglet<2" viser scikit-image matplotlib
   ```

5. Install the project in editable mode so `grasp_world` is importable everywhere:
   ```bash
   pip install -e .
   ```

6. (Optional) Developer tooling:
   ```bash
   pip install ipython black ruff
   ```

## Usage

- **Viser LSB demo:**
  ```bash
  python playground/lsb_demo.py
  ```
  The script samples SDFs for the sphere and cube assets, blends them, and streams
  isosurfaces to a Viser server at <http://localhost:8080>. Press `Ctrl+C` in the terminal
  to stop the server.

- **SDF utilities:** `grasp_world/utils/sdf.py` exposes `mesh_sdf` and
  `mesh_sdf_from_points`. By default they auto-select CUDA when available and fall back to a
  CPU implementation (`trimesh + scikit-image`) otherwise.

## Assets

Primitive meshes live in `assets/primitive_shapes/`. Add new meshes there and point the
playground scripts to the desired `.obj` files.

## Troubleshooting

- **CUDA errors (error 304 / cudaGetDeviceCount):** Indicates CUDA isn’t accessible in the
  current environment. The SDF utilities will quietly fall back to the CPU path, but Kaolin
  may still emit warnings. Ensure GPU drivers match your Torch build if you need CUDA.
- **Kaolin import errors:** Reinstall Kaolin against the exact Torch/CUDA pair in use. The
  fallback path keeps the scripts functional without GPU SDF kernels, but mesh loading still
  depends on Kaolin’s Python modules.
