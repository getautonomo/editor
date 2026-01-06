# Real Estate HDR Automation

Automated HDR processing pipeline designed for Real Estate photography.
Features:
- **Luminosity Masking**: Blends blended exposures to fill shadows naturally.
- **Window Pull**: Uses **YOLO-World** (text prompt "window pane") and **SAM 2** to isolate and restore window views from Dark exposures.
- **Headless & GPU-Optimized**: Built for RunPod / Cloud processing.

## Structure
- `processor.py`: Main logic.
- `models/`: Place `sam2_s.pt` and `yolov8s-world.pt` here (will be auto-downloaded by Ultralytics if missing, or you can pre-load).
- `requirements.txt`: Dependencies.

## Setup on RunPod

1. **Select a Pod**: Use the standard **PyTorch 2.x** template (e.g., `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel`).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Processing**:
   ```bash
   python processor.py --dark /path/to/dark.jpg \
                       --medium /path/to/medium.jpg \
                       --bright /path/to/bright.jpg \
                       --output /path/to/result.jpg
   ```

## Local Installation

Ensure you have Python 3.8+ and a CUDA-capable GPU (optional but recommended).

```bash
pip install -r requirements.txt
python processor.py --help
```

## How It Works

1. **Luminosity Merge**:
   - Calculates a 50% luminance mask from the **Bright** exposure.
   - Applies a massive feather (100px) to smooth transitions.
   - Blends Medium and Bright exposures.

2. **Window Restoration**:
   - Detects "window pane" using **YOLO-World**.
   - Segments exact glass area using **SAM 2**.
   - Dilates and feathers the mask.
   - Composites the **Dark** exposure into the window areas.

## Models
The system uses:
- `yolov8s-world.pt` (Automated window detection via text prompt)
- `sam2_s.pt` (Precise segmentation)
All handled via `ultralytics`.
