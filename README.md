# Real Estate HDR Automation

Automated HDR processing pipeline designed for Real Estate photography.
Features:
- **Luminosity Masking**: Blends blended exposures to fill shadows naturally.
- **Window Pull (SAM 3)**: Uses **SAM 3** with native text prompting ("window pane") to isolate and restore window views from Dark exposures.
- **Headless & GPU-Optimized**: Built for RunPod / Cloud processing.

## Structure
- `processor.py`: Main logic.
- `models/`: PLACE `sam3_b.pt` (or similar SAM 3 weights) here. Note: SAM 3 model weights might need manual download or access request.
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
   - Detects and segments "window pane" using **SAM 3** with text prompts.
   - Dilates and feathers the mask.
   - Composites the **Dark** exposure into the window areas.

## Models
The system uses:
- `sam3_b.pt` (Promptable Concept Segmentation)
All handled via `ultralytics`.
