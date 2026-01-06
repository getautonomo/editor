# Real Estate HDR Automation

Automated HDR processing pipeline designed for Real Estate photography.
Features:
- **Luminosity Masking**: Blends blended exposures to fill shadows naturally.
- **Window Pull**: Uses **SAM 3** (via `SAM3SemanticPredictor`) with native text prompting ("window pane") to isolate and restore window views from Dark exposures.
- **Headless & GPU-Optimized**: Built for RunPod / Cloud processing.

## Structure
- `processor.py`: Main logic.
- `models/`: Place `sam3.pt` here.
- `requirements.txt`: Dependencies.

## Setup on RunPod

1. **Select a Pod**: Use the standard **PyTorch 2.x** template.
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

Ensure you have Python 3.8+ and a CUDA-capable GPU.

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
   - Detects and segments "window pane" using **SAM 3** (Promptable Concept Segmentation).
   - Dilates and feathers the mask.
   - Composites the **Dark** exposure into the window areas.

## Models
The system uses:
- `sam3.pt` (SAM 3 Model)
Note: You must obtain `sam3.pt` appropriately.
All handled via `ultralytics`.
