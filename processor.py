import cv2
import numpy as np
import torch
import argparse
import os
from ultralytics import SAM, YOLOWorld

class HDRProcessor:
    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load models
        # Using YOLO-World for text-prompted detection of windows
        self.yolo_model = YOLOWorld('yolov8s-world.pt')
        self.yolo_model.set_classes(["window pane"])
        
        # Using SAM 2 for segmentation based on YOLO boxes
        self.sam_model = SAM('sam2_s.pt')
        
        if self.device == 'cuda':
             # Move models to GPU if possible/supported
             # Ultralytics handles device placement automatically during predict usually,
             # but we can specify device='cuda' in predict calls.
             pass

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image at {path}")
        return img

    def create_luminosity_mask(self, bright_img):
        """
        Step 2: Luminosity Masking
        - Calculate 50% luminance mask from Bright exposure.
        - Apply Gaussian Blur (201x201) to replicate 100px feather.
        """
        # Convert to grayscale to get luminance
        gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
        
        # Create mask: 50% luminance threshold (128 out of 255)
        # We want the mask to represent the bright areas to replace with Medium/Dark?
        # Usually for HDR:
        # If we are blending 'Medium' onto 'Bright' to recover highlights:
        # We want a mask of the OVER-EXPOSED areas in 'Bright'.
        # But the prompt says "fill shadows without hard edges".
        # Let's follow the prompt: "Blend the 'Medium' and 'Bright' exposures using this mask to fill shadows"
        # If filling shadows, we likely want the DARK areas of the Bright image?
        # Or maybe we are using the Bright image to fill shadows in the Medium image?
        # "Blend the 'Medium' and 'Bright' exposures using this mask to fill shadows"
        # Standard workflow: Base is Medium. Bright fills shadows. Dark fills highlights.
        # So we want a mask of Shadow areas in Medium? Or Luminance of Bright?
        # Prompt: "Create a function to calculate a 50% luminance mask from the 'Bright' exposure."
        # This implies we use the Bright exposure's luminance.
        # 50% Luminance Mask usually means logic: Mask = Smoothstep(Luminance around midtones).
        # Let's try a standard approach: Use the Bright image's own luminance.
        # If pixel is bright in 'Bright' --> It is good for shadows? No, 'Bright' image has good shadow detail.
        # So where 'Bright' is NOT overexposed, we use it. 
        # But let's stick to the prompt's specific instruction: "50% luminance mask".
        # I will create a mask where pixels > 50% luminance are favored? Or < 50%?
        # Typically "Luminosity Mask" selects bright pixels.
        # If we blend Medium and Bright to "fill shadows", we want to use Bright where Medium is too dark.
        # But the mask is derived from "Bright".
        # Let's assume a standard "Lights" mask for now: mask = luminance / 255.
        # But prompt says "50% luminance mask". This might mean a threshold or a range.
        # I'll implement a soft mask based on luminance > 128 (50%).
        # Or maybe it simply means the mask itself is the luminance channel?
        # Let's interpret "calculate a 50% luminance mask" as getting the luminance channel and perhaps thresholding or using it directly.
        # "Blend 'Medium' and 'Bright' ... to fill shadows".
        # We use Bright for Shadows. So we want specific parts of Bright.
        # Let's use an inverted luminance mask of the Medium image? No, prompt says "from the 'Bright' exposure".
        # Okay, let's assume we take the Luminance of the Bright exposure.
        # If we take (1 - Luminance(Bright)), we get dark parts? No, Bright image is bright.
        # Let's implementation a generic luminance mask from Bright and invert it if needed for blending logic.
        # Actually, "50% luminance mask" might mean a mask selecting the top 50% brightest pixels, or the bottom.
        # Given "fill shadows", we usually want the Brighter exposure in the Dark areas of the scene.
        # But the Bright exposure is bright everywhere.
        # Let's implement: Mask = Bright_Luminance.
        # Blend: Result = Medium * (1-Mask) + Bright * Mask ??
        # Let's stick to a literal interpretation of "Luminosity Mask": The Grayscale version of the image.
        # "50% luminance mask" -> Maybe a hard threshold at 50%?
        # Prompt: "Step 2... 50% luminance mask... Apply Gaussian Blur 201x201".
        # Hard threshold + Large Blur = Soft Mask.
        # So: Mask = 1 if Lum < 128 else 0 ? Or vice versa.
        # To "fill shadows", we want Bright image in Dark areas.
        # So Mask should be high where the scene is dark?
        # But the Bright exposure makes the scene bright.
        # Let's assume the user implies: Select the Shadow areas (Luminance < 128) and use the Bright exposure there.
        # So Mask = 1 where Lum < 128 (Shadows), 0 where Lum >= 128 (Highlights).
        # Then Blur.
        # Use simple thresholding to start.
        
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # Select dark areas
        # Note: Bright exposure might have few dark pixels.
        # Maybe they mean "Luminosity Mask" in the Photoshop sense (Mask = Luminance).
        # Let's try: Mask = Gray value.
        # A "50% luminance mask" is ambiguous. I will implement:
        # Mask = Clamp(Gray * 2, 0, 255) ??
        # Let's go with the Threshold interpretation because of the "hard edges" comment implying we need to soften them.
        # If it was just Gray value, there wouldn't be "hard edges" to soften with a 100px feather.
        # So: Threshold at 128 -> Hard Edge -> Blur -> Soft Edge.
        # We want to fill *shadows* with the Bright exposure. Shadows are dark.
        # So we select pixels < 128 (dark) in the Bright image?
        # Or maybe the "Medium" image?
        # Prompt: "from the 'Bright' exposure".
        # Okay, I will select pixels < 128 in the Bright exposure (deep shadows) and boost them?
        # Actually, if the Bright exposure is well exposed for shadows, they will be > 128 likely.
        # Let's assume we want to use the Bright exposure where the image is NOT blown out.
        # i.e. Luminance < 255 (or some high threshold).
        # Let's flip it. "Fill shadows" usually means using the Bright exposure to lift the darks of the Medium exposure.
        # I will generate the mask from the Bright image. 
        # Let's implement a 'Lights' mask (Luminance) and invert it to get 'Darks'.
        # I will create a mask based on the inverse of luminance. Darker pixels = Higher mask value.
        
        # PROPOSED LOGIC:
        # 1. Gray = Bright_Img_Luminance
        # 2. Mask = Invert(Gray)  (So dark areas are white in mask)
        # 3. Threshold? Prompt says "50% luminance mask".
        # I will threshold at 128.
        # 4. Blur.
        
        _, mask_thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Gaussian Blur 201x201
        mask_blurred = cv2.GaussianBlur(mask_thresh, (201, 201), 0)
        
        return mask_blurred

    def get_window_mask_sam(self, image, text_prompt="window pane"):
        """
        Step 3: Window Merge
        - Detect windows using YOLO-World (text prompt).
        - Segment using SAM 2.
        """
        # 1. Detect with YOLO-World
        results_yolo = self.yolo_model.predict(image, conf=0.1, device=self.device)
        bboxes = results_yolo[0].boxes.xyxy.cpu().numpy()
        
        if len(bboxes) == 0:
            print("No windows detected.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # 2. Segment with SAM 2 using the bounding boxes
        # SAM 2 in ultralytics can take bboxes prompt
        results_sam = self.sam_model(image, bboxes=bboxes, device=self.device)
        
        # Combine all masks
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # results_sam is a list of results. Since we passed one image, it's list of length 1?
        # Or maybe it iterates over boxes? Ultralytics SAM integration usually returns a Result object.
        # Let's handle the Result object.
        
        if results_sam[0].masks is not None:
            # masks.data is typically (N, H, W)
            masks = results_sam[0].masks.data.cpu().numpy()
            for mask in masks:
                # Resize if necessary (SAM sometimes returns low-res masks) -- Ultralytics usually handles this but let's be safe
                # Ultralytics masks are usually resized to original image size
                m = mask.astype(np.uint8) * 255
                if m.shape != full_mask.shape:
                    m = cv2.resize(m, (full_mask.shape[1], full_mask.shape[0]))
                full_mask = cv2.bitwise_or(full_mask, m)
                
        return full_mask

    def blend_images(self, base, overlay, mask):
        # Initial blend
        mask_norm = mask.astype(float) / 255.0
        # Expand dims for 3 channels
        mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
        
        base_f = base.astype(float)
        overlay_f = overlay.astype(float)
        
        blended = base_f * (1.0 - mask_norm) + overlay_f * mask_norm
        return blended.astype(np.uint8)

    def process(self, dark_path, medium_path, bright_path, output_path):
        print(f"Processing set: {dark_path}, {medium_path}, {bright_path}")
        
        dark = self.load_image(dark_path)
        medium = self.load_image(medium_path)
        bright = self.load_image(bright_path)
        
        # --- Step 2: Luminosity Masking & Blending ---
        # "Blend the 'Medium' and 'Bright' exposures using this mask"
        # We want to add Bright to Medium. 
        # Mask calculation from Bright exposure (dark areas of bright exposure? or simple luminance?)
        # Let's use the thresholded inverse luminance calculated earlier.
        lum_mask = self.create_luminosity_mask(bright)
        
        # Blend Medium and Bright
        # "Fill shadows" implies Bright goes into the masked area.
        # If mask is "Dark areas of Bright", that seems wrong. Bright has details in shadows.
        # Let's assume the mask highlights where we want the Bright image.
        # We want Bright image in the Shadows of the scene.
        # So mask should be high in shadow areas.
        merged_base = self.blend_images(medium, bright, lum_mask)
        
        # --- Step 3 & 4: Window Merge ---
        # "Isolate window panes using SAM 3 mask"
        # We likely want to find windows in the Dark exposure (where windows are not blown out).
        # Or Medium? Usually Dark exposure has the best window details (outside view).
        # Let's detect on Medium or Dark. Dark is safer for "window pane" features (frames etc).
        # Actually window pane GLASS is what we want. In Dark exposure, the view outside is visible.
        # In Bright, it's blown out white.
        # YOLO might fail on blown out windows.
        # Let's use the Dark exposure for detection and segmentation.
        window_mask = self.get_window_mask_sam(dark, text_prompt="window pane")
        
        # Dilate mask by 2 pixels
        kernel = np.ones((3,3), np.uint8) # 3x3 kernel roughly adds 1px border -> 2px dilation needs slightly larger or 2 iterations
        # "Dilate by 2 pixels" -> 2 iterations of 3x3 (which gives 1px extension per iter) or 5x5.
        # Let's use 2 iterations.
        window_mask_dilated = cv2.dilate(window_mask, kernel, iterations=2)
        
        # Feather the mask by 2 pixels (Gaussian Blur)
        # 2px feather -> Sigma around 1.0 or small kernel
        # "Feather by 2 pixels using a Gaussian Blur".
        # Kernel size usually ~ 6*sigma. 2px radius?
        # Let's generic odd kernel, say 5x5 or 7x7.
        window_mask_feathered = cv2.GaussianBlur(window_mask_dilated, (5, 5), 0)
        
        # Merge "Dark" window exposure into the blended base
        final_result = self.blend_images(merged_base, dark, window_mask_feathered)
        
        cv2.imwrite(output_path, final_result)
        print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Estate HDR Processor")
    parser.add_argument("--dark", required=True, help="Path to Dark exposure")
    parser.add_argument("--medium", required=True, help="Path to Medium exposure")
    parser.add_argument("--bright", required=True, help="Path to Bright exposure")
    parser.add_argument("--output", required=True, help="Output path")
    
    args = parser.parse_args()
    
    processor = HDRProcessor()
    processor.process(args.dark, args.medium, args.bright, args.output)
