import cv2
import numpy as np
import torch
import argparse
import os
from ultralytics import SAM, YOLOWorld

from ultralytics.models.sam import SAM3SemanticPredictor

class HDRProcessor:
    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize the SAM 3 predictor
        # Using the user-provided configuration pattern
        overrides = dict(conf=0.25, model="sam3.pt", task="segment", save=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

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
        
        # Create mask: Threshold at 50% luminance (128)
        # We invert so we select the shadows (where we want to blend the bright image in)
        _, mask_thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Gaussian Blur 201x201
        mask_blurred = cv2.GaussianBlur(mask_thresh, (201, 201), 0)
        
        return mask_blurred

    def get_window_mask_sam(self, image_path, text_prompt="window pane"):
        """
        Step 3: Window Merge
        - Segment windows using SAM 3 with text prompt.
        """
        # Set the target image for the predictor
        self.predictor.set_image(image_path)
        
        # Segment using text prompt (Concept Segmentation)
        results = self.predictor(text=text_prompt)
        
        # Load the image to get dimensions for mask creation
        # (We could also get it from predictor, but we need the array for processing anyway)
        # Using cv2.imread here again just to be safe on shape, or pass shape in.
        # But efficiently, we should rely on the results shape.
        
        full_mask = None
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    # Resize or process mask
                    m = mask.astype(np.uint8) * 255
                    
                    if full_mask is None:
                         # Initialize full_mask with proper shape
                         full_mask = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint8)
                    
                    if m.shape != full_mask.shape:
                        m = cv2.resize(m, (full_mask.shape[1], full_mask.shape[0]))
                    
                    full_mask = cv2.bitwise_or(full_mask, m)
        
        if full_mask is None:
             print("No windows detected.")
             # Need to return a blank mask of image size. 
             # We need to read image dimensions if no results.
             temp_img = cv2.imread(image_path)
             return np.zeros(temp_img.shape[:2], dtype=np.uint8)
             
        return full_mask

    def blend_images(self, base, overlay, mask):
        # Initial blend
        mask_norm = mask.astype(float) / 255.0
        # Expand dims for 3 channels
        if len(mask_norm.shape) == 2:
            mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
        elif mask_norm.shape[2] == 1:
            mask_norm = np.repeat(mask_norm, 3, axis=2)
        
        base_f = base.astype(float)
        overlay_f = overlay.astype(float)
        
        # Ensure shapes match
        if base_f.shape != overlay_f.shape:
             overlay_f = cv2.resize(overlay_f, (base_f.shape[1], base_f.shape[0]))
        if mask_norm.shape[:2] != base_f.shape[:2]:
             mask_norm = cv2.resize(mask_norm, (base_f.shape[1], base_f.shape[0]))
             if len(mask_norm.shape) == 2: # Resize might lose channel dim
                  mask_norm = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)
        
        blended = base_f * (1.0 - mask_norm) + overlay_f * mask_norm
        return blended.astype(np.uint8)

    def process(self, dark_path, medium_path, bright_path, output_path):
        print(f"Processing set: {dark_path}, {medium_path}, {bright_path}")
        
        dark = self.load_image(dark_path)
        medium = self.load_image(medium_path)
        bright = self.load_image(bright_path)
        
        # --- Step 2: Luminosity Masking & Blending ---
        lum_mask = self.create_luminosity_mask(bright)
        merged_base = self.blend_images(medium, bright, lum_mask)
        
        # --- Step 3 & 4: Window Merge ---
        # SAM 3 with text prompt on Dark exposure
        # Note: Passing PATH to the predictor as per user instruction
        window_mask = self.get_window_mask_sam(dark_path, text_prompt="window pane")
        
        # Dilate mask by 2 pixels (kernel 3x3, iter 2)
        kernel = np.ones((3,3), np.uint8) 
        window_mask_dilated = cv2.dilate(window_mask, kernel, iterations=2)
        
        # Feather the mask by 2 pixels (Gaussian Blur 5x5)
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
