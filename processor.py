import cv2
import numpy as np
import torch
import argparse
import os
from ultralytics.models.sam import SAM3SemanticPredictor

class HDRProcessor:
    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        overrides = dict(
            conf=0.25, 
            task="segment", 
            mode="predict", 
            model="sam3.pt", 
            imgsz=1022, 
            device=self.device
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def get_window_mask_sam(self, image_path, text_prompt="window pane"):
        """Isolate glass using SAM 3; targets panes specifically to avoid frames."""
        self.predictor.set_image(image_path)
        results = self.predictor(text=[text_prompt])

        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        h, w = orig_img.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    m = (mask > 0).astype(np.uint8) * 255
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    full_mask = cv2.bitwise_or(full_mask, m)
        return full_mask

    def create_luminosity_mask(self, bright_img):
        """Refined shadow recovery: Simplified for 'Zone' blending."""
        gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
        
        # NEW STEP: Pre-processing to kill high-frequency detail (texture)
        # This ensures the mask focuses on 'Areas' of light, not 'Pixels' of light.
        gray_zones = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # Normalize the 'Zone' map
        mask_continuous = gray_zones.astype(np.float32) / 255.0
        
        # 100px Feathering for the final falloff
        feather_radius = 100
        kernel_size = (feather_radius * 2) + 1
        blurred_mask = cv2.GaussianBlur(mask_continuous, (kernel_size, kernel_size), 0)
        
        mask_visual = (blurred_mask * 255).astype(np.uint8)
        return blurred_mask, mask_visual

    def blend_images(self, base, overlay, mask):
        """
        Applies Alpha Blending[cite: 34, 115].
        Formula: (Foreground * Mask) + (Background * (1-Mask))
        """
        # Ensure mask is 0.0-1.0 float range for math
        if mask.dtype != np.float32 and mask.dtype != np.float64:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask

        if len(mask_norm.shape) == 2:
            mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        
        base_f = base.astype(np.float32)
        overlay_f = overlay.astype(np.float32)
        
        # Perform the actual blend [cite: 118]
        blended = (overlay_f * mask_norm) + (base_f * (1.0 - mask_norm))
        return np.clip(blended, 0, 255).astype(np.uint8)

    def process(self, dark_path, medium_path, bright_path, output_path):
        print("Loading exposures...")
        dark = cv2.imread(dark_path)
        medium = cv2.imread(medium_path)
        bright = cv2.imread(bright_path)
        
        if dark is None or medium is None or bright is None:
            return

        # Step 1: Shadow Fill (Bright + Medium) [cite: 27, 28]
        print("Creating luminosity mask...")
        lum_mask_float, lum_mask_visual = self.create_luminosity_mask(bright)
        merged_base = self.blend_images(bright, medium, lum_mask_float)
        
        # Debug saves
        cv2.imwrite('luminosity_mask_visual.png', lum_mask_visual)
        cv2.imwrite('step1_shadow_fill.jpg', merged_base)
        
        # Step 2: SAM 3 Window Pane Detection
        print(f"Running SAM 3 segmentation for windows...")
        window_mask_binary = self.get_window_mask_sam(dark_path, text_prompt="window pane")
        cv2.imwrite('window_mask_binary.png', window_mask_binary)

        # Step 3: Refine Window Mask (Expand & Feather) [cite: 89, 90, 91]
        # Dilate 2px to ensure overlap with frames [cite: 90, 166]
        kernel = np.ones((5, 5), np.uint8) 
        dilated_mask = cv2.dilate(window_mask_binary, kernel, iterations=1)
        # Soften edges (feather) by 2px [cite: 91, 168]
        feathered_window_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (11, 11), 0)
        
        # Step 4: Final Merge [cite: 31, 32, 170]
        # Overlay dark window view onto the blended interior [cite: 170]
        final_result = self.blend_images(merged_base, dark, feathered_window_mask)
        
        cv2.imwrite(output_path, final_result)
        print(f"Process complete. Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dark", required=True, help="Path to dark/window exposure")
    parser.add_argument("--medium", required=True, help="Path to medium/neutral exposure")
    parser.add_argument("--bright", required=True, help="Path to bright/shadow exposure")
    parser.add_argument("--output", default="final_hdr.jpg", help="Output filename")
    
    args = parser.parse_args()
    processor = HDRProcessor()
    processor.process(args.dark, args.medium, args.bright, args.output)