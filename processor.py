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

    def get_semantic_masks(self, image_path, prompts):
        """
        Generates semantic masks for specified prompts.
        prompts: Dictionary mapping 'name' -> 'text_prompt'
        """
        self.predictor.set_image(image_path)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        masks = {}
        
        for name, prompt in prompts.items():
            print(f"Segmenting {name}...")
            # Detect everything first, then filter/combine
            results = self.predictor(text=[prompt])
            
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for result in results:
               if result.masks is not None:
                   result_masks = result.masks.data.cpu().numpy()
                   for m in result_masks:
                       m_resized = (m > 0).astype(np.uint8) * 255
                       if m_resized.shape[:2] != (h, w):
                           m_resized = cv2.resize(m_resized, (w, h), interpolation=cv2.INTER_NEAREST)
                       combined_mask = cv2.bitwise_or(combined_mask, m_resized)
            
            masks[name] = combined_mask
            
        return masks

    def apply_color_corrections(self, img, masks):
        """
        Generalizes color cast correction based on semantic segments.
        masks: A dictionary containing SAM 3 masks for 'ceiling', 'floor', 'wood', 'tiles'.
        """
        # Convert to HLS for easier saturation manipulation
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        h, l, s = cv2.split(hls)

        # 1. CEILING CLEANING (Targeting all casts)
        if 'ceiling' in masks:
            # [cite] Apply 5px feather to ceiling mask
            ceiling_mask = masks['ceiling'].astype(np.float32) / 255.0
            feathered_ceiling = cv2.GaussianBlur(ceiling_mask, (5, 5), 0)
            
            # [cite] Set Saturation to 0 (Perfect Neutral)
            # We use the feathered mask to blend between original S and 0
            # S_new = S_old * (1 - mask) + 0 * mask  =>  S_old * (1 - mask)
            s = s * (1.0 - feathered_ceiling) 

        # 2. FLOOR CLEANING (Targeting Blue/Cyan Sky Reflections)
        if 'floor' in masks:
            # Target Blue range (Hue ~90-130 in OpenCV HLS)
            blue_mask = cv2.inRange(h, 90, 130) 
            floor_blue = cv2.bitwise_and(blue_mask, masks['floor'])
            s[floor_blue > 0] *= 0.2  # Leave 20% for natural feel

        # 3. CABINET/WALL CLEANING (Targeting Orange/Yellow Artificial Light)
        # Target Yellows (Hue ~10-45)
        orange_yellow_mask = cv2.inRange(h, 10, 45) 
        
        # 4. PROTECTION LOGIC (The "Selective" Step)
        # Ensure mask is uint8 for bitwise operations
        protection_mask = np.zeros(s.shape, dtype=np.uint8)
        if 'wood' in masks: protection_mask = cv2.bitwise_or(protection_mask, masks['wood'])
        if 'tiles' in masks: protection_mask = cv2.bitwise_or(protection_mask, masks['tiles'])
        
        # Also protect the floor from general orange reduction if not requested, but usually floor is its own thing
        # The prompt implies general correction unless protected.
        
        # Apply correction only where NOT protected
        correction_area = cv2.bitwise_and(orange_yellow_mask, cv2.bitwise_not(protection_mask))
        s[correction_area > 0] *= 0.3 # Reduce orange cast by 70%

        # Merge back and convert to BGR
        corrected_img = cv2.merge([h, l, s])
        return cv2.cvtColor(corrected_img.astype(np.uint8), cv2.COLOR_HLS2BGR)

    def create_tone_curve(self, contrast=0, highlights=0, shadows=0, whites=0, blacks=0):
        """
        Generates a Lookup Table (LUT) simulating Lightroom tone sliders.
        Inputs are normalized roughly to -100 to 100 range (or similar scale).
        """
        x = np.arange(256).astype(np.float32) / 255.0
        
        # 1. Exposure/Tone Mapping (Shadows/Highlights)
        # Simple polynomial or gamma conceptualization
        # Shadows: Boost low end. Highlights: Compress high end.
        
        # Shadows (+60 means brighten shadows). 
        # We can use a simple quadratic or cubic offset for the lower half.
        # But a robust way is to use splines or composite functions.
        # Let's use a simplified approach:
        
        # Highlights (- : recover/darken, + : blow out/brighten)
        # Shadows (+ : recover/brighten, - : crush/darken)
        
        # Apply Shadows/Highlights using a smooth weighting function
        # Mask for shadows (1 at 0, 0 at 1)
        shadow_mask = 1.0 - x
        # Mask for highlights (0 at 0, 1 at 1)
        highlight_mask = x
        
        # Strength factors (tuned empirically)
        s_factor = shadows * 0.002 # +60 -> +0.12
        h_factor = highlights * 0.002 # -75 -> -0.15
        
        # Apply boosts weighted by masks (restricted to relevant ranges)
        # Gaussian weight prefered to linear to limit effect to deep shadows/high lights
        shadow_weight = np.exp(-((x - 0.0)**2) / 0.2)
        highlight_weight = np.exp(-((x - 1.0)**2) / 0.2)
        
        y = x + (s_factor * shadow_weight) + (h_factor * highlight_weight)
        y = np.clip(y, 0.0, 1.0)
        
        # 2. Whites and Blacks (Levels/Clipping equivalent)
        # Blacks: Shift black point. Whites: Shift white point.
        # +25 Whites means input 0.75 maps to 1.0 (roughly)? Or just boosting brights?
        # Typically Whites increases the "white point" clipping.
        # Blacks decreases the "black point".
        
        # Simplified "Levels" stretch
        # blacks -45 -> slightly crush blacks (move black point right?)
        # Actually in LR:
        # Blacks < 0 : darken blacks. Blacks > 0 : lift blacks.
        # Whites > 0 : brighten whites. Whites < 0 : darken whites.
        
        # Let's map this to a curve adjustment similar to S-curve but focused on ends.
        b_shift = blacks * 0.001 # -45 -> -0.045
        w_shift = whites * 0.001 # +25 -> +0.025
        
        # We can just apply this as an offset weighted to ends again, or a power curve.
        # Let's use simple power stretch for whites/blacks effect
        # (This is an approximation)
        
        # 3. Contrast (S-Curve)
        # +10 Contrast
        if contrast != 0:
            c = contrast * 0.01 # +0.1
            # S-curve formula: https://stackoverflow.com/questions/13840061
            # k = tan((45 + c/2) * rad) ...
            # Simpler:
            factor = (1.01 + c) / (1.01 - c) if c < 1.0 else 1.0
            y = 0.5 + factor * (y - 0.5)
        
        # Re-apply Blacks/Whites as linear stretch on the result
        # This acts closer to setting the "white point" and "black point"
        # Blacks -45 means we want input value X to become 0. Meaning black point moves up? 
        # No, Blacks -45 in LR crushes blacks. So output gets darker. 
        # So we can just subtract/add.
        y = y + b_shift * (1-x) + w_shift * x
        
        return np.clip(y * 255, 0, 255).astype(np.uint8)

    def final_bump(self, img):
        """
        Final Polishing Stage matching user specs:
        Tone: Contrast +10, High -75, Shad +60, Whites +25, Blacks -45
        Presence: Clarity +10, Texture 0
        """
        print("Starting Final Bump (Lightroom Style)...")
        current_img = img.copy()

        # 1. Apply Tone Curve (Highlights, Shadows, Whites, Blacks, Contrast)
        try:
            # Generate LUT
            lut = self.create_tone_curve(contrast=10, highlights=-75, shadows=60, whites=25, blacks=-45)
            
            # Apply to Lightness channel only (to preserve color)
            # Or Value in HSV. LAB is best for perceptual lightness.
            lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            l_toned = cv2.LUT(l, lut)
            
            # Merge back
            current_img = cv2.cvtColor(cv2.merge([l_toned, a, b]), cv2.COLOR_LAB2BGR)
            cv2.imwrite('debug_step7_1_tone_curve.jpg', current_img)
            print("Finished Tone Curve adjustment.")
        except Exception as e:
            print(f"Skipping Tone Curve due to error: {e}")

        # 2. Clarity (Mid-tone Contrast)
        # LR Clarity +10 is subtle.
        try:
            # Radius: Large (e.g., 20% of image dimension is robust, but fixed 20px is decent for 1022px)
            # Amount: +10 is small. 
            # Previous was 1.5 (huge). +10 might correspond to alpha=0.1 or 0.2
            alpha = 0.2 
            
            gaussian = cv2.GaussianBlur(current_img, (0, 0), 25)
            # Unsharp Mask: Img + alpha * (Img - Blur)
            # = (1+alpha)*Img - alpha*Blur
            clarity_img = cv2.addWeighted(current_img, 1.0 + alpha, gaussian, -alpha, 0)
            
            current_img = clarity_img
            cv2.imwrite('debug_step7_2_clarity.jpg', current_img)
            print("Finished Clarity (+10).")
        except Exception as e:
            print(f"Skipping Clarity due to error: {e}")

        # 3. Texture (Explicitly 0)
        # DISABLED
        
        return current_img

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
        
        # Step 4: Blend Window
        print("Blending window exposure...")
        # Overlay dark window view onto the blended interior [cite: 170]
        final_with_window = self.blend_images(merged_base, dark, feathered_window_mask)
        cv2.imwrite('step4_window_blend.jpg', final_with_window)
        
        # Step 5: Semantic Feature Detection
        print("Generating semantic masks for color correction...")
        # Dictionary of things to find
        prompts = {
            'ceiling': 'ceiling',
            'floor': 'floor',
            'wood': 'wood cabinets, wood floor, wood furniture', # targeted wood prompt
            'tiles': 'tiled floor, tiled wall'
        }
        # Using medium exposure for semantic segmentation as it's the most balanced
        semantic_masks = self.get_semantic_masks(medium_path, prompts)
        
        # Debug: Save semantic masks
        for name, mask in semantic_masks.items():
            cv2.imwrite(f'debug_mask_{name}.png', mask)

        # Step 6: Smart Color Correction
        print("Applying semantic color corrections...")
        corrected_img = self.apply_color_corrections(final_with_window, semantic_masks)
        cv2.imwrite('step6_color_corrected.jpg', corrected_img)
        
        # Step 7: Final Polish
        print("Applying final clarity and contrast...")
        final_result = self.final_bump(corrected_img)
        
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