from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --------------------------------------------------
# Configuration
# --------------------------------------------------
IMAGE_PATH = Path("data/sample_images/truck.jpg")
OUTPUT_DIR = Path("results/partitions/sam3")

PROMPTS = [
    "pickup truck",
    "wheel",
    "window",
    "road",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"
assert torch.cuda.is_available(), "CUDA is not available"

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# --------------------------------------------------
# Load image and SAM3
# --------------------------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.asarray(image)

print("Image size:", image.size)
print("Loading SAM3 pretrained checkpoint...")

model = build_sam3_image_model()
model.eval()
processor = Sam3Processor(model)

print("SAM3 model loaded successfully.")

# --------------------------------------------------
# Text-conditioned inference
# --------------------------------------------------
with torch.inference_mode():
    state = processor.set_image(image)

    for prompt in PROMPTS:
        slug = prompt.replace(" ", "_")

        print(f"\nPrompt: {prompt}")

        output = processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

        masks = output["masks"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        boxes = output.get("boxes")
        if boxes is not None:
            boxes = boxes.detach().cpu().numpy()
        else:
            boxes = np.empty((0, 4), dtype=np.float32)

        # SAM3 may return [N, 1, H, W].
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        if masks.ndim != 3:
            raise ValueError(
                f"Unexpected mask dimensions: {masks.shape}"
            )

        masks = masks > 0.5

        print("Instances:", len(scores))
        print("Masks:", masks.shape)
        print("Scores:", scores)

        np.savez_compressed(
            OUTPUT_DIR / f"{slug}.npz",
            masks=masks.astype(np.uint8),
            scores=scores,
            boxes=boxes,
        )

        # Create an overlay visualization.
        overlay = image_np.astype(np.float32).copy()
        rng = np.random.default_rng(42)

        for mask in masks:
            if mask.shape != image_np.shape[:2]:
                raise ValueError(
                    f"Mask/image mismatch: {mask.shape} "
                    f"vs {image_np.shape[:2]}"
                )

            color = rng.integers(40, 256, size=3)

            overlay[mask] = (
                0.55 * overlay[mask]
                + 0.45 * color
            )

        Image.fromarray(
            np.clip(overlay, 0, 255).astype(np.uint8)
        ).save(OUTPUT_DIR / f"{slug}_overlay.png")

        print("Saved:", OUTPUT_DIR / f"{slug}.npz")
        print("Overlay:", OUTPUT_DIR / f"{slug}_overlay.png")

print("\nAll SAM3 tests completed.")