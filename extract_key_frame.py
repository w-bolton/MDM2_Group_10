"""
Extract key frames (first, middle, last) from two GIF files.

Default inputs: (from simulate/pipeline_forward.py)
- Outputs_simulate/sim.gif
- Outputs_pipeline/sim_pipeline.gif

Default outputs: (3 images per GIF from Outputs_simulate/pipeline file)
- Key_Outputs_image/Key_simulate/
- Key_Outputs_image/Key_pipeline/

- Hongze Lin
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image

# Function - Extract key frames from a GIF and save them as PNG images.
def extract_key_frames(gif_path: Path, output_dir: Path) -> None:
    """Extract frame 0, middle frame, and last frame from a GIF."""
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(gif_path) as gif:
        total_frames = getattr(gif, "n_frames", 1)
        first_idx = 0
        mid_idx = total_frames // 2
        last_idx = total_frames - 1

        frame_plan = [
            ("frame0", first_idx),
            ("frame_mid", mid_idx),
            ("frame_last", last_idx),
        ]

        for tag, frame_idx in frame_plan:
            gif.seek(frame_idx)
            frame = gif.convert("RGB")
            save_path = output_dir / f"{gif_path.stem}_{tag}.png"
            frame.save(save_path)
            print(f"Saved: {save_path}")

# Main Function
def main() -> None:
    base_dir = Path.cwd()
    target_root = base_dir / "Key_Outputs_image"

    jobs = [
        (base_dir / "Outputs_simulate" / "sim.gif", target_root / "Key_simulate"),
        (
            base_dir / "Outputs_pipeline" / "sim_pipeline.gif",
            target_root / "Key_pipeline",
        ),
    ]

    for gif_path, output_dir in jobs:
        extract_key_frames(gif_path, output_dir)

    print(f"\nDone. Key frames are in: {target_root}")


if __name__ == "__main__":
    main()
