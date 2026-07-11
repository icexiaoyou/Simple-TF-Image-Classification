"""Create a clean, resized classification dataset without modifying the source files."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("voc_dataset"))
    parser.add_argument("--output", type=Path, default=Path("pic_source"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def resize_and_pad(image: Image.Image, image_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), "white")
    canvas.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
    return canvas

def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.output}. Use --overwrite to replace files.")
    count = 0
    for class_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        destination = args.output / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS), 1):
            with Image.open(source_path) as image:
                resize_and_pad(image, args.image_size).save(destination / f"{class_dir.name}_{index:04d}.jpg", quality=95)
            count += 1
    if count == 0:
        raise SystemExit("No supported images were found in class subdirectories.")
    print(f"Wrote {count} images to {args.output}")

if __name__ == "__main__":
    main()
