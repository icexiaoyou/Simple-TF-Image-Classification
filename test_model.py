"""Run image-classification inference for one or more images."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/classifier.keras"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/class_names.json"))
    parser.add_argument("images", type=Path, nargs="+", help="Images to classify")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    height, width = model.input_shape[1:3]
    for path in args.images:
        with Image.open(path) as image:
            batch = np.expand_dims(tf.keras.utils.img_to_array(image.convert("RGB").resize((width, height))), axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        print(f"{path}: {labels[index]} ({probabilities[index]:.2%})")

if __name__ == "__main__":
    main()
