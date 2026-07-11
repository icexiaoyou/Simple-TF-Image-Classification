"""Run real-time OpenCV inference with a trained Keras classifier."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/classifier.keras"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/class_names.json"))
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    height, width = model.input_shape[1:3]
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open source: {args.source}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
            probabilities = model.predict(np.expand_dims(rgb, axis=0), verbose=0)[0]
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            label = labels[index] if confidence >= args.threshold else "uncertain"
            cv2.putText(frame, f"{label}: {confidence:.1%}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image classification", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
