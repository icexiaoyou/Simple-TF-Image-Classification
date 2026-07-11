"""Train an image classifier with a current TensorFlow/Keras workflow."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tensorflow as tf

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("pic_source"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()

def build_model(image_size: int, class_count: int, weights: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=None if weights == "none" else weights, input_tensor=augment(inputs)
    )
    backbone.trainable = False
    features = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="predictions")(
        tf.keras.layers.Dropout(0.25)(features)
    )
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data}")
    if not 0 < args.validation_split < 1:
        raise SystemExit("--validation-split must be between 0 and 1.")
    tf.keras.utils.set_random_seed(args.seed)
    options = dict(directory=args.data, validation_split=args.validation_split, seed=args.seed,
                   image_size=(args.image_size, args.image_size), batch_size=args.batch_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **options)
    validation_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **options)
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("At least two class directories are required.")
    train_ds = train_ds.shuffle(1000, seed=args.seed).prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    model_path = args.output / "classifier.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.2),
    ]
    model = build_model(args.image_size, len(class_names), args.weights)
    model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs, callbacks=callbacks)
    loss, accuracy = model.evaluate(validation_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    print(f"Best model: {model_path}")

if __name__ == "__main__":
    main()
