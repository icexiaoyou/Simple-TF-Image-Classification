# Simple TensorFlow Image Classification

A current TensorFlow/Keras image-classification workflow for a directory-per-class dataset. It uses transfer learning, saves the portable `.keras` format, and keeps preprocessing non-destructive.

## Setup

TensorFlow is installed with pip inside Conda because the official TensorFlow wheels are the supported distribution on Windows. The default workflow is CPU-friendly.

```powershell
conda env create -f environment.yml
conda activate tf-image-classification
```

## Dataset layout

Put images into one directory per class. Annotations are not required for classification.

```text
voc_dataset/
  black/
    image_001.jpg
  white/
    image_001.jpg
```

Create a clean 224 x 224 copy without overwriting your originals:

```powershell
python preprocess.py --source voc_dataset --output pic_source --image-size 224
```

## Train, test, and run

```powershell
python train.py --data pic_source --output artifacts --epochs 20
python test_model.py --model artifacts/classifier.keras --labels artifacts/class_names.json test.jpg
python deploy.py --model artifacts/classifier.keras --labels artifacts/class_names.json --source 0
```

Use `--weights none` when the machine cannot download ImageNet weights. All commands support `--help`.
