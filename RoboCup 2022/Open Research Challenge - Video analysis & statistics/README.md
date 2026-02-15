# Open Research Challenge - Video analysis & statistics

## Annotation format and labeling instructions
Please have a look into the provided PDF: [Open_Research_Challenge_-_Labeling_Instruction.pdf](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/RoboCup-SPL/Datasets/master/RoboCup%202022/Open%20Research%20Challenge%20-%20Video%20analysis%20%26%20statistics/Open_Research_Challenge_-_Labeling_Instruction.pdf)

## Contributions at the RoboCup 2022
https://spl.robocup.org/rc2022/#open-research-challenge

## Download script

The download script populates the subdirectories by loading images and GameController/TCM log files from the RoboCup cloud server according to the `index.yml` file.

### Requirements
- click
- PyYAML

### Execution

**Warning: This takes a lot of time (unfortunately without progress bar) and disk space (~13.2GB).**

The following command downloads all sub-datasets and extracts them to their respective subdirectories:
```bash
./download.py --extract
```

Instead of downloading the entire dataset, you can also download only specific teams' data:
```bash
./download.py --extract "Berlin United" RoboEireann
```

## Robot color classifier

### 1) Extract robot pictures

This script reads the labels CSVs and crops robot bounding boxes into per-team `robot_pics` folders.

```bash
python extract_robot_pics.py
```

To store all crops in a single shared folder, pass `--output-root`:

```bash
python extract_robot_pics.py --output-root robot_pics
```

If a dataset stores images under a nested folder (e.g., `SPQR/SPQR/images`), the script detects that layout automatically.

### 2) (Optional) Build a folder dataset

If you want a classic folder structure for training/validation, this script builds:
`color_dataset/train/<label>` and `color_dataset/val/<label>`.

```bash
python build_color_dataset.py --input-root . --output color_dataset --val-split 0.2
```

### 3) Train the CNN

The trainer scans `**/robot_pics/*.png`, parses the color from the filename (e.g.,
`SPQR_137504_Gray_2.png` -> `Gray`), and trains the CNN defined in `color_classifier.py`.

```bash
python train_color_classifier.py --input-root . --epochs 15 --batch-size 64
```

The best model (by validation accuracy) is saved to `color_classifier.pt` by default.

### End-to-end pipeline (single config)

Edit [pipeline.yml](pipeline.yml) to set your root, common `robot_pics` path, and training options.
Then run:

```bash
python run_pipeline.py --config pipeline.yml
```

The pipeline only downloads datasets that do not already contain images, then runs extraction,
dataset build, and training using the shared `robot_pics` path.
