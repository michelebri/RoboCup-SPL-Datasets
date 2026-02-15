#! /usr/bin/env python3

import csv
import pathlib
import sys

import click
import yaml
import pandas as pd
import cv2
import tqdm

COLOR_MAP = {
    -1: "Unknown",
    0: "Blue",
    1: "Red",
    2: "Yellow",
    3: "Black",
    4: "White",
    5: "Green",
    6: "Orange",
    7: "Purple",
    8: "Brown",
    9: "Gray"
}

OUTPUT_SIZE = 128


@click.command()
@click.option("--output-root", type=str, default=None, help="Common output folder for all robot pics.")
@click.argument("datasets", type=str, nargs=-1)
def main(output_root, datasets):
    root_dir = pathlib.Path(__file__).resolve().parent
    with (root_dir / "index.yml").open(mode="r", encoding="utf-8", newline="\n") as f:
        dataindex = yaml.safe_load(f)

    if not all(ds_name in dataindex for ds_name in datasets):
        print(f"Unknown dataset(s): {', '.join(ds_name for ds_name in datasets if not ds_name in dataindex)}", file=sys.stderr)
        sys.exit(1)

    if output_root:
        output_root_path = pathlib.Path(output_root)
        if not output_root_path.is_absolute():
            output_root_path = root_dir / output_root_path
        output_root_path.mkdir(parents=True, exist_ok=True)
        output_images_path = output_root_path / "images"
        output_images_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_root_path / "robot_pics.csv"
    else:
        output_root_path = None
        output_images_path = None
        csv_path = root_dir / "robot_pics.csv"

    try:
        with csv_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image", "color"])
            writer.writeheader()
            for ds_name in (datasets or dataindex.keys()):
                ds_info = dataindex[ds_name]
                ds_path = root_dir / ds_name
                if output_root:
                    output_path = output_images_path
                else:
                    output_path = ds_path / "robot_pics"
                output_path.mkdir(parents=True, exist_ok=True)
                labels = pd.read_csv(root_dir / ds_info["labels"])  # labels entry already includes dataset dir, weirdly enough
                image_names = labels["filename"].unique()
                images_root = ds_path / "images"
                if not images_root.is_dir():
                    nested_images_root = ds_path / ds_name / "images"
                    if nested_images_root.is_dir():
                        images_root = nested_images_root
                for img_name in tqdm.tqdm(image_names):
                    img_path = images_root / img_name
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Warning: could not read image: {img_path}", file=sys.stderr)
                        continue
                    robot_labels = labels[(labels["filename"] == img_name) & (labels["label"] == 0)]
                    for rl in robot_labels.itertuples():  # iterrows but better
                        xmin = int(rl.x_min)
                        xmax = int(rl.x_max)
                        ymin = int(rl.y_min)
                        ymax = int(rl.y_max)
                        bot_pic = image[ymin:ymax, xmin:xmax]
                        if bot_pic.size == 0:
                            continue
                        height, width = bot_pic.shape[:2]
                        size = max(height, width)
                        top = (size - height) // 2
                        bottom = size - height - top
                        left = (size - width) // 2
                        right = size - width - left
                        bot_pic = cv2.copyMakeBorder(
                            bot_pic,
                            top,
                            bottom,
                            left,
                            right,
                            borderType=cv2.BORDER_CONSTANT,
                            value=(0, 0, 0),
                        )
                        bot_pic = cv2.resize(bot_pic, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
                        n = "_".join([ds_name, img_path.stem, COLOR_MAP[rl.color], str(rl.number)]) + ".png"
                        cv2.imwrite(str(output_path / n), bot_pic)
                        writer.writerow({"image": n, "color": COLOR_MAP[rl.color]})
    except Exception as e:
        print(f"Error processing datasets: {e}", file=sys.stderr)
        pass
    


if __name__ == "__main__":
    main()
