#! /usr/bin/env python3

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


@click.command()
@click.argument("datasets", type=str, nargs=-1)
def main(datasets):
    root_dir = pathlib.Path(__file__).resolve().parent
    with (root_dir / "index.yml").open(mode="r", encoding="utf-8", newline="\n") as f:
        dataindex = yaml.safe_load(f)

    if not all(ds_name in dataindex for ds_name in datasets):
        print(f"Unknown dataset(s): {', '.join(ds_name for ds_name in datasets if not ds_name in dataindex)}", file=sys.stderr)
        sys.exit(1)

    for ds_name in (datasets or dataindex.keys()):
        ds_info = dataindex[ds_name]
        ds_path = root_dir / ds_name
        output_path = ds_path / "robot_pics"
        output_path.mkdir(exist_ok=True)
        labels = pd.read_csv(root_dir / ds_info["labels"])  # labels entry already includes dataset dir, weirdly enough
        image_names = labels["filename"].unique()
        for img_name in tqdm.tqdm(image_names):
            img_path = ds_path / "images" / img_name
            image = cv2.imread(str(img_path))
            robot_labels = labels[(labels["filename"] == img_name) & (labels["label"] == 0)]
            for rl in robot_labels.itertuples():  # iterrows but better
                xmin = int(rl.x_min)
                xmax = int(rl.x_max)
                ymin = int(rl.y_min)
                ymax = int(rl.y_max)
                bot_pic = image[ymin:ymax, xmin:xmax]
                n = "_".join([ds_name, img_path.stem, COLOR_MAP[rl.color], str(rl.number)]) + ".png"
                cv2.imwrite(str(output_path / n), bot_pic)


if __name__ == "__main__":
    main()
