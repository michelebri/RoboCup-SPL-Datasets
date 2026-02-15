#! /usr/bin/env python3

import pathlib
import requests
import shutil
import sys
import zipfile

import click
import yaml


@click.command()
@click.option("--extract", is_flag=True, help="Extract the zip archives.")
@click.argument("datasets", type=str, nargs=-1)
def main(extract, datasets):
    root_dir = pathlib.Path(__file__).resolve().parent
    with (root_dir / "index.yml").open(mode="r", encoding="utf-8", newline="\n") as f:
        data = yaml.safe_load(f)

    if not all(key in data for key in datasets):
        print(f"Unknown dataset(s): {', '.join(key for key in datasets if not key in data)}", file=sys.stderr)
        sys.exit(1)

    for key in (datasets or data.keys()):
        value = data[key]
        local_path = root_dir / key / f"{key}.zip"
        print(f"Downloading {key}...")
        """
        # No progress bar because the server doesn't tell us a Content-Length.
        with requests.get(f"{value['data']}/download", stream=True) as r:
            with local_path.open(mode="wb") as f:
                shutil.copyfileobj(r.raw, f)
                """
        if True and key != "B-Human":
            # This could totally be parallelized with downloading the next dataset.
            print(f"Extracting {key}...")
            with zipfile.ZipFile(local_path) as archive:
                archive.extractall(path=root_dir, members=[_ for _ in archive.namelist() if not _.endswith(".md")])
            local_path.unlink()


if __name__ == "__main__":
    main()
