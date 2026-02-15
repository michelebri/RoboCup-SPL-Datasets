import csv
import pathlib
import random
from collections import defaultdict

def main():
    input_root = pathlib.Path("robot_pics")
    output = pathlib.Path("color_dataset")

    # Do not modify if you want to reproduce the same dataset split
    val_split = 0.2
    test_split = 0.1
    seed = 42

    items = []
    csv_path =  input_root / "robot_pics.csv"
    with csv_path.open(mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = (row.get("image") or "").strip()
            label = (row.get("color") or "").strip()
            if not image_name or not label:
                continue
            img_path = input_root / "images" / image_name
            try:
                rel_path = img_path.relative_to(input_root.parent)
                items.append((str(rel_path), label))
            except ValueError:
                items.append((str(img_path), label))

    if not items:
        raise SystemExit("No labeled images found. Check *robot_pics.csv files.")

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for image_path, label in items:
        by_label[label].append((image_path, label))

    train_items = []
    val_items = []
    test_items = []
    for label, label_items in by_label.items():
        rng.shuffle(label_items)
        n_test = int(len(label_items) * test_split)
        n_val = int(len(label_items) * val_split)
        test_items.extend(label_items[:n_test])
        val_items.extend(label_items[n_test : n_test + n_val])
        train_items.extend(label_items[n_test + n_val :])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    output.mkdir(parents=True, exist_ok=True)
    for filename, split_items in (
        ("train.csv", train_items),
        ("valid.csv", val_items),
        ("test.csv", test_items),
    ):
        out_path = output / filename
        with out_path.open(mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "color"])
            writer.writeheader()
            for image_path, label in split_items:
                writer.writerow({"image": image_path, "color": label})

    print(f"Total items: {len(items)}")
    print(f"Train: {len(train_items)} | Valid: {len(val_items)} | Test: {len(test_items)}")


if __name__ == "__main__":
    main()
