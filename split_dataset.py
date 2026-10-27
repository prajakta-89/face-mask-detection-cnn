
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path("dataset")
OUTPUT = Path("data")

classes = ["with_mask", "without_mask", "incorrect_mask"]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_name in classes:
    source_folder = SOURCE / class_name

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Missing folder: {source_folder}"
        )

    files = [
        p for p in source_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if len(files) < 3:
        raise ValueError(
            f"Not enough images in {source_folder}"
        )

    train_files, remaining = train_test_split(
        files, test_size=0.30, random_state=42
    )

    val_files, test_files = train_test_split(
        remaining, test_size=0.50, random_state=42
    )

    for split, split_files in [
        ("train", train_files),
        ("validation", val_files),
        ("test", test_files)
    ]:
        destination = OUTPUT / split / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for file in split_files:
            shutil.copy2(file, destination / file.name)

        print(split, class_name, len(split_files))

print("Dataset split completed!")
