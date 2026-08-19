from pathlib import Path

import random
import shutil


random.seed(42)


SOURCE = Path(
    "data/raw"
)

DESTINATION = Path(
    "data"
)


EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def prepare_dataset():

    classes = [
        folder
        for folder in SOURCE.iterdir()
        if folder.is_dir()
    ]

    if not classes:

        print(
            "No dataset found."
        )

        print(
            "Put class folders inside:"
        )

        print(
            "data/raw/"
        )

        return

    for class_folder in classes:

        images = [
            image
            for image in class_folder.rglob("*")
            if image.suffix.lower()
            in EXTENSIONS
        ]

        random.shuffle(
            images
        )

        total = len(images)

        train_end = int(
            total * 0.70
        )

        val_end = int(
            total * 0.85
        )

        train_images = images[
            :train_end
        ]

        val_images = images[
            train_end:val_end
        ]

        test_images = images[
            val_end:
        ]

        groups = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split, files in groups.items():

            destination = (
                DESTINATION /
                split /
                class_folder.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for number, image in enumerate(files):

                shutil.copy2(
                    image,
                    destination /
                    f"{number}_{image.name}"
                )

        print(
            class_folder.name,
            "Train:",
            len(train_images),
            "Val:",
            len(val_images),
            "Test:",
            len(test_images)
        )


if __name__ == "__main__":

    prepare_dataset()