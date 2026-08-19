from pathlib import Path

import cv2


OUTPUT = Path("enhanced")

OUTPUT.mkdir(
    exist_ok=True
)


def enhance_image(
    input_path,
    filename
):

    image = cv2.imread(
        str(input_path)
    )

    if image is None:

        raise ValueError(
            "Invalid image"
        )

    denoised = cv2.fastNlMeansDenoisingC
        image,
        None,
        7,
        7,
        7,
        21
    )

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.cvtColor(
        cv2.merge(
            (l, a, b)
        ),
        cv2.COLOR_LAB2BGR
    )

    output = (
        OUTPUT /
        f"enhanced_{Path(filename).name}"
    )

    cv2.imwrite(
        str(output),
        result
    )

    return str(
        output
    ).replace("\\", "/")