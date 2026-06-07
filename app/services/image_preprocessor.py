from pathlib import Path

import cv2

from app.core.config import (
    PROCESSED_PREPROCESSED_IMAGES_DIR,
    PREPROCESS_DENOISE,
    PREPROCESS_GRAYSCALE,
    PREPROCESS_RESIZE_WIDTH,
    PREPROCESS_THRESHOLD,
)
from app.schemas.document import ImagePreprocessResponse


def create_preprocessed_image_dir(document_id: str) -> Path:
    output_dir = PROCESSED_PREPROCESSED_IMAGES_DIR / document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resize_image(image, target_width: int):
    height, width = image.shape[:2]

    if width <= target_width:
        return image

    scale_ratio = target_width / width
    target_height = int(height * scale_ratio)

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )


def preprocess_single_image(image_path: str, output_path: Path) -> str:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = resize_image(image, PREPROCESS_RESIZE_WIDTH)

    if PREPROCESS_GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if PREPROCESS_DENOISE:
        image = cv2.fastNlMeansDenoising(image, None, h=10)

    if PREPROCESS_THRESHOLD:
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )

    cv2.imwrite(str(output_path), image)

    return str(output_path)


def preprocess_document_images(
    document_id: str,
    page_images: list[str],
) -> ImagePreprocessResponse:
    output_dir = create_preprocessed_image_dir(document_id)
    preprocessed_images: list[str] = []

    for index, image_path in enumerate(page_images, start=1):
        output_path = output_dir / f"page_{index}.png"

        preprocessed_image_path = preprocess_single_image(
            image_path=image_path,
            output_path=output_path,
        )

        preprocessed_images.append(preprocessed_image_path)

    return ImagePreprocessResponse(
        document_id=document_id,
        input_images=page_images,
        preprocessed_images=preprocessed_images,
        page_count=len(preprocessed_images),
        status="preprocessed",
    )
