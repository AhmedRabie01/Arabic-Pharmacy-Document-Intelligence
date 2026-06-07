from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import (
    HF_TOKEN,
    SIGNATURE_YOLO_CONFIDENCE_THRESHOLD,
    SIGNATURE_YOLO_HF_FALLBACK_FILENAME,
    SIGNATURE_YOLO_HF_FALLBACK_REPO_ID,
    SIGNATURE_YOLO_HF_FILENAME,
    SIGNATURE_YOLO_HF_REPO_ID,
    SIGNATURE_YOLO_MODEL_PATH,
    SIGNATURE_YOLO_USE_HF_DEFAULT,
)
from app.schemas.document import SignatureDetectionResult

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from huggingface_hub import hf_hub_download, login as hf_login
except Exception:
    hf_hub_download = None
    hf_login = None


DEFAULT_SIGNATURE_CLASS_NAMES = {
    "signature",
    "sign",
    "handwritten_signature",
    "signature_mark",
}

HF_TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


def _is_remote_model_reference(model_reference: str) -> bool:
    parsed = urlparse(model_reference)
    return parsed.scheme in {"http", "https"}


def _hf_spec_to_parts(model_reference: str) -> tuple[str, str] | None:
    """
    Parse hf://<repo_id>/<filename> into (repo_id, filename).
    Example:
      hf://ori-ron/yolov8s-signature-detector/yolov8s.pt
      -> ("ori-ron/yolov8s-signature-detector", "yolov8s.pt")
    """
    if not model_reference.startswith("hf://"):
        return None

    path_value = model_reference.removeprefix("hf://").strip("/")
    parts = path_value.split("/")
    if len(parts) < 3:
        return None

    filename = parts[-1]
    repo_id = "/".join(parts[:-1])
    return repo_id, filename


def _resolve_hf_token() -> str | None:
    token_candidates = [
        os.getenv("HF_TOKEN", "").strip(),
        os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip(),
        os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip(),
        HF_TOKEN.strip(),
    ]
    for value in token_candidates:
        if value:
            return value
    return None


def _download_model_from_hf(repo_id: str, filename: str) -> tuple[str | None, list[str]]:
    if hf_hub_download is None:
        return None, ["error:huggingface_hub_not_installed"]

    token = _resolve_hf_token()

    # Optional programmatic login helps with gated models.
    if token and hf_login is not None:
        try:
            hf_login(
                token=token,
                add_to_git_credential=False,
                skip_if_logged_in=True,
            )
        except TypeError:
            # Compatibility with older huggingface_hub signatures.
            try:
                hf_login(
                    token=token,
                    add_to_git_credential=False,
                )
            except Exception:
                pass
        except Exception:
            pass

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
        return downloaded_path, [f"info:hf_model_downloaded:{repo_id}/{filename}"]
    except Exception as error:
        error_text = str(error)
        lower_error = error_text.lower()

        messages = [f"warn:hf_download_failed:{repo_id}/{filename}:{error_text}"]
        if (
            "401" in error_text
            or "cannot access gated repo" in lower_error
            or "please log in" in lower_error
            or "unauthorized" in lower_error
        ):
            if not token:
                messages.append("hint:missing_hf_token")
            messages.append(f"link:hf_token_settings:{HF_TOKEN_SETTINGS_URL}")
            messages.append(f"link:hf_model_access:https://huggingface.co/{repo_id}")
            messages.append("hint:put_HF_TOKEN_in_.env_or_system_env")

        return None, messages


def _resolve_model_reference(model_path: str) -> tuple[str | None, list[str]]:
    messages: list[str] = []

    hf_parts = _hf_spec_to_parts(model_reference=model_path)
    if hf_parts is not None:
        repo_id, filename = hf_parts
        downloaded_path, download_messages = _download_model_from_hf(
            repo_id=repo_id,
            filename=filename,
        )
        messages.extend(download_messages)
        if downloaded_path is not None:
            messages.append(f"info:model_source:hf_spec:{model_path}")
            return downloaded_path, messages
        return None, messages

    if _is_remote_model_reference(model_reference=model_path):
        messages.append(f"info:model_source:url:{model_path}")
        return model_path, messages

    local_model_file = Path(model_path)
    if local_model_file.exists():
        messages.append(f"info:model_source:local:{model_path}")
        return str(local_model_file), messages

    if not SIGNATURE_YOLO_USE_HF_DEFAULT:
        return None, [f"error:model_not_found:{model_path}"]

    default_repo_id = SIGNATURE_YOLO_HF_REPO_ID.strip()
    default_filename = SIGNATURE_YOLO_HF_FILENAME.strip()
    fallback_repo_id = SIGNATURE_YOLO_HF_FALLBACK_REPO_ID.strip()
    fallback_filename = (
        SIGNATURE_YOLO_HF_FALLBACK_FILENAME.strip()
        or default_filename
        or "yolov8s.pt"
    )

    if not default_repo_id:
        return None, [f"error:model_not_found:{model_path}"]

    first_path, first_messages = _download_model_from_hf(
        repo_id=default_repo_id,
        filename=default_filename,
    )
    messages.extend(first_messages)
    if first_path is not None:
        messages.append(
            "info:model_source:hf_default_fallback:"
            f"{default_repo_id}/{default_filename}"
        )
        return first_path, messages

    if fallback_repo_id and (
        fallback_repo_id != default_repo_id or fallback_filename != default_filename
    ):
        second_path, second_messages = _download_model_from_hf(
            repo_id=fallback_repo_id,
            filename=fallback_filename,
        )
        messages.extend(second_messages)
        if second_path is not None:
            messages.append(
                "info:model_source:hf_secondary_fallback:"
                f"{fallback_repo_id}/{fallback_filename}"
            )
            return second_path, messages

    messages.append(
        "error:hf_model_resolution_failed:"
        f"default={default_repo_id}/{default_filename}"
        f";fallback={fallback_repo_id}/{fallback_filename}"
    )
    return None, messages


def _resolve_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path

    return SIGNATURE_YOLO_MODEL_PATH


def _resolve_conf_threshold(conf_threshold: float | None) -> float:
    if conf_threshold is not None:
        return conf_threshold

    return SIGNATURE_YOLO_CONFIDENCE_THRESHOLD


def _class_name_from_result_names(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        value = names.get(class_id)
        if value is None:
            value = names.get(str(class_id), str(class_id))
        return str(value)

    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def detect_signature_with_yolo(
    image_path: str,
    model_path: str | None = None,
    conf_threshold: float | None = None,
    allowed_class_names: set[str] | None = None,
) -> SignatureDetectionResult:
    """
    Detect signature region from one page image using YOLO.

    This service is visual-only:
    - It detects whether signature-like handwriting exists.
    - It returns bounding box and confidence.
    - It does not read signer name text.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=[f"error:image_not_found:{image_path}"],
            confidence=0.0,
        )

    if YOLO is None:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=["error:ultralytics_not_installed"],
            confidence=0.0,
        )

    selected_model_path = _resolve_model_path(model_path=model_path)
    selected_conf = _resolve_conf_threshold(conf_threshold=conf_threshold)
    target_class_names = {
        name.lower().strip()
        for name in (allowed_class_names or DEFAULT_SIGNATURE_CLASS_NAMES)
    }
    resolved_model_reference, model_messages = _resolve_model_reference(
        model_path=selected_model_path
    )
    if resolved_model_reference is None:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=model_messages,
            confidence=0.0,
        )

    try:
        model = YOLO(resolved_model_reference)
        predictions = model.predict(
            source=str(image_file),
            conf=selected_conf,
            verbose=False,
        )
    except Exception as error:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=[*model_messages, f"error:yolo_inference_failed:{error}"],
            confidence=0.0,
        )

    if not predictions:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=[*model_messages, "info:no_predictions"],
            confidence=0.0,
        )

    first_result = predictions[0]
    boxes = getattr(first_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=[*model_messages, "info:no_boxes"],
            confidence=0.0,
        )

    best_detection = None
    best_confidence = 0.0
    best_class_name = ""

    names = getattr(first_result, "names", {})

    for box in boxes:
        class_id = int(box.cls[0]) if box.cls is not None else -1
        class_name = _class_name_from_result_names(names=names, class_id=class_id)
        class_name_lower = class_name.lower().strip()
        confidence = float(box.conf[0]) if box.conf is not None else 0.0

        # If model has explicit class names, keep only signature-related classes.
        # If model is a single-class model with id 0, allow it by default.
        class_allowed = (
            class_name_lower in target_class_names
            or (len(names) == 1 and class_id == 0)
        )
        if not class_allowed:
            continue

        if confidence > best_confidence:
            xyxy = box.xyxy[0].tolist()
            best_detection = [round(float(value), 2) for value in xyxy]
            best_confidence = confidence
            best_class_name = class_name

    if best_detection is None:
        return SignatureDetectionResult(
            signature_present=False,
            signer_name=None,
            signature_bbox=None,
            nearby_text=[*model_messages, "info:no_signature_class_detected"],
            confidence=0.0,
        )

    return SignatureDetectionResult(
        signature_present=True,
        signer_name=None,
        signature_bbox=best_detection,
        nearby_text=[
            *model_messages,
            f"model:{resolved_model_reference}",
            f"class:{best_class_name}",
        ],
        confidence=round(best_confidence, 3),
    )
