from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # App
    APP_NAME: str = "Pharmacy Invoice OCR System"
    APP_ENV: str = "local"
    DEBUG: bool = True

    # Raw folders
    DATA_DIR: str = "data"
    APPROVED_DATASET_DIR: str = "data/approved_dataset"
    FEW_SHOT_DIR: str = "data/few_shot"
    RAWDATA_DIR: str = "rawdata"
    ALL_INVOICES_DIR: str = "rawdata/all_invoices"
    DAILY_INVOICE_DIR: str = "rawdata/daily_invoices"
    MONTHLY_INVOICE_DIR: str = "rawdata/monthly_invoices"
    MIXED_DOCUMENTS_DIR: str = "rawdata/mixed_documents"
    UNKNOWN_DOCUMENTS_DIR: str = "rawdata/unknown"

    # Processed folders
    PROCESSED_DIR: str = "processed"
    PROCESSED_IMAGES_DIR: str = "processed/images"
    PROCESSED_TEXT_DIR: str = "processed/text"
    PROCESSED_METADATA_DIR: str = "processed/metadata"

    # Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: str = ".pdf,.jpg,.jpeg,.png"

   # OCR
    DOCUMENT_AI_PROVIDER: str = "paddleocr"
    LOCAL_OCR_ENGINE: str = "paddleocr"
    OCR_LANGUAGE: str = "ar"
    OCR_CONFIDENCE_THRESHOLD: float = 0.70
    OCR_USE_ANGLE_CLS: bool = True
    OCR_USE_PREPROCESSED_IMAGES: bool = True
    # Chandra / Ollama
    CHANDRA_MODEL: str = "fredrezones55/chandra-ocr-2:patch"
    CHANDRA_BASE_URL: str = "http://localhost:11434"
    CHANDRA_OUTPUT_FORMAT: str = "json"

    CHANDRA_TIMEOUT_SECONDS: int = 600
    CHANDRA_NUM_PREDICT: int = 800
    CHANDRA_TEMPERATURE: float = 0
    ENABLE_CHANDRA_CACHE: bool = True
    PROCESSED_CHANDRA_JSON_DIR: str = "processed/chandra_json"

    # Output
    PROCESSED_OCR_JSON_DIR: str = "processed/ocr_json"
    PROCESSED_SPATIAL_JSON_DIR: str = "processed/spatial_json"
    PROCESSED_CLASSIFICATION_JSON_DIR: str = "processed/classification_json"
    PROCESSED_EXTRACTION_JSON_DIR: str = "processed/extraction_json"
    PROCESSED_SIGNATURE_JSON_DIR: str = "processed/signature_json"
    PROCESSED_PIPELINE_JSON_DIR: str = "processed/pipeline_json"
    WEB_DIR: str = "app/web"

    # PDF/Image
    PDF_RENDER_DPI: int = 300
    SAVE_PAGE_IMAGES: bool = True
    IMAGE_FORMAT: str = "png"
    
    # Local LLM
    USE_LOCAL_LLM: bool = True
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    HF_TOKEN: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///./invoice_ocr.db"

    PROCESSED_PREPROCESSED_IMAGES_DIR: str = "processed/preprocessed_images"

    # OpenAI
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_TIMEOUT_SECONDS: int = 120
    OPENAI_IMAGE_DETAIL: str = "high"
    ENABLE_OPENAI_CACHE: bool = True
    PROCESSED_OPENAI_JSON_DIR: str = "processed/openai_json"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = ""

    # Azure Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: str = ""
    AZURE_DOCUMENT_INTELLIGENCE_KEY: str = ""
    AZURE_DOCUMENT_INTELLIGENCE_MODEL: str = "prebuilt-invoice"

    # Google Document AI
    GOOGLE_DOCUMENT_AI_PROJECT_ID: str = ""
    GOOGLE_DOCUMENT_AI_LOCATION: str = "us"
    GOOGLE_DOCUMENT_AI_PROCESSOR_ID: str = ""

    # fallbacks
    ENABLE_FALLBACK: bool = True
    PRIMARY_PROVIDER: str = "paddleocr"
    FALLBACK_PROVIDER: str = "chandra_ollama"
    FALLBACK_CONFIDENCE_THRESHOLD: float = 0.75

    # Signature detection
    SIGNATURE_YOLO_MODEL_PATH: str = "models/signature_yolo/best.pt"
    SIGNATURE_YOLO_CONFIDENCE_THRESHOLD: float = 0.25
    SIGNATURE_YOLO_USE_HF_DEFAULT: bool = True
    SIGNATURE_YOLO_HF_REPO_ID: str = "tech4humans/yolov8s-signature-detector"
    SIGNATURE_YOLO_HF_FILENAME: str = "yolov8s.pt"
    SIGNATURE_YOLO_HF_FALLBACK_REPO_ID: str = "ori-ron/yolov8s-signature-detector"
    SIGNATURE_YOLO_HF_FALLBACK_FILENAME: str = "yolov8s.pt"

    PREPROCESS_GRAYSCALE: bool = True
    PREPROCESS_DENOISE: bool = True
    PREPROCESS_THRESHOLD: bool = True
    PREPROCESS_RESIZE_WIDTH: int = 1600

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value

        text_value = str(value).strip().lower()

        if text_value in {"1", "true", "yes", "on", "debug", "dev", "development"}:
            return True

        if text_value in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False

        return False

settings = Settings()

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / settings.DATA_DIR
APPROVED_DATASET_DIR = BASE_DIR / settings.APPROVED_DATASET_DIR
FEW_SHOT_DIR = BASE_DIR / settings.FEW_SHOT_DIR
RAWDATA_DIR = BASE_DIR / settings.RAWDATA_DIR
ALL_INVOICES_DIR = BASE_DIR / settings.ALL_INVOICES_DIR
DAILY_INVOICE_DIR = BASE_DIR / settings.DAILY_INVOICE_DIR
MONTHLY_INVOICE_DIR = BASE_DIR / settings.MONTHLY_INVOICE_DIR
MIXED_DOCUMENTS_DIR = BASE_DIR / settings.MIXED_DOCUMENTS_DIR
UNKNOWN_DOCUMENTS_DIR = BASE_DIR / settings.UNKNOWN_DOCUMENTS_DIR

PROCESSED_DIR = BASE_DIR / settings.PROCESSED_DIR
PROCESSED_IMAGES_DIR = BASE_DIR / settings.PROCESSED_IMAGES_DIR
PROCESSED_TEXT_DIR = BASE_DIR / settings.PROCESSED_TEXT_DIR
PROCESSED_METADATA_DIR = BASE_DIR / settings.PROCESSED_METADATA_DIR

PROCESSED_PREPROCESSED_IMAGES_DIR = BASE_DIR / settings.PROCESSED_PREPROCESSED_IMAGES_DIR
PROCESSED_OCR_JSON_DIR = BASE_DIR / settings.PROCESSED_OCR_JSON_DIR
PROCESSED_CHANDRA_JSON_DIR = BASE_DIR / settings.PROCESSED_CHANDRA_JSON_DIR
PROCESSED_OPENAI_JSON_DIR = BASE_DIR / settings.PROCESSED_OPENAI_JSON_DIR
PROCESSED_SPATIAL_JSON_DIR = BASE_DIR / settings.PROCESSED_SPATIAL_JSON_DIR
PROCESSED_CLASSIFICATION_JSON_DIR = BASE_DIR / settings.PROCESSED_CLASSIFICATION_JSON_DIR
PROCESSED_EXTRACTION_JSON_DIR = BASE_DIR / settings.PROCESSED_EXTRACTION_JSON_DIR
PROCESSED_SIGNATURE_JSON_DIR = BASE_DIR / settings.PROCESSED_SIGNATURE_JSON_DIR
PROCESSED_PIPELINE_JSON_DIR = BASE_DIR / settings.PROCESSED_PIPELINE_JSON_DIR
WEB_DIR = BASE_DIR / settings.WEB_DIR

PDF_RENDER_DPI = settings.PDF_RENDER_DPI
IMAGE_FORMAT = settings.IMAGE_FORMAT

PREPROCESS_GRAYSCALE = settings.PREPROCESS_GRAYSCALE
PREPROCESS_DENOISE = settings.PREPROCESS_DENOISE
PREPROCESS_THRESHOLD = settings.PREPROCESS_THRESHOLD
PREPROCESS_RESIZE_WIDTH = settings.PREPROCESS_RESIZE_WIDTH
DOCUMENT_AI_PROVIDER = settings.DOCUMENT_AI_PROVIDER
LOCAL_OCR_ENGINE = settings.LOCAL_OCR_ENGINE
OCR_LANGUAGE = settings.OCR_LANGUAGE
OCR_CONFIDENCE_THRESHOLD = settings.OCR_CONFIDENCE_THRESHOLD
OCR_USE_ANGLE_CLS = settings.OCR_USE_ANGLE_CLS
OCR_USE_PREPROCESSED_IMAGES = settings.OCR_USE_PREPROCESSED_IMAGES

CHANDRA_MODEL = settings.CHANDRA_MODEL
CHANDRA_BASE_URL = settings.CHANDRA_BASE_URL
CHANDRA_OUTPUT_FORMAT = settings.CHANDRA_OUTPUT_FORMAT

CHANDRA_TIMEOUT_SECONDS = settings.CHANDRA_TIMEOUT_SECONDS
CHANDRA_NUM_PREDICT = settings.CHANDRA_NUM_PREDICT
CHANDRA_TEMPERATURE = settings.CHANDRA_TEMPERATURE
ENABLE_CHANDRA_CACHE = settings.ENABLE_CHANDRA_CACHE

DOCUMENT_AI_PROVIDER = settings.DOCUMENT_AI_PROVIDER

OPENAI_BASE_URL = settings.OPENAI_BASE_URL
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_MODEL = settings.OPENAI_MODEL
OPENAI_TIMEOUT_SECONDS = settings.OPENAI_TIMEOUT_SECONDS
OPENAI_IMAGE_DETAIL = settings.OPENAI_IMAGE_DETAIL
ENABLE_OPENAI_CACHE = settings.ENABLE_OPENAI_CACHE

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
AZURE_DOCUMENT_INTELLIGENCE_KEY = settings.AZURE_DOCUMENT_INTELLIGENCE_KEY
AZURE_DOCUMENT_INTELLIGENCE_MODEL = settings.AZURE_DOCUMENT_INTELLIGENCE_MODEL

GOOGLE_DOCUMENT_AI_PROJECT_ID = settings.GOOGLE_DOCUMENT_AI_PROJECT_ID
GOOGLE_DOCUMENT_AI_LOCATION = settings.GOOGLE_DOCUMENT_AI_LOCATION
GOOGLE_DOCUMENT_AI_PROCESSOR_ID = settings.GOOGLE_DOCUMENT_AI_PROCESSOR_ID

ENABLE_FALLBACK = settings.ENABLE_FALLBACK
PRIMARY_PROVIDER = settings.PRIMARY_PROVIDER
FALLBACK_PROVIDER = settings.FALLBACK_PROVIDER
FALLBACK_CONFIDENCE_THRESHOLD = settings.FALLBACK_CONFIDENCE_THRESHOLD
DATABASE_URL = settings.DATABASE_URL
HF_TOKEN = settings.HF_TOKEN

SIGNATURE_YOLO_MODEL_PATH = settings.SIGNATURE_YOLO_MODEL_PATH
SIGNATURE_YOLO_CONFIDENCE_THRESHOLD = settings.SIGNATURE_YOLO_CONFIDENCE_THRESHOLD
SIGNATURE_YOLO_USE_HF_DEFAULT = settings.SIGNATURE_YOLO_USE_HF_DEFAULT
SIGNATURE_YOLO_HF_REPO_ID = settings.SIGNATURE_YOLO_HF_REPO_ID
SIGNATURE_YOLO_HF_FILENAME = settings.SIGNATURE_YOLO_HF_FILENAME
SIGNATURE_YOLO_HF_FALLBACK_REPO_ID = settings.SIGNATURE_YOLO_HF_FALLBACK_REPO_ID
SIGNATURE_YOLO_HF_FALLBACK_FILENAME = settings.SIGNATURE_YOLO_HF_FALLBACK_FILENAME

ALLOWED_EXTENSIONS = {
    ext.strip().lower()
    for ext in settings.ALLOWED_EXTENSIONS.split(",")
}

MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
