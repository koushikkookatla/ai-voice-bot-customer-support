from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WHISPER_MODEL_SIZE: str = "base"
    INTENT_MODEL_PATH: str = "models/intent_model"
    INTENT_MODEL_NAME: str = "distilbert-base-uncased"
    NUM_INTENTS: int = 10
    MAX_SEQ_LENGTH: int = 128
    TTS_LANGUAGE: str = "en"
    TTS_SLOW: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    DEVICE: str = "cpu"

    class Config:
        env_file = ".env"


settings = Settings()

INTENT_LABELS = [
    "order_status", "cancel_order", "refund_request",
    "subscription_issue", "payment_problem", "account_access",
    "product_inquiry", "shipping_info", "return_request",
    "general_complaint"
]

INTENT_TO_IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
IDX_TO_INTENT = {idx: label for idx, label in enumerate(INTENT_LABELS)}
