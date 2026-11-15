import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "Retina Medical AI Platform"
    PROJECT_DESCRIPTION: str = "Nền tảng phân vùng và định lượng tổn thương y khoa tự động ứng dụng Deep Learning"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "vkuretina-medical-ai-super-secret-key-2026-secure")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day
    
    # Database
    # Defaults to SQLite for local standalone development, switches to PostgreSQL via env var
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./retina_medical.db"
    )
    
    # Redis / Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # MinIO / Object Storage
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "retina-medical")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1")
    
    # Local Storage fallback (used when MinIO service is offline or in local dev)
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./backend_storage")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]
    
    # AI Model & Checkpoint Paths
    IDRID_CHECKPOINT_DIR: str = os.getenv("IDRID_CHECKPOINT_DIR", "./ai/checkpoints")
    DRIVE_CHECKPOINT_DIR: str = os.getenv("DRIVE_CHECKPOINT_DIR", "./ai/checkpoints")
    DEVICE: str = os.getenv("DEVICE", "auto")  # 'auto', 'cuda', 'cpu'

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()
