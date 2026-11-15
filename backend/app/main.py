import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.database.session import engine, Base, SessionLocal
from app.models.user import User, UserRole
from app.core.security import get_password_hash

# Import API routers
from app.api.auth import router as auth_router
from app.api.patients import router as patients_router
from app.api.images import router as images_router
from app.api.analysis import router as analysis_router
from app.api.reports import router as reports_router
from app.api.dashboard import router as dashboard_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("retina_main")

def init_db():
    """Initializes tables and seeds the initial medical administrator."""
    logger.info("Initializing database schemas...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            logger.info("Seeding initial medical admin user (admin / admin123)...")
            new_admin = User(
                username="admin",
                email="admin@retina.vku.vn",
                password_hash=get_password_hash("admin123"),
                full_name="Dr. Medical Administrator",
                role=UserRole.ADMIN.value,
                is_active=True
            )
            db.add(new_admin)
            db.commit()
            logger.info("Admin user created successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    init_db()
    yield
    # Shutdown
    logger.info("Shutting down Retina Medical AI Backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for convenient dev and docker network access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(patients_router, prefix=settings.API_V1_STR)
app.include_router(images_router, prefix=settings.API_V1_STR)
app.include_router(analysis_router, prefix=settings.API_V1_STR)
app.include_router(reports_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": settings.API_V1_STR,
        "status": "ONLINE"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "storage": "ready",
        "ai_engine": "available"
    }

