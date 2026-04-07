import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

from src.api.routes import router
from src.validation.corpus_loader import build_corpus

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # build GST corpus on startup if not already built
    logger.info("Building GST corpus...")
    build_corpus()
    logger.info("GST corpus ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Invoice Intelligence API",
    description="OCR + LLM field extraction + GST compliance validation for invoices",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")