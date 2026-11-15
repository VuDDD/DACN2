import logging
from app.workers.celery_app import celery_app
from app.database.session import SessionLocal
from app.services.inference_pipeline import process_retinal_analysis

logger = logging.getLogger("celery_tasks")

@celery_app.task(name="celery_run_analysis", bind=True, max_retries=1)
def celery_run_analysis(self, analysis_id: int):
    """
    Celery asynchronous worker task executing Deep Learning retinal inference.
    """
    logger.info(f"[Celery] Received task for Analysis ID {analysis_id}")
    db = SessionLocal()
    try:
        process_retinal_analysis(analysis_id, db)
        logger.info(f"[Celery] Successfully completed Analysis ID {analysis_id}")
        return {"status": "SUCCESS", "analysis_id": analysis_id}
    except Exception as exc:
        logger.error(f"[Celery] Task failed for Analysis ID {analysis_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()
