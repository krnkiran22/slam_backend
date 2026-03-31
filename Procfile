web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
worker: celery -A api.tasks worker --loglevel=info --concurrency=2
