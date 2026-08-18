FROM python:3.12-slim
WORKDIR /app
ENV ZENML_ANALYTICS_OPT_IN=false ZENML_LOGGING_VERBOSITY=INFO AUTO_OPEN_DASHBOARD=false
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline.py run_pipeline.py ./
# initialise a ZenML repo (local stack) inside the image so `python run_pipeline.py` just works
RUN zenml init >/dev/null 2>&1 || true
CMD ["python", "run_pipeline.py"]
