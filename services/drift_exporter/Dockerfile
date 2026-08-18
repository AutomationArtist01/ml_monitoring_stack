FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY drift_exporter.py build_reference.py ./
COPY reference ./reference
EXPOSE 9105
CMD ["uvicorn", "drift_exporter:app", "--host", "0.0.0.0", "--port", "9105"]
