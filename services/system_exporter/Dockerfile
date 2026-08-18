FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY system_exporter.py .
EXPOSE 9106
CMD ["uvicorn", "system_exporter:app", "--host", "0.0.0.0", "--port", "9106"]
