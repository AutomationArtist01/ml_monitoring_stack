FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY loadgen.py .
EXPOSE 8090
CMD ["uvicorn", "loadgen:app", "--host", "0.0.0.0", "--port", "8090"]
