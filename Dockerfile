FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant ./support_assistant
COPY documents ./documents

CMD ["uvicorn", "support_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]