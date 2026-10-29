FROM python:3.11-slim

WORKDIR /app

# Torch는 별도 레이어로 설치해 캐시 재사용
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch

# 나머지 패키지만 별도로 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


