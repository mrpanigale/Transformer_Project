FROM public.ecr.aws/docker/library/python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir \
    --index-url https://package-mirror.liara.ir/repository/pypi/simple \
    -r requirements-api.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]