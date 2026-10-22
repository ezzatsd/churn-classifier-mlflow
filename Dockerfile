FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN groupadd --system app && useradd --system --gid app --home /app app
COPY --chown=app:app configs configs
COPY --chown=app:app src src
RUN mkdir -p /app/data /app/artifacts && chown -R app:app /app

USER app

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--config", "configs/config.yaml"]
