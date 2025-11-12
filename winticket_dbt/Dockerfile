FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DBT_PROFILES_DIR=/workspace/app/winticket_dbt/profiles \
    WINTICKET_DATA_ROOT=/workspace/app/data/winticket_lake

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=180 dbt-core==1.9.1 dbt-duckdb==1.9.1

WORKDIR /workspace/app/winticket_dbt
ENTRYPOINT ["dbt"]
