FROM manimcommunity/manim:v0.19.0

USER root
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/work /app/work/jobs && chmod -R a+rwX /app/work

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["sh", "-lc", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
