FROM manimcommunity/manim:v0.19.0

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["sh", "-lc", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
