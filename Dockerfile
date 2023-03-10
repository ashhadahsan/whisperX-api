FROM python:3.9

WORKDIR /app

COPY . /app

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
