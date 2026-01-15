# Use the official Unstructured image
FROM downloads.unstructured.io/unstructured-io/unstructured:latest

WORKDIR /app

# Switch to root to install dependencies
USER root

# Wolfi equivalents:
# build-essential -> build-base
# libpq-dev       -> postgresql-dev (or postgresql-16-dev depending on version)
# musl-dev        -> Not needed (Wolfi uses glibc)
RUN apk update && apk add --no-cache \
    build-base \
    postgresql-dev \
    python3-dev \
    gcc

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]