# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpython3-dev g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application files
COPY app.py /app/
COPY word2vec_twitter.model /app/
COPY randomforest_bully_predictor.pkl /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the full NLTK data, including `punkt` and its components
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader all

# Stage 2: Final runtime
FROM gcr.io/distroless/python3

WORKDIR /app

# Copy Python environment and application files from the builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
COPY --from=builder /root/nltk_data /usr/share/nltk_data
COPY --from=builder /app /app

# Expose the application port
EXPOSE 8080

# Set Python path and NLTK data path
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages
ENV NLTK_DATA=/usr/share/nltk_data

# Run the application
CMD ["app.py"]
