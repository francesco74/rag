# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- Install System Dependencies ---
# Update package lists and install LibreOffice.
# Cleaning up apt lists afterward also reduces image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir is used to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers.cross_encoder import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=1024)"

# --- Copy Application Code ---
# Copy the rest of the application code into the container
COPY src/. .

# --- Configure and Run ---
# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the app using Gunicorn for production
CMD ["sh", "-c", "gunicorn --worker-class gevent --workers 4 --bind 0.0.0.0:5000 --timeout 300 app:app"]
