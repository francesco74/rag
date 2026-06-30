# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Install System Dependencies ---
# Update package lists and install LibreOffice.
# Using --no-install-recommends keeps the image size smaller.
# Cleaning up apt lists afterward also reduces image size.
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Copy the requirements file first to leverage Docker cache
COPY saga_extract/requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir is used to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
COPY common/ ./common/
COPY saga_extract/src/. .

# --- Configure and Run ---
# Make port 8080 available to the world outside this container

# Define the environment variable for the API key (to be set at runtime)
ENV LOG_LEVEL=""
ENV DB_HOST=""
ENV DB_USER=""
ENV DB_PORT=""
ENV DB_PASS=""
ENV DB_NAME=""
ENV BROKER_HOST=""
ENV BROKER_PORT=""
ENV BROKER_USERNAME=""
ENV BROKER_PASSWORD=""

# Define the command to run the app using Gunicorn for production
CMD ["sh", "-c", "python estrattore.py"]
