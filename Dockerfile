# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    git \
    lsof \
    coreutils \
    vim \
    nohup \
    && rm -rf /var/lib/apt/lists/*

RUN python -m spacy download en_core_web_sm && \
    python -m spacy download zh_core_web_sm

# Expose port 8003 to the outside world
EXPOSE 8003

