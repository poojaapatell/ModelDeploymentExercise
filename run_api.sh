#!/bin/bash

# Build the Docker image from the Dockerfile
echo "Building the Docker image..."
docker build -t my-model-api .

# Run the Docker container
echo "Running the Docker container..."
docker run -p 8080:8080 my-model-api

# Print message after the container starts
echo "API is now running on http://127.0.0.1:8080"
