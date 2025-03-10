#!/bin/bash

# Step 1: Build the Docker image
docker build -t model-api .

# Step 2: Run the Docker container
docker run -p 8000:8000 --name model-api-container model-api
