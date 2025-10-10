#!/bin/bash

echo "Building and running the AegisAI Docker containers..."
echo "API will be available at http://localhost:8000"
echo "Dashboard will be available at http://localhost:8501"

docker-compose up --build
