#!/bin/bash
docker build -t onnx_serving_study_serving -f Dockerfile.onnx_serving .
docker build -t onnx-serving-study_client -f Dockerfile.client .
docker-compose up -d