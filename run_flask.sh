#!/bin/bash
# Script to start Flask API Server
# This starts the REST API on http://localhost:5000

echo ""
echo "========================================"
echo "  🚀 Starting AutoML Flask API"
echo "========================================"
echo ""
echo "API Server: http://localhost:5000"
echo "API Documentation: http://localhost:5000/api/info"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app_flask.py
