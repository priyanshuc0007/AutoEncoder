#!/bin/bash
# Script to start Streamlit App
# This opens the AutoML interface in your browser

echo ""
echo "========================================"
echo "  🚀 Starting AutoML Streamlit App"
echo "========================================"
echo ""
echo "Opening browser to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app_streamlit.py --server.port=8501
