@echo off
title Automatic Video Highlight Detection - Flask Server
echo ======================================================================
echo Starting HighlightAI Web Server...
echo Model: models/best_bilstm.pt
echo URL:   http://127.0.0.1:5000
echo ======================================================================
"C:\Users\mahin\OneDrive\Desktop\FY PROJECT\lighthouse\venv\Scripts\python.exe" app.py
pause
