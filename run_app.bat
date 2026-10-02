@echo off
echo Installing libraries (first time only)...
pip install -r requirements.txt
echo.
echo Starting the project. Open http://127.0.0.1:5000 in Chrome. Login: admin / admin123
python app.py
pause
