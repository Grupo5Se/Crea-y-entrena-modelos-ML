@echo off
cd /d %~dp0..
if not exist backend\.venv (
  python -m venv backend\.venv
)
call backend\.venv\Scripts\activate
python -m backend.scripts.train_model
pause
