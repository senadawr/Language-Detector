@echo off

REM Run with only the 15 most common languages
set COMMON_LANGS_ONLY=1
py language_detector.py

pause