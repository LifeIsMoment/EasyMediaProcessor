@echo off
echo EasyMediaProcessor 빌드 시작...

REM 이전 빌드 파일 삭제
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "*.spec" del "*.spec"

REM PyInstaller로 exe 파일 생성
echo 실행파일 생성 중...
pyinstaller --onefile ^
    --windowed ^
    --name "EasyMediaProcessor" ^
    --icon=icon.ico ^
    --add-data "venv/Lib/site-packages/PySide6;PySide6" ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=cv2 ^
    --hidden-import=PIL ^
    --hidden-import=ffmpeg ^
    --hidden-import=numpy ^
    main.py

if %errorlevel% equ 0 (
    echo.
    echo 빌드 완료!
    echo 실행파일 위치: dist\EasyMediaProcessor.exe
    echo.
    pause
) else (
    echo.
    echo 빌드 실패!
    echo.
    pause
)