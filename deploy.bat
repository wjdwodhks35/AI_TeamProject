@echo off
chcp 65001 >nul 2>&1
setlocal

rem 윈도우에서 API 와 UI 를 한 포트(8010)로 띄운다. 더블클릭하거나 cmd 에서 실행.
rem
rem   deploy.bat
rem
rem 접속:  http://localhost:8010      (API 문서: /docs, 상태: /health)
rem
rem 환경변수
rem   set SKIP_INSTALL=1   의존성 설치 건너뛰기 (두 번째 실행부터 빠르다)
rem   set RELOAD=1         파이썬 코드 변경 시 자동 재시작
rem
rem UI(html/css/js)는 고칠 때마다 브라우저 새로고침만 하면 바로 반영된다.
rem 리눅스 서버 배포는 deploy.sh 를 쓴다. 이 파일은 로컬 확인용이라 localhost 에만
rem 바인딩한다 (외부에서 접속 불가).

cd /d "%~dp0"
set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "PORT=8010"

rem ---------------------------------------------------------------- 1. 가상환경

if exist "%VENV_PY%" goto :venv_ready

set "PY="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto :found_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
:found_python

if not defined PY (
  echo.
  echo 오류: Python 3.11 이상을 찾을 수 없습니다.
  echo       https://www.python.org/downloads/ 에서 설치할 때
  echo       "Add python.exe to PATH" 를 체크하세요.
  echo.
  pause
  exit /b 1
)

echo [1/3] 가상환경 생성: %VENV%
%PY% -m venv "%VENV%"
if errorlevel 1 goto :failed
goto :venv_done

:venv_ready
echo [1/3] 가상환경 확인: %VENV%
:venv_done

rem ---------------------------------------------------------------- 2. 의존성

if "%SKIP_INSTALL%"=="1" (
  echo [2/3] 의존성 설치 건너뜀 ^(SKIP_INSTALL=1^)
) else (
  echo [2/3] 의존성 설치 중 ^(처음 한 번은 몇 분 걸립니다^)
  "%VENV_PY%" -m pip install --upgrade pip --quiet
  "%VENV_PY%" -m pip install -e "%ROOT%" --quiet
  if errorlevel 1 goto :failed
)

rem ---------------------------------------------------------------- 3. 서버 시작

rem UI 와 API 가 같은 출처라서 CORS 설정은 필요 없다.
set "APARTMENT_ML_UI_DIR=%ROOT%\ui"
if not defined APARTMENT_ML_CONFIG set "APARTMENT_ML_CONFIG=%ROOT%\configs\settings.yaml"

echo [3/3] 모델 파일 확인
call :check_file "models\price_regression.joblib" "가격 예측"
call :check_file "models\complex_directory.json" "단지 목록"
call :check_file "models\anomaly_model.joblib" "이상거래 판정"

echo.
echo       http://localhost:%PORT%
echo       종료: Ctrl+C
echo.

if "%RELOAD%"=="1" (
  "%VENV_PY%" -m uvicorn apartment_ml.api.serve:app --host 127.0.0.1 --port %PORT% --log-level info --reload
) else (
  "%VENV_PY%" -m uvicorn apartment_ml.api.serve:app --host 127.0.0.1 --port %PORT% --log-level info
)
goto :eof

:check_file
if exist "%ROOT%\%~1" (
  echo       %~1 — 있음
) else (
  echo       %~1 — 없음 ^(%~2 기능은 503^)
)
goto :eof

:failed
echo.
echo 설치에 실패했습니다. 위 메시지를 확인하세요.
echo.
pause
exit /b 1
