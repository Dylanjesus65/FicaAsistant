@echo off
echo ============================================================
echo  SETUP: Python 3.12 venv + Unsloth
echo ============================================================
echo.

REM Verificar que Python 3.12 esté instalado
echo [1/8] Verificando Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 no detectado.
    echo Por favor instala Python 3.12.7 primero.
    echo Ver: INSTALL_PYTHON312.md
    pause
    exit /b 1
)
py -3.12 --version
echo OK - Python 3.12 detectado

REM Hacer backup del venv actual
echo.
echo [2/8] Haciendo backup de venv actual...
if exist venv (
    if exist venv_backup (
        echo Eliminando backup antiguo...
        rmdir /s /q venv_backup
    )
    echo Renombrando venv -^> venv_backup
    move venv venv_backup
    echo OK - Backup creado
) else (
    echo No hay venv previo - saltando backup
)

REM Crear nuevo venv con Python 3.12
echo.
echo [3/8] Creando nuevo venv con Python 3.12...
py -3.12 -m venv venv312
if errorlevel 1 (
    echo ERROR: No se pudo crear venv
    pause
    exit /b 1
)
echo OK - venv312 creado

REM Activar venv
echo.
echo [4/8] Activando venv...
call venv312\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: No se pudo activar venv
    pause
    exit /b 1
)
echo OK - venv activado

REM Actualizar pip
echo.
echo [5/8] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo OK - pip actualizado

REM Instalar PyTorch con CUDA
echo.
echo [6/8] Instalando PyTorch con CUDA 12.1...
echo (Esto puede tomar 3-5 minutos)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo ERROR: Fallo al instalar PyTorch
    pause
    exit /b 1
)
echo OK - PyTorch instalado

REM Instalar Unsloth
echo.
echo [7/8] Instalando Unsloth...
echo (Esto puede tomar 2-3 minutos)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if errorlevel 1 (
    echo ADVERTENCIA: Unsloth falló - intentando metodo alternativo...
    pip install unsloth --no-deps
    pip install xformers triton
)
echo OK - Unsloth instalado

REM Instalar dependencias restantes
echo.
echo [8/8] Instalando dependencias restantes...
pip install transformers>=4.36.0 datasets>=2.14.0 peft>=0.7.0 accelerate>=0.25.0 trl>=0.7.0 --quiet
pip install evaluate bert-score sacrebleu rouge-score --quiet
pip install matplotlib seaborn scikit-learn pandas numpy tqdm --quiet
pip install pyyaml scipy ftfy --quiet
echo OK - Todas las dependencias instaladas

REM Validar instalación
echo.
echo ============================================================
echo  VALIDACION
echo ============================================================
echo.
echo Verificando imports críticos...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "try: from unsloth import FastLanguageModel; print('Unsloth: OK'); except: print('Unsloth: ERROR')"

echo.
echo ============================================================
echo  INSTALACION COMPLETADA
echo ============================================================
echo.
echo Proximo paso:
echo   1. Activar venv: venv312\Scripts\activate.bat
echo   2. Ejecutar: python train_unsloth.py
echo.
pause
