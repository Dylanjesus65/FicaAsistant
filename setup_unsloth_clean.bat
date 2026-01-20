@echo off
echo ============================================================
echo  INSTALACION LIMPIA: Entorno Unsloth Compatible
echo ============================================================
echo.

REM Crear nuevo venv limpio
echo [1/6] Creando entorno virtual nuevo (venv_unsloth)...
py -3.12 -m venv venv_unsloth
if errorlevel 1 (
    echo ERROR: No se pudo crear venv
    pause
    exit /b 1
)
echo OK - venv_unsloth creado

REM Activar venv
echo.
echo [2/6] Activando venv...
call venv_unsloth\Scripts\activate.bat
echo OK - venv activado

REM Actualizar pip
echo.
echo [3/6] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo OK - pip actualizado

REM Instalar PyTorch 2.4.0 CUDA 12.1
echo.
echo [4/6] Instalando PyTorch 2.4.0 + CUDA 12.1...
echo (Esto puede tomar 3-5 minutos)
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Fallo al instalar PyTorch
    pause
    exit /b 1
)
echo OK - PyTorch instalado

REM Instalar dependencias core compatibles
echo.
echo [5/6] Instalando dependencias compatibles...
python -m pip install transformers==4.46.3 datasets==3.2.0 accelerate==1.2.1 peft==0.14.0 trl==0.12.2 bitsandbytes==0.45.0 --quiet
python -m pip install xformers==0.0.27.post2 protobuf==3.20.3 triton-windows --quiet
python -m pip install evaluate bert-score sacrebleu rouge-score --quiet
python -m pip install matplotlib seaborn pandas numpy==1.26.4 scikit-learn scipy --quiet
python -m pip install pyyaml ftfy tqdm --quiet
echo OK - Dependencias instaladas

REM Instalar Unsloth (versión estable)
echo.
echo [6/6] Instalando Unsloth 2024.12...
python -m pip install unsloth==2024.12.5 unsloth-zoo==2024.12.5 --no-deps
if errorlevel 1 (
    echo ADVERTENCIA: Unsloth 2024.12.5 no encontrado, usando 2024.11.7
    python -m pip install unsloth==2024.11.7 unsloth-zoo==2024.11.7 --no-deps
)
echo OK - Unsloth instalado

REM Validación
echo.
echo ============================================================
echo  VALIDACION
echo ============================================================
echo.
echo Verificando instalación...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"

echo.
echo Probando Unsloth...
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth funciona correctamente!')"

if errorlevel 1 (
    echo.
    echo ❌ ADVERTENCIA: Unsloth tiene problemas
    echo Pero las dependencias core están instaladas
    pause
) else (
    echo.
    echo ============================================================
    echo  ✅ INSTALACION COMPLETADA EXITOSAMENTE
    echo ============================================================
    echo.
    echo Próximo paso:
    echo   1. Activar venv: venv_unsloth\Scripts\activate.bat
    echo   2. Ejecutar: python train_unsloth.py
    echo.
)

pause
