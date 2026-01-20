# Guía de Instalación: Python 3.12.7 + Unsloth

## Paso 1: Descargar Python 3.12.7

**Opción A - Descarga Directa (RECOMENDADA):**
1. Abrir navegador y ir a: https://www.python.org/downloads/release/python-3127/
2. Scroll hasta "Files"
3. Descargar: **Windows installer (64-bit)** 
   - Nombre del archivo: `python-3.12.7-amd64.exe`
   - Tamaño: ~26 MB

**Opción B - Descarga desde PowerShell:**
```powershell
# Ejecutar en PowerShell como Administrador
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile "$env:TEMP\python-3.12.7-amd64.exe"
```

---

## Paso 2: Instalar Python 3.12.7

1. **Ejecutar el instalador** (`python-3.12.7-amd64.exe`)
2. **¡IMPORTANTE!** Marcar estas opciones:
   - ✅ **"Add python.exe to PATH"** (muy importante)
   - ✅ "Install for all users" (opcional pero recomendado)
3. Click en **"Customize installation"**
4. En "Optional Features", asegurar que esté marcado:
   - ✅ pip
   - ✅ tcl/tk and IDLE
   - ✅ Python test suite
   - ✅ py launcher
5. Click "Next"
6. En "Advanced Options", marcar:
   - ✅ "Install for all users"
   - ✅ "Add Python to environment variables"
   - ✅ "Precompile standard library"
7. Cambiar ruta de instalación a: `C:\Python312` (más fácil de recordar)
8. Click "Install"
9. Esperar ~2 minutos
10. Click "Close"

---

## Paso 3: Verificar Instalación

Abrir **nueva ventana de PowerShell** (importante: nueva para recargar PATH):

```powershell
# Verificar versión
python --version
# Debe mostrar: Python 3.12.7

# Verificar pip
pip --version
# Debe mostrar: pip X.X.X from C:\Python312\...
```

**Si no funciona:**
- Cerrar TODAS las ventanas de PowerShell/CMD
- Abrir nueva ventana
- Intentar de nuevo

---

## Paso 4: Ejecutar Script de Setup Automático

Una vez Python 3.12.7 esté instalado y verificado:

```powershell
cd d:\FineTuning
.\setup_python312_venv.bat
```

Este script:
1. Creará backup del venv actual
2. Creará nuevo venv con Python 3.12.7
3. Instalará PyTorch con CUDA
4. Instalará Unsloth
5. Instalará todas las dependencias
6. Validará que todo funcione

**Tiempo estimado**: 10-15 minutos

---

## Paso 5: Continuar con Fine-Tuning

Una vez el script termine exitosamente:

```powershell
# Activar nuevo venv
.\venv312\Scripts\activate

# Ejecutar entrenamiento
python train_unsloth.py
```

---

## ⚠️ Troubleshooting

### Error: "python is not recognized"
**Solución**: Python no está en PATH
1. Cerrar todas las ventanas PowerShell/CMD
2. Abrir nueva ventana
3. Si persiste, reiniciar el equipo

### Error: Instalar en C:\Python312 requiere permisos
**Solución**: Ejecutar instalador como Administrador
1. Click derecho en `python-3.12.7-amd64.exe`
2. "Ejecutar como administrador"

### Error: pip no funciona
**Solución**: Usar python -m pip en su lugar
```powershell
python -m pip install --upgrade pip
```

---

## 📁 Estructura de Directorios

```
d:\FineTuning\
├── venv\              # Viejo venv (Python 3.14) - se renombrará a venv_backup
├── venv312\           # Nuevo venv (Python 3.12.7) ⭐
├── train_unsloth.py
├── evaluate_model.py
└── ...
```

---

**¡Todo listo para continuar cuando Python 3.12.7 esté instalado!** 🚀
