# Setup Script para ObesityMine - Cookiecutter Data Science

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  ObesityMine Setup - CCDS Compliant Structure  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si conda está disponible
Write-Host "📦 Verificando entorno Conda..." -ForegroundColor Yellow

try {
    $condaPath = conda info --base 2>$null
    if ($condaPath) {
        Write-Host "✅ Conda detectado en: $condaPath" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Conda no detectado. Por favor, usa Anaconda Prompt." -ForegroundColor Red
    Write-Host ""
    Write-Host "Instrucciones:" -ForegroundColor Yellow
    Write-Host "1. Abre Anaconda Prompt" -ForegroundColor White
    Write-Host "2. Navega a este directorio:" -ForegroundColor White
    Write-Host "   cd 'd:\OneDrive\Escritorio\Maestria IA\Trimestre 4\MLOps\Git_Local\ObesityMine53'" -ForegroundColor White
    Write-Host "3. Ejecuta: .\setup_project.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "🔧 Instalando paquete en modo desarrollo..." -ForegroundColor Yellow

# Activar entorno
conda activate obesitymine

# Instalar paquete
pip install -e .

Write-Host ""
Write-Host "✅ Paquete instalado correctamente!" -ForegroundColor Green
Write-Host ""

# Verificar instalación
Write-Host "🧪 Verificando instalación..." -ForegroundColor Yellow
python -c "import src; print(f'✅ Paquete src instalado - Versión: {src.__version__}')" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Verificación exitosa!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Verifica la instalación manualmente con:" -ForegroundColor Yellow
    Write-Host "   python -c 'import src; print(src.__version__)'" -ForegroundColor White
}

Write-Host ""
Write-Host "📋 Comandos disponibles:" -ForegroundColor Cyan
Write-Host "   make help              - Ver todos los comandos" -ForegroundColor White
Write-Host "   make test_environment  - Probar configuración" -ForegroundColor White
Write-Host "   make sync_data_down    - Descargar datos de DVC" -ForegroundColor White
Write-Host "   make lint              - Revisar estilo de código" -ForegroundColor White
Write-Host "   make format            - Formatear código" -ForegroundColor White
Write-Host ""

Write-Host "⚠️  RECORDATORIO:" -ForegroundColor Yellow
Write-Host "   1. Renombrar notebooks según CCDS: PHASE.NUMBER-initials-description.ipynb" -ForegroundColor White
Write-Host "   2. Mover CSVs a data/processed/" -ForegroundColor White
Write-Host "   3. Ver detalles completos en CCDS_AUDIT.md" -ForegroundColor White
Write-Host ""

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  ✅ Setup completado - ¡Listo para trabajar!   " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
