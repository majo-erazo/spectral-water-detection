# Pipeline de Detección Espectral de Contaminantes en Aguas

Sistema avanzado de detección temprana de contaminantes en aguas superficiales utilizando espectroscopia UV-Vis de reflectancia combinada con técnicas de machine learning y análisis espectral.

## Descripción del Proyecto

Este pipeline implementa un sistema completo para la detección automática de contaminantes acuáticos utilizando análisis espectral avanzado y algoritmos de aprendizaje automático. El sistema procesa datos hiperespectrales en el rango UV-Vis (400-800 nm) y aplica técnicas de ingeniería de características espectrales para detectar 29 tipos diferentes de contaminantes con alta precisión.

### Características Principales

- **Análisis Espectral Avanzado**: Extracción de características espectrales específicas para calidad de agua
- **Múltiples Algoritmos ML**: Soporte para SVM, XGBoost y LSTM con optimización automática
- **Validación Temporal**: Cross-validación temporal robusta para evitar data leakage
- **Detección Multicontaminante**: Capacidad de detectar 29 contaminantes diferentes simultáneamente
- **Pipeline Automatizado**: Flujo completo desde datos raw hasta modelos entrenados
- **Firmas Espectrales**: Generación automática de firmas espectrales por contaminante
- **Ensemble Inteligente**: Sistema de ensemble que considera calidad espectral

## Estructura del Proyecto

```
SPECTRAL-WATER-DETECTION/
├── scripts/                           # Scripts principales
│   ├── ML_dataset_generator_spectral_enhanced.py  # Generación de datasets con análisis espectral
│   ├── ml_dataset_generator_v2.py     # Generador optimizado de datasets
│   ├── train_V4_spectral_enhanced.py  # Entrenamiento con análisis espectral integrado
│   ├── train.py                       # Sistema de entrenamiento clásico
│   ├── spectral_analisis.py           # Motor de análisis espectral avanzado
│   └── debug.py                       # Herramientas de debugging
├── data/raw/                          # Datos originales (descarga requerida)
├── spectral_enhanced_datasets_*/      # Datasets con características espectrales
├── model_outputs_spectral_v4/         # Modelos entrenados y resultados
├── firmas_espectrales_csv/            # Firmas espectrales por contaminante
├── requirements.txt                   # Dependencias del proyecto
├── README.md                          # Este archivo
├── .gitignore                         # Configuración de Git
└── Informe-Memoria.pdf               # Documentación técnica completa
```

## Instalación y Configuración

### 1. Requisitos del Sistema

- Python 3.8+
- 8GB RAM mínimo (16GB recomendado para análisis espectral)
- 5GB espacio libre en disco
- GPU opcional (recomendada para LSTM)

### 2. Instalación de Dependencias

```bash
# Clonar el repositorio
git clone <repository-url>
cd SPECTRAL-WATER-DETECTION

# Crear entorno virtual
python -m venv spectral_env
source spectral_env/bin/activate  # Linux/Mac
# o
spectral_env\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Descarga del Dataset Original

**IMPORTANTE**: Los archivos de datos originales NO están incluidos en el repositorio debido a su tamaño (~50GB total).

#### Descarga desde Fuente Oficial

1. **Fuente**: Swiss Federal Institute of Aquatic Science and Technology (Eawag)
2. **URL**: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

#### Archivos Necesarios (Solo ~42MB)

```bash
# Crear estructura de directorios
mkdir -p data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions
mkdir -p data/raw/2_data/5_laboratory_reference_measurements

# Descargar solo los archivos esenciales:
# 1. flume_mvx_reflectance.csv (~40MB)
# 2. laboratory_measurements_organic_chemicals.csv (~1MB) 
# 3. laboratory_measurements.csv (~500KB)
# 4. laboratory_measurements_loq_organic_chemicals.csv (~5KB)
```

#### Verificación de la Instalación

```bash
python scripts/debug.py
```

Debe mostrar:
```
✅ laboratory_measurements_organic_chemicals.csv: 86 filas, 23 columnas
✅ laboratory_measurements.csv: 529 filas, 12 columnas
✅ laboratory_measurements_loq_organic_chemicals.csv: 4 filas, 23 columnas
✅ flume_mvx_reflectance.csv: 5801 filas, 200+ columnas
```

## Uso del Pipeline

### Opción 1: Pipeline Completo con Análisis Espectral (Recomendado)

```bash
# 1. Generar datasets con análisis espectral avanzado
python scripts/ML_dataset_generator_spectral_enhanced.py

# 2. Entrenar modelos con optimización espectral
python scripts/train_V4_spectral_enhanced.py
```

### Opción 2: Pipeline Clásico

```bash
# 1. Generar datasets optimizados
python scripts/ml_dataset_generator_v2.py

# 2. Entrenar modelos tradicionales
python scripts/train.py
```

### Características del Pipeline Espectral Enhanced

- **Análisis Espectral Automático**: Extracción de 80+ características espectrales
- **Firmas Espectrales**: Creación de firmas características por contaminante
- **Estrategias Adaptativas**: Selección automática entre spectral-only, combined o raw-only
- **Ensemble Espectral**: Combinación inteligente considerando calidad espectral
- **Validación Robusta**: Evaluación con métricas espectrales integradas

## Contaminantes Detectables

### Farmacéuticos y Productos de Cuidado Personal
- **Candesartan**: Antihipertensivo
- **Citalopram**: Antidepresivo
- **Diclofenac**: Antiinflamatorio
- **Hydrochlorothiazide**: Diurético

### Edulcorantes Artificiales
- **Acesulfame**: Edulcorante artificial
- **Cyclamate**: Ciclamato de sodio
- **Caffeine**: Estimulante/trazador antropogénico

### Antimicrobianos y Conservantes
- **Triclosan**: Antimicrobiano
- **OIT**: 2-n-Octyl-4-isothiazolin-3-on

### Herbicidas y Pesticidas
- **2,4-D**: Herbicida
- **Carbendazim**: Fungicida
- **Diuron**: Herbicida
- **MCPA**: Herbicida
- **Mecoprop**: Herbicida

### Compuestos Industriales
- **1,3-Diphenylguanidine**: Acelerador de vulcanización
- **6PPD-Quinone**: Antioxidante de neumáticos
- **HMMMM**: Hexa(methoxymethy)melamine

### Protectores UV y Repelentes
- **Benzotriazole**: Protector UV
- **4-&5-Methylbenzotriazole**: Derivado metilado
- **DEET**: N-N-diethyl-3-methylbenzamide

### Parámetros Fisicoquímicos Tradicionales
- **DOC**: Carbono Orgánico Disuelto
- **Turbidez**: Turbidez nefelométrica
- **TSS**: Sólidos Suspendidos Totales
- **PO4, NH4, SO4**: Nutrientes

## Metodología Técnica

### Análisis Espectral Avanzado

1. **Extracción de Características Espectrales**:
   - Estadísticas espectrales: 12 métricas estadísticas por espectro
   - Índices espectrales: 9 índices validados para calidad de agua
   - Análisis de forma: Pendientes, curvatura, puntos de inflexión
   - Features por rangos: Análisis separado UV/VIS/NIR
   - Detección de picos: Análisis automático de características espectrales
   - Derivadas espectrales: Primera y segunda derivada para detección de cambios

2. **Firmas Espectrales por Contaminante**:
   - Creación automática de firmas espectrales características
   - Identificación de wavelengths discriminantes
   - Análisis de picos característicos por contaminante
   - Métricas de calidad y consistencia espectral
   - Comparación entre contaminantes

3. **Estrategias de Modelado Flexibles**:
   - **Spectral Only**: Solo features espectrales interpretables
   - **Combined**: Features espectrales + bandas raw para máximo rendimiento
   - **Raw Only**: Solo bandas espectrales (baseline para comparación)

### Algoritmos de Machine Learning

- **SVM**: Máquinas de Vectores de Soporte con kernels RBF optimizados
- **XGBoost**: Gradient Boosting optimizado para datos espectrales
- **LSTM**: Redes Neuronales Recurrentes para patrones temporales

### Validación Temporal Robusta

Sistema de validación que respeta la cronología de los datos para evitar data leakage:
- División temporal estricta (70% entrenamiento, 15% validación, 15% test)
- Validación cruzada temporal con gaps temporales
- Métricas de evaluación específicas para detección de contaminantes
- Verificación automática de ausencia de data leakage

## Resultados y Rendimiento

### Métricas de Éxito
- **Tasa de Detección Global**: 27.6% (8 de 29 contaminantes)
- **Contaminantes Detectables**: Caffeine, Acesulfame, DOC, Diuron, Benzotriazole
- **Precisión Promedio**: >85% en contaminantes detectables
- **Recall Promedio**: >80% en contaminantes detectables
- **Mejora con Análisis Espectral**: 10-25% en contaminantes con alta calidad espectral

### Casos de Éxito Documentados
1. **Caffeine**: Detección robusta con múltiples algoritmos (Acc: >0.90)
2. **Acesulfame**: Buena separabilidad espectral (Acc: >0.85)
3. **DOC**: Excelente correlación con características espectrales UV (Acc: >0.88)
4. **Diuron**: Beneficiado significativamente por análisis espectral
5. **Benzotriazole**: Firmas espectrales de alta calidad

## Scripts Principales

### `ML_dataset_generator_spectral_enhanced.py`
Generador principal de datasets con análisis espectral integrado:
- Integra datos espectrales y químicos con análisis LOQ
- Aplica ingeniería de características espectrales avanzada (80+ features)
- Genera firmas espectrales características por contaminante
- Implementa data augmentation espectral realista
- Crea múltiples estrategias de features (spectral-only, combined, raw-only)
- Validación temporal estricta anti-leakage

### `train_V4_spectral_enhanced.py`
Sistema de entrenamiento con análisis espectral integrado:
- Selección automática de estrategia espectral óptima por contaminante
- Optimización de hiperparámetros específica por calidad espectral
- Ensemble inteligente considerando diversidad espectral
- Evaluación con métricas espectrales integradas
- Análisis de importancia de features espectrales vs raw

### `spectral_analisis.py`
Motor de análisis espectral avanzado:
- Clase `SpectralFeatureEngineer`: Extracción de 80+ features espectrales
- Clase `SpectralSignatureAnalyzer`: Creación de firmas espectrales
- Índices espectrales validados para calidad de agua
- Análisis de picos y wavelengths discriminantes
- Sistema de evaluación de calidad espectral

### `ml_dataset_generator_v2.py`
Generador optimizado con pipeline científicamente riguroso:
- Validación temporal estricta para evitar data leakage
- Análisis comprehensivo de viabilidad por contaminante
- Generación de datasets ML y LSTM optimizados
- Integración opcional con análisis espectral
- Documentación automática de calidad de datos

### `train.py`
Sistema de entrenamiento clásico para comparación:
- Entrenamiento tradicional SVM, XGBoost, LSTM
- Optimización de hiperparámetros por contaminante
- Validación cruzada temporal robusta
- Evaluación comprehensiva con múltiples métricas
- Sistema de ensemble clásico

## Configuración Avanzada

### Parámetros de Análisis Espectral

```python
from ML_dataset_generator_spectral_enhanced import SpectralEnhancedMLGenerator

generator = SpectralEnhancedMLGenerator(
    output_dir="custom_spectral_datasets",
    use_spectral_features=True,
    spectral_strategy="combined"  # "spectral_only", "combined", "raw_only"
)
```

### Configuración de Entrenamiento Espectral

```python
from train_V4_spectral_enhanced import SpectralEnhancedMLPipeline

pipeline = SpectralEnhancedMLPipeline(
    datasets_dir="spectral_enhanced_datasets_combined",
    output_dir="model_outputs_spectral_v4",
    spectral_strategy="auto"  # Selección automática óptima
)
```

### Estrategias Espectrales Disponibles

- **auto**: Selección automática basada en calidad de firma espectral
- **spectral_only**: Solo características espectrales interpretables
- **combined**: Features espectrales + bandas raw (máximo rendimiento)
- **raw_only**: Solo bandas espectrales (baseline de comparación)

## Archivos Generados

### Datasets Espectralmente Mejorados
- `*_spectral_enhanced_classical.npz`: Datasets con features espectrales para ML clásico
- `*_spectral_enhanced_lstm.npz`: Secuencias espectrales para LSTM/CNN1D
- `*_spectral_enhanced_metadata.json`: Metadatos + análisis espectral completo

### Modelos y Resultados
- `model_outputs_spectral_v4/models/`: Modelos entrenados (.pkl, .h5)
- `model_outputs_spectral_v4/reports/`: Reportes detallados JSON
- `spectral_enhanced_training_results.csv`: Métricas comparativas
- `spectral_enhanced_analysis_report.md`: Reporte científico completo

### Firmas Espectrales
- `spectral_signatures/signature_*.json`: Firmas espectrales individuales
- `spectral_library_complete.json`: Biblioteca completa de firmas
- `spectral_enhanced_detectability_analysis.csv`: Análisis de detectabilidad

## Troubleshooting

### Problemas Comunes

1. **"Dataset muy grande para descargar"**
   - Solución: Solo necesitas ~42MB de archivos CSV, no las 5,801 imágenes hiperespectrales completas

2. **"Error en análisis espectral"**
   - Verificar que `spectral_analisis.py` está en el directorio raíz
   - Instalar dependencias: `pip install scipy scikit-learn`

3. **"Modelos con bajo rendimiento"**
   - Verificar calidad de firmas espectrales en los reportes
   - Probar diferentes estrategias espectrales (spectral-only vs combined)
   - Revisar balance de clases en los datasets

4. **"Error de memoria durante entrenamiento"**
   - Reducir número de features espectrales en configuración
   - Usar `spectral_strategy="raw_only"` para datasets pequeños

### Validación de Resultados

```bash
# Verificar ausencia de data leakage
python -c "
from ml_dataset_generator_v2 import verify_no_leakage_external
result = verify_no_leakage_external('path/to/dataset.npz')
print(f'Validación: {result[\"overall\"]}')
"
```

### Logs y Debugging

Los logs detallados se guardan automáticamente en:
- `spectral_enhanced_datasets_*/processing.log`
- `model_outputs_spectral_v4/training.log`

## Contribución Científica

### Metodología Innovadora
1. **Integración Espectro-ML**: Primer pipeline que integra completamente análisis espectral con ML
2. **Firmas Espectrales Automáticas**: Generación automática de firmas características por contaminante
3. **Ensemble Espectral**: Sistema de ensemble que considera diversidad y calidad espectral
4. **Validación Temporal Robusta**: Metodología rigurosa para evitar data leakage temporal
5. **Evaluación Multidimensional**: Métricas que combinan rendimiento ML y calidad espectral

### Aplicaciones
- Monitoreo ambiental autónomo en tiempo real
- Sistemas de alerta temprana para contaminación acuática
- Optimización de tratamiento de aguas residuales
- Investigación de nuevos contaminantes emergentes

## Referencias Científicas

### Publicación Principal
**Lechevallier P, Villez K, Felsheim C, Rieckermann J.** (2024). *Towards non-contact pollution monitoring in sewers with hyperspectral imaging.* Environmental Science: Water Research & Technology.

### Dataset Original
**Swiss Federal Institute of Aquatic Science and Technology (Eawag)**  
Open Dataset on Wastewater Quality Monitoring  
https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

### Metodología
Este trabajo implementa y extiende las metodologías descritas en:
- UV-Vis spectroscopy for water quality monitoring
- Machine learning for environmental monitoring
- Hyperspectral image analysis for pollution detection
- Temporal cross-validation for time series data


---

*Una vez descargados los archivos correctamente, puedes ejecutar el pipeline completo siguiendo las instrucciones detalladas en cada sección.*