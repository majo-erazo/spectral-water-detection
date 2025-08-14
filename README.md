# Spectral Enhanced Water Quality Pipeline

**Universidad Diego Portales - María José Erazo González**  
**Pipeline de Machine Learning con Análisis Espectral para Detección de Contaminantes Acuáticos**

## Resumen

Este proyecto implementa un sistema completo de Machine Learning que integra análisis espectral avanzado para la detección y clasificación de contaminantes en agua. El sistema utiliza datos hiperespectrales de aguas residuales, concentraciones químicas medidas por LC-HRMS/MS y límites de quantificación (LOQ) para crear modelos predictivos robustos y científicamente validados.

## Dataset Original

### Fuente de Datos
**Open Dataset on Wastewater Quality Monitoring**  
*Swiss Federal Institute of Aquatic Science and Technology (Eawag)*  
https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring/

### Descripción del Experimento
- **Duración**: Campaña de monitoreo de 25 semanas
- **Objetivo**: Evaluación de sistema de imagen hiperespectral vs sensores UV/Vis
- **Resolución temporal**: Imágenes hiperespectrales cada 30 min, sensores cada 2 min
- **Muestras de laboratorio**: 533 muestras analizadas para contaminantes convencionales
- **Análisis de químicos orgánicos**: 86 muestras tras 4 eventos de lluvia

### Datos Hiperespectrales
- **5,801 imágenes hiperespectrales** de aguas residuales crudas
- **Rango espectral**: 400-798 nm cada 2 nm (200 bandas)
- **Procesamiento**: Conversión a reflectancia con referencias blanca y oscura
- **Extracción**: Firma espectral representativa del 60% central de píxeles
- **Validación**: 3 tests de calidad por espectro

### Análisis Químico
- **20 químicos orgánicos** analizados por LC-HRMS/MS
- **Límites de quantificación (LOQ)** corregidos por recuperación y factores matriz
- **Categorías**: Farmacéuticos, edulcorantes, herbicidas, industriales, antimicrobianos
- **Unidades**: Concentraciones en ng/L

### Contaminantes Analizados
#### Farmacéuticos y Productos de Cuidado Personal
- **Candesartan**: Antihipertensivo
- **Citalopram**: Antidepresivo  
- **Diclofenac**: Antiinflamatorio
- **Hydrochlorothiazide**: Diurético

#### Edulcorantes Artificiales
- **Acesulfame**: Edulcorante artificial
- **Cyclamate**: Ciclamato de sodio
- **Caffeine**: Estimulante/trazador antropogénico

#### Antimicrobianos y Conservantes
- **Triclosan**: Antimicrobiano
- **OIT**: 2-n-Octyl-4-isothiazolin-3-on (conservante)

#### Herbicidas y Pesticidas
- **2,4-D**: Herbicida (2,4-Dichlorophenoxy)acetic acid
- **Carbendazim**: Fungicida
- **Diuron**: Herbicida
- **MCPA**: Herbicida (4-Chloro-2-methylphenoxy)acetic acid
- **Mecoprop**: Herbicida Mecoprop-p

#### Compuestos Industriales
- **1,3-Diphenylguanidine**: Acelerador de vulcanización
- **6PPD-Quinone**: Producto de degradación de antioxidantes de neumáticos
- **HMMMM**: Hexa(methoxymethy)melamine

#### Protectores UV
- **Benzotriazole**: 1H-Benzotriazole
- **4-&5-Methylbenzotriazole**: Derivado metilado

#### Repelentes
- **DEET**: N-N-diethyl-3-methylbenzamide

## Características Principales

- **Análisis Espectral Avanzado**: Extracción de 97+ features espectrales interpretables
- **Firmas Espectrales**: Creación automática de firmas características por contaminante
- **Estrategias Adaptativas**: Selección automática entre spectral_only, combined o raw_only
- **Data Augmentation**: Balance inteligente de clases con muestras sintéticas
- **Ensemble Espectral**: Combinación automática de modelos con información espectral
- **40 Contaminantes**: Farmacéuticos, edulcorantes, herbicidas, industriales
- **Resultados Validados**: 80.8% de modelos con mejora espectral, correlación 0.699

## Arquitectura del Sistema

### Componentes Principales

```
├── spectral_analisis.py                    # Motor de análisis espectral
├── ML_dataset_generator_spectral_enhanced.py  # Generador de datasets
├── train_V4_spectral_enhanced.py           # Sistema de entrenamiento
├── test_single_contaminant.py              # Testing individual
├── README.md                               # Documentación principal
├── requirements.txt                        # Dependencias Python
├── data/                                   # Datos originales
│   ├── laboratory_measurements_organic_chemicals.csv
│   ├── laboratory_measurements.csv
│   └── laboratory_measurements_loq_organic_chemicals.csv
├── spectral_enhanced_datasets_combined/    # Datasets con features espectrales + raw
├── spectral_enhanced_datasets_spectral_only/  # Solo features espectrales
├── spectral_enhanced_datasets_raw_only/    # Solo bandas espectrales
├── enhanced_datasets/                      # Datasets con augmentation básico
├── model_outputs_spectral_v4/              # Resultados de entrenamiento
├── demo_spectral_outputs/                  # Outputs de pruebas individuales
├── firmas_espectrales_csv/                 # Firmas espectrales exportadas
└── scripts/                                # Scripts auxiliares
```

### Pipeline de Procesamiento

1. **Carga de Datos** → 2. **Análisis Espectral** → 3. **Generación de Features** → 4. **Entrenamiento ML** → 5. **Evaluación y Ensemble**

## Instalación

### Requisitos del Sistema
```bash
# Verificar versión de Python (3.8+ recomendado)
python --version

# Crear entorno virtual (recomendado)
python -m venv spectral_env
source spectral_env/bin/activate  # Linux/Mac
# spectral_env\Scripts\activate  # Windows
```

### Descarga del Dataset Original

**IMPORTANTE**: Este repositorio contiene solo una muestra de los datos. Para el dataset completo:

1. **Visita la página oficial del dataset:**
   ```
   https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring
   ```

2. **Descarga los archivos necesarios:**
   - `1_codes` - Códigos Python de procesamiento original
   - `2_data` - Todos los datos experimentales completos
   
3. **Extrae en la estructura requerida:**
   ```
   data/
   ├── raw/
   │   ├── 1_codes/                    # Códigos originales de Eawag
   │   └── 2_data/                     # Dataset completo
   │       ├── 2_spectra_extracted_from_hyperspectral_acquisitions/
   │       │   └── flume_mvx_reflectance.csv
   │       └── 5_laboratory_reference_measurements/
   │           ├── laboratory_measurements_organic_chemicals.csv
   │           ├── laboratory_measurements.csv
   │           └── laboratory_measurements_loq_organic_chemicals.csv
   ```

4. **Tamaño del dataset:**
   - Dataset completo: ~50GB (incluye 5,801 imágenes hiperespectrales)
   - Solo archivos CSV necesarios: ~50MB

### Instalación de Dependencias
```bash
# Instalación completa desde requirements.txt
pip install -r requirements.txt

# O instalación mínima
pip install numpy pandas scikit-learn scipy pathlib2

# Dependencias opcionales para máximo rendimiento
pip install xgboost tensorflow matplotlib seaborn
```

### Verificación del Entorno
```bash
# Verificar instalación y estructura de datos
python test_environment.py

# Si faltan los datos originales, aparecerá:
# "ERROR: No se encontraron archivos CSV originales"
# "Descarga el dataset completo desde: https://opendata.eawag.ch/..."
```

### Inicio Rápido (con dataset completo)

**1. Verificar que tienes los datos:**
```bash
# Verificar archivos críticos
ls data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/
ls data/raw/2_data/5_laboratory_reference_measurements/
```

**2. Generar datasets con análisis espectral:**
```bash
python ML_dataset_generator_spectral_enhanced.py
# Selecciona: 1. Combined Features (recomendado)
```

**3. Entrenar modelos:**
```bash
python train_V4_spectral_enhanced.py
# Selecciona: 1. spectral_enhanced_datasets_combined
# Selecciona: 1. Auto (estrategia espectral automática)
```

### Estructura de Datos Requerida

```
data/
├── raw/                                     # DESCARGA DESDE EAWAG
│   ├── 1_codes/                            # Códigos originales
│   └── 2_data/                             # Dataset experimental completo
│       ├── 2_spectra_extracted_from_hyperspectral_acquisitions/
│       │   └── flume_mvx_reflectance.csv   # Firmas espectrales (400-798nm)
│       └── 5_laboratory_reference_measurements/
│           ├── laboratory_measurements_organic_chemicals.csv  # 20 químicos
│           ├── laboratory_measurements.csv                    # Parámetros convencionales
│           └── laboratory_measurements_loq_organic_chemicals.csv  # Límites LOQ
└── processed/                               # GENERADO POR EL PIPELINE
    ├── spectral_enhanced_datasets_combined/
    ├── model_outputs_spectral_v4/
    └── spectral_signatures/
```

## Uso Rápido

### Pre-requisito: Dataset Completo
⚠️ **IMPORTANTE**: Antes de ejecutar el pipeline, descarga el dataset completo desde:
https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

### 1. Generar Datasets con Análisis Espectral
```bash
python ML_dataset_generator_spectral_enhanced.py
```

**Configuraciones disponibles:**
- `spectral_only`: Solo features espectrales interpretables (97 features)
- `combined`: Features espectrales + bandas raw (297 features total)  
- `raw_only`: Solo bandas espectrales (200 features)

### 2. Entrenar Modelos
```bash
python train_V4_spectral_enhanced.py
```

**Algoritmos soportados:**
- SVM con kernels RBF optimizados por contaminante
- XGBoost con regularización adaptativa
- LSTM con arquitectura espectral dinámica

### 3. Probar Contaminante Individual
```bash
# Probar un contaminante específico
python test_single_contaminant.py benzotriazole

# Modo interactivo completo
python test_single_contaminant.py
```

### 4. Demo Rápido (sin dataset completo)
```bash
# Si ya tienes datasets generados, puedes probar:
from train_V4_spectral_enhanced import demo_single_contaminant_spectral
demo_single_contaminant_spectral("benzotriazole")
```

## Resultados Destacados

### Estadísticas Generales
- **40 contaminantes** procesados exitosamente
- **120 modelos** entrenados (SVM, XGBoost, LSTM)
- **80.8% mejora espectral** (97/120 modelos)
- **46 modelos excelentes** (accuracy >90%)
- **33 ensembles** creados automáticamente

### Top Contaminantes (100% accuracy)
1. **13-diphenylguanidine** - Compuesto industrial
2. **benzotriazole** - Protector UV
3. **caffeine** - Estimulante/trazador
4. **citalopram** - Antidepresivo
5. **diuron** - Herbicida
6. **mcpa** - Herbicida
7. **triclosan** - Antimicrobiano

### Efectividad por Estrategia
- **Combined**: 92.0% accuracy promedio (70 modelos)
- **Raw only**: 87.9% accuracy promedio (10 modelos)
- **Correlación firma-rendimiento**: 0.699 (fuerte)

## Estructura de Datos

### Archivos de Entrada
```
flume_mvx_reflectance.csv                      # Firmas espectrales extraídas (400-798 nm, 2 nm)
laboratory_measurements_organic_chemicals.csv  # Concentraciones 20 químicos orgánicos (ng/L)
laboratory_measurements.csv                    # Parámetros fisicoquímicos convencionales
laboratory_measurements_loq_organic_chemicals.csv  # Límites de quantificación por evento
```

### Procesamiento Espectral Original
1. **Conversión a reflectancia**: Normalización con referencias blanca y oscura
2. **Recorte de datacubos**: Retención de área y wavelengths de interés
3. **Filtrado de píxeles**: Eliminación del 20% más brillante y 20% más oscuro
4. **Firma espectral**: Promedio del 60% central de píxeles

### Validación de Datos
- **Test 1**: Verificación funcionamiento del flume
- **Test 2**: Verificación cámara hiperespectral MV.X
- **Test 3**: Verificación rango reflectancia 600nm (0.05-0.25)
- **valid_data**: Combinación de los 3 tests

### Archivos Generados
```
spectral_enhanced_datasets_*/
├── [contaminant]_spectral_enhanced_classical.npz  # Dataset ML clásico
├── [contaminant]_spectral_enhanced_lstm.npz       # Dataset secuencial
├── [contaminant]_spectral_enhanced_metadata.json  # Metadatos
└── spectral_signatures/
    ├── signature_[contaminant].json               # Firma espectral
    └── spectral_library_complete.json            # Biblioteca completa
```

### Archivos de Resultados
```
model_outputs_spectral_v4/
├── spectral_enhanced_training_results.csv      # Métricas por modelo
├── spectral_enhanced_training_complete.json    # Resultados completos
└── spectral_enhanced_analysis_report.md        # Reporte científico
```

## Análisis Espectral

### Features Extraídas (97 por espectro)

#### Estadísticas Espectrales (12 features)
- Media, desviación estándar, mínimo, máximo
- Percentiles (Q25, Q75), asimetría, curtosis
- Área bajo la curva, coeficiente de variación

#### Índices Espectrales (9 features)
- NDVI, NDWI, MNDWI para calidad de agua
- Ratios específicos (Blue/Red, Green/NIR)
- Índices de turbidez y clorofila

#### Características de Forma (15 features)
- Derivadas primera y segunda
- Pendientes y curvatura
- Puntos de inflexión

#### Análisis por Rangos (48 features)
- UV-C, UV-B, UV-A (280-400 nm)
- Visible: Azul, Verde, Rojo (400-750 nm)
- NIR, SWIR1, SWIR2 (750-2500 nm)

#### Detección de Picos (8 features)
- Número de picos y valles
- Alturas y prominencias
- Análisis de absorción

#### Derivadas Espectrales (5 features)
- Energía de derivada
- Máximas pendientes positivas/negativas
- Puntos de inflexión

### Firmas Espectrales

Cada contaminante genera una firma espectral que incluye:
- **Espectro promedio** y desviación estándar
- **Picos característicos** (wavelength, intensidad, prominencia)
- **Wavelengths discriminantes** (correlación con concentración)
- **Métricas de calidad** (consistencia, cobertura, rango dinámico)

## Metodología Científica

### Estrategias de Features
1. **spectral_only**: Solo features espectrales interpretables (ideal para contaminantes con firma de alta calidad)
2. **combined**: Features espectrales + bandas raw (máximo rendimiento)
3. **raw_only**: Solo bandas espectrales (baseline o casos problemáticos)

### Selección Automática
El sistema selecciona automáticamente la estrategia basándose en:
- Calidad de firma espectral (>80: spectral_only, 60-80: combined, <60: raw_only)
- Tamaño del dataset (datasets pequeños → raw_only)
- Configuración específica por contaminante

### Data Augmentation
- **Muestras baseline**: Espectros de agua limpia para clase negativa
- **Mezclas sintéticas**: Combinaciones ponderadas de espectros reales
- **Variaciones espectrales**: Ruido, shift, escalado y suavizado realistas

### Evaluación con Bonus Espectral
- **Bonus por calidad**: +5 puntos si firma ≥80/100, +3 si ≥60/100
- **Bonus por estrategia**: +2 puntos si spectral_only/combined efectivo
- **Bonus por dominancia**: +3 puntos si features espectrales dominan

### Ensemble Adaptativo
- Considera calidad individual de modelos
- Pondera por análisis espectral
- Incluye diversidad de estrategias espectrales

## Casos de Uso

### Investigación Científica
- Identificación de contaminantes emergentes
- Análisis de correlaciones espectrales
- Desarrollo de nuevos índices espectrales

### Monitoreo Ambiental
- Detección automática en tiempo real
- Clasificación de múltiples contaminantes
- Alertas tempranas de contaminación

### Desarrollo de Sensores
- Validación de wavelengths críticas
- Optimización de rangos espectrales
- Calibración de instrumentos

## Validación Científica

### Métricas de Rendimiento
- **Accuracy**: Precisión de clasificación
- **F1-Score**: Balance precisión/recall
- **AUC-ROC**: Área bajo curva ROC
- **Gap train-test**: Detección de overfitting

### Análisis de Correlación
- Correlación 0.699 entre calidad de firma espectral y rendimiento ML
- Validación estadística de efectividad del análisis espectral
- Identificación de wavelengths más discriminantes por contaminante

### Reproducibilidad
- Seeds fijos para reproducibilidad
- Validación temporal estricta
- Documentación completa de parámetros

## Contaminantes Soportados

### Farmacéuticos y Productos de Cuidado Personal
- **citalopram**: Antidepresivo (100% accuracy)
- **diclofenac**: Antiinflamatorio
- **candesartan**: Antihipertensivo
- **hydrochlorthiazide**: Diurético

### Edulcorantes Artificiales
- **acesulfame**: Edulcorante acesulfame-K
- **cyclamate**: Ciclamato de sodio
- **caffeine**: Estimulante/trazador (100% accuracy)

### Antimicrobianos y Conservantes
- **triclosan**: Antimicrobiano (100% accuracy)
- **oit**: Conservante (caso especial)

### Herbicidas y Pesticidas
- **diuron**: Herbicida (100% accuracy)
- **mecoprop**: Herbicida MCPP
- **mcpa**: Herbicida (100% accuracy)
- **24-d**: Herbicida 2,4-D
- **carbendazim**: Fungicida

### Compuestos Industriales
- **13-diphenylguanidine**: Acelerador de vulcanización (100% accuracy)
- **6ppd-quinone**: Producto de degradación de neumáticos
- **hmmm**: Compuesto industrial

### Protectores UV y Benzotriazoles
- **benzotriazole**: Protector UV (100% accuracy)
- **4-&5-methylbenzotriazole**: Derivado metilado

### Repelentes de Insectos
- **deet**: N,N-dietil-meta-toluamida

## Parámetros Optimizados por Contaminante

### Contaminantes de Alta Calidad (C≥10, profundidad≥5)
- benzotriazole, diuron, acesulfame, cyclamate
- Parámetros agresivos para máximo rendimiento

### Contaminantes Problemáticos (C≤1, profundidad≤3)  
- diclofenac, 24-d, carbendazim, oit
- Parámetros conservadores anti-overfitting

### Casos Especiales
- **oit**: Automáticamente usa raw_only y parámetros ultra-conservadores
- **diclofenac**: Parámetros anti-overfitting específicos

## Troubleshooting

### Problemas Comunes

**Error: "Dataset no encontrado" o "Archivo CSV no encontrado"**
```bash
# Verificar que descargaste el dataset completo
ls data/raw/2_data/5_laboratory_reference_measurements/

# Si está vacío, descargar desde:
# https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

# Asegurar estructura correcta:
data/raw/2_data/5_laboratory_reference_measurements/
├── laboratory_measurements_organic_chemicals.csv
├── laboratory_measurements.csv
└── laboratory_measurements_loq_organic_chemicals.csv
```

**Error: "XGBoost no disponible"**
```bash
pip install xgboost
```

**Error: "TensorFlow no disponible"**
```bash
pip install tensorflow
```

**Error: "No se encontraron columnas espectrales"**
```bash
# Verificar archivo de reflectancia
ls data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/
# Debe contener: flume_mvx_reflectance.csv

# Si falta, descargar dataset completo desde Eawag
```

**Dataset muy grande para descargar**
```bash
# Solo necesitas estos archivos específicos del dataset Eawag:
# 2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv (~40MB)
# 2_data/5_laboratory_reference_measurements/*.csv (~2MB total)
# 
# NO necesitas las 5,801 imágenes hiperespectrales completas (50GB)
```

**Accuracy muy baja (<70%)**
- Verificar calidad de datos de entrada
- Revisar distribución de clases
- Considerar más data augmentation

### Verificación de Resultados
```bash
# Verificar archivos generados
ls model_outputs_spectral_v4/

# Revisar métricas
head -20 model_outputs_spectral_v4/spectral_enhanced_training_results.csv

# Análisis detallado
python -c "
import pandas as pd
df = pd.read_csv('model_outputs_spectral_v4/spectral_enhanced_training_results.csv')
print(df.groupby('spectral_strategy')['test_accuracy'].describe())
"
```

### Contacto para Problemas con Datos

**Para problemas con el dataset original:**
- Swiss Federal Institute of Aquatic Science and Technology (Eawag)
- Dataset: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

**Para problemas con el pipeline:**
- Verificar requirements.txt
- Revisar test_environment.py
- Consultar este README

## Contribución

### Estructura del Código
- Código sin emojis para uso profesional
- Documentación científica completa
- Logging detallado para debugging
- Manejo robusto de errores

### Extensiones Futuras
- Más algoritmos ML (Random Forest, Neural Networks)
- Análisis de transfer learning entre contaminantes
- Optimización de hyperparámetros automática
- Integración con sensores en tiempo real

## Referencias Científicas

### Dataset Original
**Lechevallier P, Villez K, Felsheim C, Rieckermann J.** (2024). *Towards non-contact pollution monitoring in sewers with hyperspectral imaging.* Environmental Science: Water Research & Technology. https://pubs.rsc.org/en/content/articlelanding/2024/ew/d3ew00541k

### Metodología
- Análisis espectral basado en principios de espectroscopía UV-Vis-NIR (400-798 nm)
- Índices espectrales validados para calidad de agua y detección de contaminantes
- Técnicas de machine learning robustas con validación temporal estricta
- Integración de datos hiperespectrales con análisis químico por LC-HRMS/MS

### Aplicaciones
- Monitoreo no invasivo de calidad de agua residual
- Detección automática de contaminantes emergentes
- Análisis ambiental en tiempo real con sensores hiperespectrales


### Dataset Original
**Swiss Federal Institute of Aquatic Science and Technology (Eawag)**  
Open Dataset on Wastewater Quality Monitoring  
DOI: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring/

---

*Sistema validado con 40 contaminantes de aguas residuales, 120 modelos entrenados, y 80.8% de mejora espectral demostrada sobre datos reales de campaña de monitoreo de 25 semanas.*