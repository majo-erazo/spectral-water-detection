# Dataset Original - Datos Raw

****  
**Pipeline de Machine Learning con Análisis Espectral para Detección de Contaminantes Acuáticos**

## ⚠️ IMPORTANTE: Directorio Vacío Intencionalmente

Este directorio está **vacío** en el repositorio Git porque los archivos de datos originales son demasiado grandes para ser incluidos (>50GB total). Los usuarios deben **descargar el dataset completo** desde la fuente oficial.

## 📥 Descarga del Dataset Original

### Fuente Oficial
**Open Dataset on Wastewater Quality Monitoring**  
*Swiss Federal Institute of Aquatic Science and Technology (Eawag)*

🔗 **URL de descarga**: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

### Archivos Requeridos

Descarga estos dos archivos principales del sitio de Eawag:

1. **`1_codes.zip`** - Códigos Python originales de procesamiento
2. **`2_data.zip`** - Dataset experimental completo

## 📂 Estructura Requerida Después de la Descarga

```
data/raw/
├── 1_codes/                                    # Códigos originales de Eawag
│   ├── main_script.py
│   ├── processing_functions.py
│   └── README.md
└── 2_data/                                     # Dataset experimental
    ├── 1_raw_mvx_hyperspectral_datacubes/     # Imágenes hiperespectrales (OPCIONAL)
    ├── 2_spectra_extracted_from_hyperspectral_acquisitions/
    │   └── flume_mvx_reflectance.csv           # ⭐ CRÍTICO
    ├── 3_sensors_raw_data/
    ├── 4_sensors_data_interpolated_and_aggregated/
    └── 5_laboratory_reference_measurements/    # ⭐ CRÍTICO
        ├── laboratory_measurements_organic_chemicals.csv
        ├── laboratory_measurements.csv
        └── laboratory_measurements_loq_organic_chemicals.csv
```

## 📊 Archivos Críticos para el Pipeline

### Solo Necesitas Estos Archivos (no todo el dataset):

#### 1. Firmas Espectrales
**Archivo**: `2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv`
- **Tamaño**: ~40MB
- **Contenido**: 5,801 firmas espectrales extraídas
- **Rango**: 400-798 nm cada 2 nm (200 bandas)
- **Formato**: CSV con timestamp_iso + reflectancia por wavelength

#### 2. Concentraciones de Químicos Orgánicos
**Archivo**: `2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv`
- **Tamaño**: ~1MB
- **Contenido**: Concentraciones de 20 químicos orgánicos
- **Método**: Análisis LC-HRMS/MS
- **Unidades**: ng/L

#### 3. Parámetros Fisicoquímicos
**Archivo**: `2_data/5_laboratory_reference_measurements/laboratory_measurements.csv`
- **Tamaño**: ~500KB
- **Contenido**: Parámetros convencionales (turbidez, DOC, etc.)

#### 4. Límites de Quantificación
**Archivo**: `2_data/5_laboratory_reference_measurements/laboratory_measurements_loq_organic_chemicals.csv`
- **Tamaño**: ~5KB
- **Contenido**: LOQ corregidos por matriz por evento de lluvia

## 💾 Tamaños de Archivos

| Componente | Tamaño | Necesario |
|------------|---------|-----------|
| **Archivos críticos** | ~42MB | ✅ SÍ |
| Imágenes hiperespectrales | ~50GB | ❌ NO |
| Códigos originales | ~10MB | ⚠️ Opcional |
| Metadatos | ~100KB | ⚠️ Opcional |
| **TOTAL MÍNIMO** | **~42MB** | ✅ **SÍ** |

## 🚀 Instrucciones de Descarga Paso a Paso

### Opción 1: Descarga Solo Archivos Necesarios
1. Ir a: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring
2. Descargar `2_data.zip` (contiene todos los CSVs necesarios)
3. Extraer solo las carpetas necesarias:
   ```bash
   unzip 2_data.zip
   mkdir -p data/raw/2_data
   cp -r 2_data/2_spectra_extracted_from_hyperspectral_acquisitions data/raw/2_data/
   cp -r 2_data/5_laboratory_reference_measurements data/raw/2_data/
   ```

### Opción 2: Descarga Completa
```bash
# 1. Descargar ambos archivos desde Eawag
wget https://opendata.eawag.ch/.../1_codes.zip
wget https://opendata.eawag.ch/.../2_data.zip

# 2. Extraer en la estructura correcta
unzip 1_codes.zip -d data/raw/
unzip 2_data.zip -d data/raw/
```

### Opción 3: Solo Archivos CSV (Más Rápido)
Si solo quieres los archivos esenciales, puedes descargar únicamente:
- `flume_mvx_reflectance.csv`
- `laboratory_measurements_organic_chemicals.csv`
- `laboratory_measurements.csv` 
- `laboratory_measurements_loq_organic_chemicals.csv`

Y colocarlos en la estructura correcta manualmente.

## ✅ Verificación de Descarga

Una vez descargado, verifica que tienes los archivos correctos:

```bash
# Ejecutar desde la raíz del proyecto
python test_environment.py

# Debe mostrar:
# ✅ laboratory_measurements_organic_chemicals.csv: X filas, Y columnas
# ✅ laboratory_measurements.csv: X filas, Y columnas  
# ✅ laboratory_measurements_loq_organic_chemicals.csv: X filas, Y columnas
# ✅ flume_mvx_reflectance.csv: 5801 filas, 203+ columnas
```

### Verificación Manual
```bash
# Verificar archivos críticos
ls -la data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/
ls -la data/raw/2_data/5_laboratory_reference_measurements/

# Verificar tamaños aproximados
du -h data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv
# Debe mostrar ~40MB

du -h data/raw/2_data/5_laboratory_reference_measurements/*.csv
# Debe mostrar ~1-2MB total
```

## 📋 Información del Dataset Original

### Descripción del Experimento
- **Duración**: 25 semanas de monitoreo continuo
- **Ubicación**: Sistema de flume de laboratorio
- **Método**: Imagen hiperespectral + análisis químico
- **Frecuencia**: Imágenes cada 30 min, sensores cada 2 min
- **Eventos**: 4 eventos de lluvia analizados

### Datos Hiperespectrales
- **Total imágenes**: 5,801 imágenes hiperespectrales
- **Procesamiento**: Conversión a reflectancia con referencias
- **Filtrado**: 60% central de píxeles (elimina 20% más brillante y 20% más oscuro)
- **Resultado**: Una firma espectral representativa por timestamp

### Análisis Químico
- **Método**: LC-HRMS/MS (cromatografía líquida - espectrometría de masas)
- **Químicos**: 20 contaminantes orgánicos emergentes
- **Categorías**: Farmacéuticos, edulcorantes, herbicidas, industriales
- **LOQ**: Límites corregidos por recuperación y factores matriz

## 🔬 Contaminantes Incluidos en el Dataset

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

### Protectores UV
- **Benzotriazole**: Protector UV
- **4-&5-Methylbenzotriazole**: Derivado metilado

### Repelentes
- **DEET**: N-N-diethyl-3-methylbenzamide

## 📖 Referencias del Dataset

### Publicación Original
**Lechevallier P, Villez K, Felsheim C, Rieckermann J.** (2024). *Towards non-contact pollution monitoring in sewers with hyperspectral imaging.* Environmental Science: Water Research & Technology.

**DOI**: https://pubs.rsc.org/en/content/articlelanding/2024/ew/d3ew00541k

### Dataset
**Swiss Federal Institute of Aquatic Science and Technology (Eawag)**  
Open Dataset on Wastewater Quality Monitoring  
**URL**: https://opendata.eawag.ch/dataset/open-dataset-on-wastewater-quality-monitoring

## ❓ Troubleshooting

### Problema: "Dataset muy grande para descargar"
**Solución**: Solo necesitas ~42MB de archivos CSV, no las 5,801 imágenes hiperespectrales completas.

### Problema: "Archivos no encontrados"
**Solución**: Verificar estructura de carpetas y nombres exactos de archivos.

### Problema: "Error al extraer archivos"
**Solución**: Asegurar suficiente espacio en disco y permisos de escritura.

### Problema: "test_environment.py falla"
**Solución**: Verificar que los 4 archivos CSV críticos están en las ubicaciones correctas.



*Una vez descargados los archivos correctamente, puedes ejecutar el pipeline completo siguiendo las instrucciones en el README.md principal.*