# xgboost_firmas_espectrales.py
# XGBoost optimizado para trabajar con firmas espectrales directas
# Proyecto: María José Erazo González - UDP

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
from scipy.stats import randint, uniform
from scipy import signal

# IMPORTACIÓN COMPATIBLE DE TRAPZ
try:
    # Intentar con scipy.integrate.trapezoid (nuevo)
    from scipy.integrate import trapezoid as trapz
    print("✅ Usando scipy.integrate.trapezoid")
except ImportError:
    try:
        # Intentar con numpy.trapz (clásico)
        from numpy import trapz
        print("✅ Usando numpy.trapz")
    except ImportError:
        try:
            # Intentar con numpy.trapezoid (nuevo numpy)
            from numpy import trapezoid as trapz
            print("✅ Usando numpy.trapezoid")
        except ImportError:
            # Implementación manual como último recurso
            def trapz(y, x=None):
                if x is None:
                    return np.sum((y[1:] + y[:-1]) / 2)
                else:
                    return np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)
            print("⚠️ Usando implementación manual de trapz")

# Intentar importar herramientas de balanceo
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    BALANCEO_DISPONIBLE = True
    print("✅ Herramientas de balanceo disponibles")
except ImportError:
    BALANCEO_DISPONIBLE = False
    print("⚠️ imblearn no disponible")

# Benchmarks SVM conocidos (del contexto del proyecto) - ACTUALIZADOS CON NOMBRES CORRECTOS
SVM_BENCHMARKS = {
    'caffeine_ng_l': 0.8000,
    'doc_mg_l': 0.8750, 
    'turbidity_ntu': 0.7630,
    'acesulfame_ng_l': 0.6880,
    'nh4_mg_l': 0.5880,
    # Benchmarks adicionales del proyecto
    'diuron_ng_l': 0.6250,
    'mecoprop_ng_l': 0.8000,
    'candesartan_ng_l': 1.0000,
    'benzotriazole_ng_l': 0.0000,
    # Contaminantes adicionales con estimaciones conservadoras
    'po4_mg_l': 0.7000,
    'so4_mg_l': 0.6500,
    'nsol_mg_l': 0.6000
}

def clasificar_contaminante(contaminante):
    """
    Clasifica contaminantes según su naturaleza química para análisis estadístico
    (Igual que en el sistema SVM para consistencia)
    
    Args:
        contaminante (str): Nombre del contaminante a clasificar
        
    Returns:
        str: Categoría del contaminante ('Inorgánico', 'Orgánico', 'Fisicoquímico')
    """
    
    inorganicos = ['Nh4_Mg_L', 'Po4_Mg_L', 'So4_Mg_L', 'Nsol_Mg_L']
    parametros_fisicos = ['Doc_Mg_L', 'Turbidity_Ntu']
    
    if contaminante in inorganicos:
        return 'Inorgánico'
    elif contaminante in parametros_fisicos:
        return 'Fisicoquímico'
    else:
        return 'Orgánico'

class XGBoostFirmasEspectrales:
    """
    XGBoost optimizado para firmas espectrales directas
    Convierte datos espectrales (wavelength, high_mean, low_mean) en features para XGBoost
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.contaminantes_disponibles = self._detectar_contaminantes()
        
        # Configuración para reproducibilidad
        np.random.seed(42)
        
        # Configurar directorios de salida
        self.results_dir = "resultados_xgboost_firmas_espectrales"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _detectar_contaminantes(self):
        """Detecta contaminantes disponibles usando el mapeo estándar del proyecto"""
        
        # Mapeo estándar del proyecto (igual que en sistema SVM)
        self.mapeo_carpetas = {
            'Doc_Mg_L': 'Doc',
            'Nh4_Mg_L': 'Nh4', 
            'Turbidity_Ntu': 'Turbidity',
            'Caffeine_Ng_L': 'Caffeine',
            'Acesulfame_Ng_L': 'Acesulfame',
            '4-&5-Methylbenzotriazole_Ng_L': 'Methylbenzotriazole',
            '6Ppd-Quinone_Ng_L': 'Quinone',
            '13-Diphenylguanidine_Ng_L': 'Diphenylguanidine',
            'Benzotriazole_Ng_L': 'Benzotriazole',
            'Candesartan_Ng_L': 'Candesartan',
            'Citalopram_Ng_L': 'Citalopram',
            'Cyclamate_Ng_L': 'Cyclamate',
            'Deet_Ng_L': 'Deet',
            'Diclofenac_Ng_L': 'Diclofenac',
            'Diuron_Ng_L': 'Diuron',
            'Hmmm_Ng_L': 'Hmmm',
            'Hydrochlorthiazide_Ng_L': 'Hydrochlorthiazide',
            'Mecoprop_Ng_L': 'Mecoprop',
            'Nsol_Mg_L': 'Nsol',
            'Oit_Ng_L': 'Oit',
            'Po4_Mg_L': 'Po4',
            'So4_Mg_L': 'So4'
        }
        
        contaminantes_disponibles = []
        
        if not os.path.exists(self.directorio_base):
            print(f"❌ No se encontró directorio base: {self.directorio_base}")
            return []
        
        for contaminante, carpeta in self.mapeo_carpetas.items():
            ruta_carpeta = os.path.join(self.directorio_base, carpeta)
            
            if os.path.exists(ruta_carpeta):
                # Buscar específicamente archivos *_datos_espectrales.csv
                archivos_espectrales = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
                
                if archivos_espectrales:
                    contaminantes_disponibles.append(contaminante)
        
        print(f"🔬 Contaminantes detectados: {len(contaminantes_disponibles)}")
        for cont in sorted(contaminantes_disponibles):
            carpeta = self.mapeo_carpetas[cont]
            print(f"   • {cont} -> {carpeta}/")
        
        return sorted(contaminantes_disponibles)
    
    def cargar_firma_espectral(self, contaminante):
        """Carga firma espectral de un contaminante usando el mapeo estándar"""
        print(f"\n📊 Cargando firma espectral: {contaminante}")
        
        # Obtener nombre de carpeta usando mapeo estándar
        if not hasattr(self, 'mapeo_carpetas'):
            # Si no se ha ejecutado _detectar_contaminantes, recrear mapeo
            self._detectar_contaminantes()
        
        if contaminante not in self.mapeo_carpetas:
            raise ValueError(f"Contaminante {contaminante} no está en el mapeo estándar")
        
        carpeta = self.mapeo_carpetas[contaminante]
        ruta_carpeta = os.path.join(self.directorio_base, carpeta)
        
        if not os.path.exists(ruta_carpeta):
            raise FileNotFoundError(f"No existe carpeta {ruta_carpeta}")
        
        # Buscar específicamente archivos que terminen en _datos_espectrales.csv
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
        
        if not archivos_espectrales:
            archivos_disponibles = os.listdir(ruta_carpeta)
            raise FileNotFoundError(f"No se encontró archivo *_datos_espectrales.csv en {ruta_carpeta}\nArchivos disponibles: {archivos_disponibles}")
        
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        # Cargar datos
        datos = pd.read_csv(ruta_archivo)
        
        print(f"   📁 Archivo: {archivo_espectral}")
        print(f"   📂 Carpeta: {carpeta}/")
        print(f"   📏 Dimensiones: {datos.shape}")
        print(f"   📋 Columnas: {list(datos.columns)}")
        
        # Verificar columnas requeridas
        columnas_requeridas = ['wavelength', 'high_mean', 'low_mean']
        columnas_encontradas = list(datos.columns)
        
        if not all(col in columnas_encontradas for col in columnas_requeridas):
            print(f"❌ Error: El archivo no tiene las columnas requeridas")
            print(f"   Requeridas: {columnas_requeridas}")
            print(f"   Encontradas: {columnas_encontradas}")
            raise ValueError(f"Columnas faltantes en {archivo_espectral}")
        
        # Limpiar y ordenar
        datos = datos.dropna()
        datos = datos.sort_values('wavelength').reset_index(drop=True)
        
        print(f"   ✅ Datos preparados: {datos.shape}")
        print(f"   🌈 Rango wavelength: {datos['wavelength'].min():.1f} - {datos['wavelength'].max():.1f} nm")
        
        return datos

    
    def extraer_features_espectrales_avanzadas(self, datos):
        """
        Extrae features avanzadas de las firmas espectrales para XGBoost
        Convierte datos espectrales en tabla de features tabulares
        """
        print(f"🔧 Extrayendo features espectrales avanzadas...")
        
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        features_dict = {}
        
        # Para cada concentración (alta y baja)
        for concentracion, response in [('high', high_response), ('low', low_response)]:
            prefix = f"{concentracion}_"
            
            # 1. ESTADÍSTICAS BÁSICAS
            features_dict[f'{prefix}mean'] = np.mean(response)
            features_dict[f'{prefix}std'] = np.std(response)
            features_dict[f'{prefix}median'] = np.median(response)
            features_dict[f'{prefix}min'] = np.min(response)
            features_dict[f'{prefix}max'] = np.max(response)
            features_dict[f'{prefix}range'] = np.ptp(response)  # max - min
            features_dict[f'{prefix}iqr'] = np.percentile(response, 75) - np.percentile(response, 25)
            
            # 2. ESTADÍSTICAS DE FORMA
            from scipy.stats import skew, kurtosis
            features_dict[f'{prefix}skewness'] = skew(response)
            features_dict[f'{prefix}kurtosis'] = kurtosis(response)
            features_dict[f'{prefix}cv'] = np.std(response) / (np.mean(response) + 1e-8)  # Coeficiente de variación
            
            # 3. PERCENTILES
            for p in [10, 25, 75, 90]:
                features_dict[f'{prefix}p{p}'] = np.percentile(response, p)
            
            # 4. CARACTERÍSTICAS ESPECTRALES ESPECÍFICAS
            
            # Área bajo la curva (integral) - USANDO LA FUNCIÓN IMPORTADA
            features_dict[f'{prefix}auc_total'] = trapz(response, wavelengths)
            
            # Áreas por regiones espectrales
            uv_mask = wavelengths <= 400
            vis_mask = (wavelengths > 400) & (wavelengths <= 700)
            nir_mask = wavelengths > 700
            
            if np.any(uv_mask):
                features_dict[f'{prefix}auc_uv'] = trapz(response[uv_mask], wavelengths[uv_mask])
            else:
                features_dict[f'{prefix}auc_uv'] = 0
                
            if np.any(vis_mask):
                features_dict[f'{prefix}auc_vis'] = trapz(response[vis_mask], wavelengths[vis_mask])
            else:
                features_dict[f'{prefix}auc_vis'] = 0
                
            if np.any(nir_mask):
                features_dict[f'{prefix}auc_nir'] = trapz(response[nir_mask], wavelengths[nir_mask])
            else:
                features_dict[f'{prefix}auc_nir'] = 0
            
            # 5. PICOS Y VALLES
            # Encontrar picos
            peaks, _ = signal.find_peaks(response, height=np.percentile(response, 70))
            features_dict[f'{prefix}n_peaks'] = len(peaks)
            
            if len(peaks) > 0:
                features_dict[f'{prefix}peak_max'] = np.max(response[peaks])
                features_dict[f'{prefix}peak_mean'] = np.mean(response[peaks])
                features_dict[f'{prefix}peak_std'] = np.std(response[peaks]) if len(peaks) > 1 else 0
                features_dict[f'{prefix}peak_wavelength_mean'] = np.mean(wavelengths[peaks])
            else:
                features_dict[f'{prefix}peak_max'] = 0
                features_dict[f'{prefix}peak_mean'] = 0
                features_dict[f'{prefix}peak_std'] = 0
                features_dict[f'{prefix}peak_wavelength_mean'] = 0
            
            # Encontrar valles (picos invertidos)
            valleys, _ = signal.find_peaks(-response, height=-np.percentile(response, 30))
            features_dict[f'{prefix}n_valleys'] = len(valleys)
            
            if len(valleys) > 0:
                features_dict[f'{prefix}valley_min'] = np.min(response[valleys])
                features_dict[f'{prefix}valley_mean'] = np.mean(response[valleys])
                features_dict[f'{prefix}valley_std'] = np.std(response[valleys]) if len(valleys) > 1 else 0
            else:
                features_dict[f'{prefix}valley_min'] = 0
                features_dict[f'{prefix}valley_mean'] = 0
                features_dict[f'{prefix}valley_std'] = 0
            
            # 6. PENDIENTES Y DERIVADAS
            # Primera derivada (pendiente)
            primera_derivada = np.gradient(response, wavelengths)
            features_dict[f'{prefix}derivada_mean'] = np.mean(primera_derivada)
            features_dict[f'{prefix}derivada_std'] = np.std(primera_derivada)
            features_dict[f'{prefix}derivada_max'] = np.max(primera_derivada)
            features_dict[f'{prefix}derivada_min'] = np.min(primera_derivada)
            
            # Pendiente por regiones
            if np.any(uv_mask) and np.sum(uv_mask) > 1:
                uv_slope, _ = np.polyfit(wavelengths[uv_mask], response[uv_mask], 1)
                features_dict[f'{prefix}slope_uv'] = uv_slope
            else:
                features_dict[f'{prefix}slope_uv'] = 0
                
            if np.any(vis_mask) and np.sum(vis_mask) > 1:
                vis_slope, _ = np.polyfit(wavelengths[vis_mask], response[vis_mask], 1)
                features_dict[f'{prefix}slope_vis'] = vis_slope
            else:
                features_dict[f'{prefix}slope_vis'] = 0
                
            if np.any(nir_mask) and np.sum(nir_mask) > 1:
                nir_slope, _ = np.polyfit(wavelengths[nir_mask], response[nir_mask], 1)
                features_dict[f'{prefix}slope_nir'] = nir_slope
            else:
                features_dict[f'{prefix}slope_nir'] = 0
            
            # 7. CARACTERÍSTICAS DE FRECUENCIA (usando FFT)
            fft = np.fft.fft(response)
            fft_magnitude = np.abs(fft)
            
            # Energía en diferentes bandas de frecuencia
            features_dict[f'{prefix}fft_energy_low'] = np.sum(fft_magnitude[:len(fft)//4])
            features_dict[f'{prefix}fft_energy_mid'] = np.sum(fft_magnitude[len(fft)//4:len(fft)//2])
            features_dict[f'{prefix}fft_energy_high'] = np.sum(fft_magnitude[len(fft)//2:3*len(fft)//4])
            
            # Frecuencia dominante
            dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft)//2]) + 1  # Excluir DC
            features_dict[f'{prefix}dominant_frequency'] = dominant_freq_idx
            
            # 8. CARACTERÍSTICAS ESPECTRALES QUÍMICAS ESPECÍFICAS
            
            # Índices espectrales comunes
            # NDVI-like para espectroscopia (usando regiones apropiadas)
            if np.any(vis_mask) and np.any(nir_mask):
                vis_mean = np.mean(response[vis_mask])
                nir_mean = np.mean(response[nir_mask])
                features_dict[f'{prefix}ndvi_like'] = (nir_mean - vis_mean) / (nir_mean + vis_mean + 1e-8)
            else:
                features_dict[f'{prefix}ndvi_like'] = 0
            
            # Ratio azul/verde (si están disponibles)
            blue_mask = (wavelengths >= 450) & (wavelengths <= 495)
            green_mask = (wavelengths >= 495) & (wavelengths <= 570)
            
            if np.any(blue_mask) and np.any(green_mask):
                blue_mean = np.mean(response[blue_mask])
                green_mean = np.mean(response[green_mask])
                features_dict[f'{prefix}blue_green_ratio'] = blue_mean / (green_mean + 1e-8)
            else:
                features_dict[f'{prefix}blue_green_ratio'] = 1
            
            # 9. CARACTERÍSTICAS DE SUAVIDAD/RUGOSIDAD
            # Suavidad usando segunda derivada
            segunda_derivada = np.gradient(primera_derivada, wavelengths)
            features_dict[f'{prefix}suavidad'] = -np.mean(np.abs(segunda_derivada))  # Negativo porque menor abs = más suave
            
            # Rugosidad espectral
            features_dict[f'{prefix}rugosidad'] = np.sum(np.abs(primera_derivada))
        
        # 10. FEATURES DE COMPARACIÓN ENTRE CONCENTRACIONES
        features_dict['concentracion_ratio_mean'] = features_dict['high_mean'] / (features_dict['low_mean'] + 1e-8)
        features_dict['concentracion_diff_mean'] = features_dict['high_mean'] - features_dict['low_mean']
        features_dict['concentracion_ratio_auc'] = features_dict['high_auc_total'] / (features_dict['low_auc_total'] + 1e-8)
        features_dict['concentracion_diff_peaks'] = features_dict['high_n_peaks'] - features_dict['low_n_peaks']
        
        # Correlación entre high y low
        correlation = np.corrcoef(high_response, low_response)[0, 1]
        features_dict['high_low_correlation'] = correlation if not np.isnan(correlation) else 0
        
        # 11. FEATURES ESPECÍFICOS PARA CONTAMINANTES PROBLEMÁTICOS
        features_dict = self._agregar_features_contaminante_especificos(features_dict, wavelengths, high_response, low_response)
        
        print(f"   ✅ Features extraídas: {len(features_dict)}")
        
        return features_dict
    
    def _agregar_features_contaminante_especificos(self, features_dict, wavelengths, high_response, low_response):
        """Agrega features específicos según el tipo de contaminante"""
        
        # Features para NH4 (amonio)
        # NH4 absorbe fuertemente en UV
        uv_mask = wavelengths <= 280
        if np.any(uv_mask):
            features_dict['nh4_uv_absorption_high'] = np.mean(high_response[uv_mask])
            features_dict['nh4_uv_absorption_low'] = np.mean(low_response[uv_mask])
            features_dict['nh4_uv_enhancement'] = features_dict['nh4_uv_absorption_high'] / (features_dict['nh4_uv_absorption_low'] + 1e-8)
        
        # Features para Acesulfame (edulcorante)
        # Acesulfame tiene absorción característica en UV-visible
        uv_vis_mask = (wavelengths >= 250) & (wavelengths <= 300)
        if np.any(uv_vis_mask):
            features_dict['acesulfame_signature_high'] = np.max(high_response[uv_vis_mask])
            features_dict['acesulfame_signature_low'] = np.max(low_response[uv_vis_mask])
            features_dict['acesulfame_enhancement'] = features_dict['acesulfame_signature_high'] / (features_dict['acesulfame_signature_low'] + 1e-8)
        
        # Features para compuestos orgánicos (Caffeine, DOC, etc.)
        # Absorción en UV debido a cromóforos
        cromoforo_mask = (wavelengths >= 200) & (wavelengths <= 400)
        if np.any(cromoforo_mask):
            features_dict['organic_chromophore_high'] = trapz(high_response[cromoforo_mask], wavelengths[cromoforo_mask])
            features_dict['organic_chromophore_low'] = trapz(low_response[cromoforo_mask], wavelengths[cromoforo_mask])
        
        # Features para turbidez
        # La turbidez afecta todo el espectro de manera similar
        features_dict['turbidity_broadband_high'] = np.std(high_response) / (np.mean(high_response) + 1e-8)
        features_dict['turbidity_broadband_low'] = np.std(low_response) / (np.mean(low_response) + 1e-8)
        
        return features_dict
    
    def crear_dataset_tabular(self, datos_espectrales, factor_augmentation=30):
        """
        Convierte firma espectral en dataset tabular para XGBoost con data augmentation REDUCIDO
        """
        print(f"📊 Creando dataset tabular con augmentation CONSERVADOR (factor: {factor_augmentation})...")
        
        # Extraer features base
        features_base = self.extraer_features_espectrales_avanzadas(datos_espectrales)
        
        # Crear dataset con augmentation REDUCIDO
        samples = []
        labels = []
        
        # Muestras originales
        # Alta concentración = clase 1
        samples.append(features_base)
        labels.append(1)
        
        # Baja concentración = clase 0 (modificar features apropiadamente)
        features_low = {}
        for key, value in features_base.items():
            if key.startswith('high_'):
                # Cambiar features de alta concentración por características de baja concentración
                low_key = key.replace('high_', 'low_')
                if low_key in features_base:
                    features_low[key] = features_base[low_key]
                else:
                    features_low[key] = value * 0.5  # Simular baja concentración
            elif key.startswith('low_'):
                # Cambiar features de baja concentración por características de alta concentración  
                high_key = key.replace('low_', 'high_')
                if high_key in features_base:
                    features_low[key] = features_base[high_key]
                else:
                    features_low[key] = value * 2.0  # Simular alta concentración
            else:
                # Features comparativos se invierten
                if 'ratio' in key:
                    features_low[key] = 1 / (value + 1e-8)
                elif 'diff' in key:
                    features_low[key] = -value
                else:
                    features_low[key] = value
        
        samples.append(features_low)
        labels.append(0)
        
        # Data augmentation MUCHO MÁS CONSERVADOR para evitar overfitting
        for i in range(factor_augmentation):
            # Augmentation para clase alta concentración (1) con MAYOR variabilidad
            aug_features_high = {}
            for key, value in features_base.items():
                if 'auc' in key or 'mean' in key or 'max' in key:
                    # AUMENTAR variabilidad para evitar overfitting
                    noise = np.random.normal(1.0, 0.15)  # ±15% variación (antes 5%)
                    aug_features_high[key] = value * noise
                elif 'std' in key or 'range' in key:
                    # Aumentar variación en dispersión
                    noise = np.random.normal(1.0, 0.25)  # ±25% variación (antes 10%)
                    aug_features_high[key] = max(0, value * noise)
                elif 'n_peaks' in key or 'n_valleys' in key:
                    # Variar número de picos/valles más ampliamente
                    aug_features_high[key] = max(0, value + np.random.randint(-2, 3))  # Rango más amplio
                elif 'slope' in key:
                    # Variar pendientes más
                    noise = np.random.normal(0, 0.05)  # Más variación
                    aug_features_high[key] = value + noise
                else:
                    # Otros features con mayor variación
                    noise = np.random.normal(1.0, 0.08)  # Más variación
                    aug_features_high[key] = value * noise
            
            samples.append(aug_features_high)
            labels.append(1)
            
            # Augmentation para clase baja concentración (0) - similar incremento en variabilidad
            aug_features_low = {}
            for key, value in features_low.items():
                if 'auc' in key or 'mean' in key or 'max' in key:
                    noise = np.random.normal(1.0, 0.15)
                    aug_features_low[key] = value * noise
                elif 'std' in key or 'range' in key:
                    noise = np.random.normal(1.0, 0.25)
                    aug_features_low[key] = max(0, value * noise)
                elif 'n_peaks' in key or 'n_valleys' in key:
                    aug_features_low[key] = max(0, value + np.random.randint(-2, 3))
                elif 'slope' in key:
                    noise = np.random.normal(0, 0.05)
                    aug_features_low[key] = value + noise
                else:
                    noise = np.random.normal(1.0, 0.08)
                    aug_features_low[key] = value * noise
            
            samples.append(aug_features_low)
            labels.append(0)
        
        # Convertir a DataFrame
        df = pd.DataFrame(samples)
        df['label'] = labels
        
        print(f"   ✅ Dataset CONSERVADOR creado: {df.shape}")
        print(f"   📊 Distribución: {pd.Series(labels).value_counts().to_dict()}")
        print(f"   🔢 Features: {len([col for col in df.columns if col != 'label'])}")
        print(f"   ⚠️ Configuración anti-overfitting aplicada")
        
        return df
    
    def entrenar_xgboost_firmas(self, contaminante, factor_augmentation=50):
        """
        Entrena XGBoost específicamente para firmas espectrales
        """
        print(f"\n{'='*70}")
        print(f"🚀 ENTRENANDO XGBOOST CON FIRMAS ESPECTRALES")
        print(f"📋 Contaminante: {contaminante}")
        print(f"{'='*70}")
        
        inicio_tiempo = datetime.datetime.now()
        
        try:
            # 1. Cargar firma espectral
            datos_espectrales = self.cargar_firma_espectral(contaminante)
            
            # 2. Crear dataset tabular con data augmentation
            df_tabular = self.crear_dataset_tabular(datos_espectrales, factor_augmentation)
            
            # 3. Preparar datos para entrenamiento
            feature_columns = [col for col in df_tabular.columns if col != 'label']
            X = df_tabular[feature_columns].values
            y = df_tabular['label'].values
            
            # 4. Imputar NaN si existen
            if np.isnan(X).any():
                print("🔧 Imputando valores NaN...")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # 5. Selección de features si hay demasiadas
            n_features_max = min(50, len(feature_columns))
            if len(feature_columns) > n_features_max:
                print(f"🎯 Seleccionando mejores {n_features_max} features...")
                selector = SelectKBest(score_func=f_classif, k=n_features_max)
                X = selector.fit_transform(X, y)
                selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
                feature_columns = selected_features
            
            # 6. División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 División train/test: {X_train.shape} / {X_test.shape}")
            
            # 7. Aplicar balanceo si está disponible
            if BALANCEO_DISPONIBLE:
                print("⚖️ Aplicando SMOTE...")
                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(3, np.bincount(y_train).min() - 1))
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print(f"   ✅ Después de SMOTE: {X_train.shape}")
                except Exception as e:
                    print(f"   ⚠️ Error en SMOTE: {e}")
            
            # 8. Configurar XGBoost
            print("🔧 Configurando XGBoost...")
            
            # Parámetros optimizados para datasets pequeños aumentados
            param_dist = {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 6),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3),
                'reg_alpha': uniform(0, 0.5),
                'reg_lambda': uniform(0.5, 2.0),
                'min_child_weight': randint(1, 5)
            }
            
            # Crear modelo base
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            # 9. Optimización con RandomizedSearchCV
            print("🔍 Optimizando hiperparámetros...")
            
            cv_folds = min(5, len(X_train) // 10)  # Ajustar folds según tamaño
            cv_folds = max(3, cv_folds)
            
            search = RandomizedSearchCV(
                xgb_model, 
                param_dist,
                n_iter=30,  # Suficiente para encontrar buenos parámetros
                cv=cv_folds,
                scoring='f1',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            print(f"   ✅ Mejor score CV: {search.best_score_:.4f}")
            
            # 10. Evaluar en test
            print("📈 Evaluando modelo final...")
            
            modelo_final = search.best_estimator_
            y_pred = modelo_final.predict(X_test)
            y_pred_proba = modelo_final.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            # 11. Comparar con SVM benchmark
            comparacion_svm = self._comparar_con_svm_benchmark(contaminante, f1)
            
            # 12. Análisis de importancia
            feature_importance = {}
            if hasattr(modelo_final, 'feature_importances_'):
                importances = modelo_final.feature_importances_
                for i, importance in enumerate(importances):
                    if i < len(feature_columns):
                        feature_importance[feature_columns[i]] = float(importance)
            
            fin_tiempo = datetime.datetime.now()
            tiempo_total = (fin_tiempo - inicio_tiempo).total_seconds()
            
            # 13. Resultados finales
            resultados = {
                'contaminante': contaminante,
                'tipo': clasificar_contaminante(contaminante),
                'metodo': 'xgboost_firmas_espectrales',
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'auc': float(auc),
                'tiempo_entrenamiento': tiempo_total,
                'mejor_score_cv': float(search.best_score_),
                'mejores_parametros': search.best_params_,
                'n_muestras_train': X_train.shape[0],
                'n_muestras_test': X_test.shape[0],
                'n_features': len(feature_columns),
                'factor_augmentation': factor_augmentation,
                'comparacion_svm': comparacion_svm,
                'feature_importance': dict(list(feature_importance.items())[:10]),  # Top 10
                'features_utilizadas': feature_columns[:10],  # Primeras 10 para el log
                'archivo_fuente': os.path.join(self.directorio_base, self.mapeo_carpetas[contaminante]),
                'n_wavelengths': len(datos_espectrales)
            }
            
            # Agregar metadatos de clasificación 
            resultados.update({
                'contaminante': contaminante,
                'tipo': clasificar_contaminante(contaminante),
                'archivo_fuente': os.path.join(self.directorio_base, self.mapeo_carpetas[contaminante]),
                'n_wavelengths': len(datos_espectrales)
            })
            
            # 15. Generar visualizaciones
            self._generar_visualizaciones(contaminante, modelo_final, X_test, y_test, y_pred, 
                                        feature_columns, feature_importance)
            
            # 16. Guardar resultados
            self._guardar_resultados(contaminante, resultados)
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error entrenando {contaminante}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _comparar_con_svm_benchmark(self, contaminante, xgb_f1):
        """Compara con benchmarks SVM conocidos usando nombres estandarizados"""
        
        # Convertir nombre a formato de benchmark
        contaminante_key = contaminante.lower().replace('_', '')
        
        # Mapeo alternativo para nombres que pueden variar
        mapeo_benchmarks = {
            'docmgl': 'doc_mg_l',
            'nh4mgl': 'nh4_mg_l', 
            'turbidityntu': 'turbidity_ntu',
            'caffeinengml': 'caffeine_ng_l',
            'acesulfamengml': 'acesulfame_ng_l',
            'diuronngml': 'diuron_ng_l',
            'mecopropngml': 'mecoprop_ng_l',
            'candesartanngml': 'candesartan_ng_l',
            'benzotriazolengml': 'benzotriazole_ng_l',
            'po4mgl': 'po4_mg_l',
            'so4mgl': 'so4_mg_l',
            'nsolmgl': 'nsol_mg_l'
        }
        
        # Buscar benchmark exacto o con mapeo
        benchmark_key = None
        if contaminante_key in SVM_BENCHMARKS:
            benchmark_key = contaminante_key
        elif contaminante_key in mapeo_benchmarks and mapeo_benchmarks[contaminante_key] in SVM_BENCHMARKS:
            benchmark_key = mapeo_benchmarks[contaminante_key]
        
        if benchmark_key:
            svm_f1 = SVM_BENCHMARKS[benchmark_key]
            mejora = xgb_f1 - svm_f1
            porcentaje_mejora = (mejora / svm_f1) * 100 if svm_f1 > 0 else 0
            
            return {
                'svm_benchmark': svm_f1,
                'xgboost_f1': xgb_f1,
                'mejora_absoluta': mejora,
                'mejora_porcentual': porcentaje_mejora,
                'xgboost_es_mejor': mejora > 0,
                'mejora_significativa': mejora > 0.05,
                'benchmark_usado': benchmark_key
            }
        
        return None
    
    def _mostrar_resultados(self, resultados):
        """Muestra resultados formateados con diagnóstico de overfitting"""
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS FINALES - {resultados['contaminante'].upper()}")
        print(f"{'='*60}")
        print(f"🎯 Test Accuracy: {resultados['accuracy']:.4f}")
        print(f"🎯 F1-score: {resultados['f1_score']:.4f}")
        print(f"🎯 AUC: {resultados['auc']:.4f}")
        
        # Mostrar diagnóstico de overfitting si está disponible
        if 'train_accuracy' in resultados and 'gap_train_test' in resultados:
            print(f"🔍 Train Accuracy: {resultados['train_accuracy']:.4f}")
            print(f"🔍 Gap train-test: {resultados['gap_train_test']:+.4f}")
            print(f"🔍 Diagnóstico: {resultados.get('diagnostico_overfitting', 'N/A')}")
        
        print(f"⏱️ Tiempo: {resultados['tiempo_entrenamiento']:.1f}s")
        print(f"📊 Muestras train: {resultados['n_muestras_train']}")
        print(f"🔢 Features: {resultados['n_features']}")
        
        # Comparación con SVM
        if resultados['comparacion_svm']:
            comp = resultados['comparacion_svm']
            print(f"\n📈 COMPARACIÓN CON SVM:")
            print(f"   SVM benchmark: {comp['svm_benchmark']:.4f}")
            print(f"   XGBoost F1: {comp['xgboost_f1']:.4f}")
            print(f"   Mejora: {comp['mejora_absoluta']:+.4f} ({comp['mejora_porcentual']:+.1f}%)")
            
            if comp['mejora_significativa']:
                print(f"   ✅ MEJORA SIGNIFICATIVA")
            elif comp['xgboost_es_mejor']:
                print(f"   ✅ Mejora leve")
            else:
                print(f"   ⚠️ SVM sigue siendo mejor")
        
        # Top features
        if resultados['feature_importance']:
            print(f"\n🔑 TOP FEATURES IMPORTANTES:")
            for i, (feature, importance) in enumerate(list(resultados['feature_importance'].items())[:5], 1):
                print(f"   {i}. {feature}: {importance:.4f}")
                
        # Recomendaciones basadas en diagnóstico
        if 'diagnostico_overfitting' in resultados:
            diagnostico = resultados['diagnostico_overfitting']
            if diagnostico == 'DETECTADO':
                print(f"\n⚠️ RECOMENDACIONES:")
                print(f"   - Reducir factor de augmentation")
                print(f"   - Aumentar regularización (reg_alpha, reg_lambda)")
                print(f"   - Reducir número de features")
            elif diagnostico == 'MODERADO':
                print(f"\n✅ MODELO ACEPTABLE - Monitorear gap en producción")
    
    def _generar_visualizaciones(self, contaminante, modelo, X_test, y_test, y_pred, 
                                feature_columns, feature_importance):
        """Genera visualizaciones específicas"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'XGBoost Firmas Espectrales - {contaminante}', fontsize=14, fontweight='bold')
            
            # 1. Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
            axes[0,0].set_title('Matriz de Confusión')
            axes[0,0].set_xlabel('Predicción')
            axes[0,0].set_ylabel('Valor Real')
            
            # 2. Importancia de features (top 15)
            if feature_importance:
                top_features = list(feature_importance.items())[:15]
                names = [name[:20] for name, _ in top_features]  # Truncar nombres largos
                values = [val for _, val in top_features]
                
                y_pos = np.arange(len(names))
                axes[0,1].barh(y_pos, values, color='skyblue', edgecolor='navy')
                axes[0,1].set_yticks(y_pos)
                axes[0,1].set_yticklabels(names, fontsize=8)
                axes[0,1].set_title('Top Features Importantes')
                axes[0,1].set_xlabel('Importancia')
            
            # 3. Distribución de predicciones
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            
            axes[1,0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Clase 0', color='red')
            axes[1,0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Clase 1', color='blue')
            axes[1,0].set_title('Distribución de Probabilidades')
            axes[1,0].set_xlabel('Probabilidad Predicha')
            axes[1,0].set_ylabel('Frecuencia')
            axes[1,0].legend()
            
            # 4. Curva ROC simple
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            axes[1,1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
            axes[1,1].plot([0, 1], [0, 1], 'r--', linewidth=1)
            axes[1,1].set_title('Curva ROC')
            axes[1,1].set_xlabel('Tasa de Falsos Positivos')
            axes[1,1].set_ylabel('Tasa de Verdaderos Positivos')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar
            os.makedirs(os.path.join(self.results_dir, contaminante), exist_ok=True)
            ruta_imagen = os.path.join(self.results_dir, contaminante, f"{contaminante}_xgboost_firmas.png")
            plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 Visualizaciones guardadas: {ruta_imagen}")
            
        except Exception as e:
            print(f"   ⚠️ Error en visualizaciones: {e}")
    
    def _guardar_resultados(self, contaminante, resultados):
        """Guarda resultados en JSON"""
        
        try:
            # Crear directorio
            dir_contaminante = os.path.join(self.results_dir, contaminante)
            os.makedirs(dir_contaminante, exist_ok=True)
            
            # Guardar resultados
            ruta_json = os.path.join(dir_contaminante, f"{contaminante}_xgboost_firmas_resultados.json")
            
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"   💾 Resultados guardados: {ruta_json}")
            
        except Exception as e:
            print(f"   ⚠️ Error guardando: {e}")
    
    def entrenar_todos_contaminantes(self, factor_augmentation=25):
        """Entrena XGBoost para todos los contaminantes disponibles - CONFIGURACIÓN ANTI-OVERFITTING"""
        
        print(f"\n{'='*80}")
        print(f"🔬 ENTRENAMIENTO MASIVO XGBOOST - FIRMAS ESPECTRALES")
        print(f"⚠️ CONFIGURACIÓN ANTI-OVERFITTING APLICADA")
        print(f"{'='*80}")
        print(f"📋 Contaminantes: {len(self.contaminantes_disponibles)}")
        print(f"🔧 Factor augmentation conservador: {factor_augmentation}")
        
        resultados_todos = {}
        inicio_total = datetime.datetime.now()
        
        for i, contaminante in enumerate(self.contaminantes_disponibles):
            print(f"\n[{i+1}/{len(self.contaminantes_disponibles)}] 🔄 Procesando: {contaminante}")
            
            resultado = self.entrenar_xgboost_firmas(contaminante, factor_augmentation)
            resultados_todos[contaminante] = resultado
        
        fin_total = datetime.datetime.now()
        tiempo_total = (fin_total - inicio_total).total_seconds()
        
        # Generar reporte consolidado
        self._generar_reporte_consolidado(resultados_todos, tiempo_total)
        
        return resultados_todos
    
    def _generar_reporte_consolidado(self, resultados, tiempo_total):
        """Genera reporte consolidado final"""
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE CONSOLIDADO - XGBOOST FIRMAS ESPECTRALES")
        print(f"{'='*80}")
        
        # Filtrar resultados exitosos
        exitosos = {k: v for k, v in resultados.items() if v is not None}
        
        if not exitosos:
            print("❌ No hay resultados exitosos")
            return
        
        # Estadísticas generales
        f1_scores = [r['f1_score'] for r in exitosos.values()]
        accuracies = [r['accuracy'] for r in exitosos.values()]
        aucs = [r['auc'] for r in exitosos.values()]
        
        print(f"✅ Exitosos: {len(exitosos)}/{len(resultados)}")
        print(f"⏱️ Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   F1-score promedio: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"   Accuracy promedio: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   AUC promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        
        # Comparaciones con SVM
        comparaciones_svm = [r['comparacion_svm'] for r in exitosos.values() if r['comparacion_svm']]
        if comparaciones_svm:
            mejoras = sum(1 for comp in comparaciones_svm if comp['xgboost_es_mejor'])
            significativas = sum(1 for comp in comparaciones_svm if comp['mejora_significativa'])
            
            print(f"\n🏆 vs SVM BENCHMARKS:")
            print(f"   Mejoras: {mejoras}/{len(comparaciones_svm)}")
            print(f"   Significativas: {significativas}/{len(comparaciones_svm)}")
        
        # Ranking
        print(f"\n📋 RANKING POR F1-SCORE:")
        ranking = sorted(exitosos.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for i, (contaminante, resultado) in enumerate(ranking[:10], 1):
            f1 = resultado['f1_score']
            comp = resultado['comparacion_svm']
            
            if f1 >= 0.85:
                emoji = "🥇"
            elif f1 >= 0.75:
                emoji = "🥈"
            elif f1 >= 0.65:
                emoji = "🥉"
            else:
                emoji = "📊"
            
            svm_status = ""
            if comp:
                if comp['mejora_significativa']:
                    svm_status = "✅"
                elif comp['xgboost_es_mejor']:
                    svm_status = "⬆️"
                else:
                    svm_status = "⚠️"
            
            print(f"   {i:2d}. {contaminante:15} | F1: {f1:.4f} {emoji} {svm_status}")
        
        # Guardar reporte consolidado
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        resumen = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metodo': 'xgboost_firmas_espectrales',
            'tiempo_total_segundos': tiempo_total,
            'estadisticas': {
                'exitosos': len(exitosos),
                'f1_score_promedio': float(np.mean(f1_scores)),
                'f1_score_std': float(np.std(f1_scores)),
                'accuracy_promedio': float(np.mean(accuracies)),
                'auc_promedio': float(np.mean(aucs))
            },
            'comparaciones_svm': {
                'total_comparaciones': len(comparaciones_svm),
                'mejoras': mejoras if comparaciones_svm else 0,
                'significativas': significativas if comparaciones_svm else 0
            },
            'ranking': [(k, v['f1_score']) for k, v in ranking],
            'resultados_detallados': exitosos
        }
        
        nombre_archivo = f"xgboost_firmas_espectrales_consolidado_{timestamp}.json"
        with open(os.path.join(self.results_dir, nombre_archivo), 'w', encoding='utf-8') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Reporte consolidado: {nombre_archivo}")
        
        return resumen

def ejecutar_benchmarks_xgboost_firmas():
    """Ejecuta benchmarks específicos para contaminantes clave usando nombres estándar - ANTI-OVERFITTING"""
    
    # Contaminantes prioritarios según benchmarks SVM (nombres estándar del proyecto)
    prioritarios = ['Caffeine_Ng_L', 'Doc_Mg_L', 'Turbidity_Ntu']  # Mejores en SVM
    problematicos = ['Nh4_Mg_L', 'Acesulfame_Ng_L']  # Peores en SVM  
    benchmarks = ['Candesartan_Ng_L', 'Diuron_Ng_L', 'Mecoprop_Ng_L']  # Benchmarks conocidos
    
    print("🎯 EJECUTANDO BENCHMARKS XGBOOST FIRMAS ESPECTRALES - ANTI-OVERFITTING")
    print("="*70)
    print("⚠️ Configuración conservadora aplicada para evitar overfitting")
    
    entrenador = XGBoostFirmasEspectrales("todo/firmas_espectrales_csv")
    resultados_benchmarks = {}
    
    categorias = [
        ("🥇 PRIORITARIOS (mantener alta performance)", prioritarios),
        ("⚠️ PROBLEMÁTICOS (mejorar significativamente)", problematicos),
        ("📚 BENCHMARKS CONOCIDOS", benchmarks)
    ]
    
    for titulo, contaminantes in categorias:
        print(f"\n{titulo}:")
        print("-" * 50)
        
        for contaminante in contaminantes:
            if contaminante in entrenador.contaminantes_disponibles:
                print(f"\n🔄 Procesando: {contaminante}")
                # FACTOR AUGMENTATION REDUCIDO para evitar overfitting
                resultado = entrenador.entrenar_xgboost_firmas(contaminante, factor_augmentation=25)
                
                if resultado:
                    resultados_benchmarks[contaminante] = resultado
                    
                    # Mostrar comparación inmediata
                    if resultado['comparacion_svm']:
                        comp = resultado['comparacion_svm']
                        emoji = "✅" if comp['xgboost_es_mejor'] else "⚠️"
                        print(f"   {emoji} XGBoost: {comp['xgboost_f1']:.3f} vs SVM: {comp['svm_benchmark']:.3f}")
                        print(f"       Benchmark usado: {comp['benchmark_usado']}")
                    
                    # Mostrar diagnóstico de overfitting
                    gap = resultado.get('gap_train_test', 0)
                    diagnostico = resultado.get('diagnostico_overfitting', 'DESCONOCIDO')
                    print(f"   🔍 Gap train-test: {gap:+.3f} ({diagnostico})")
                    
            else:
                print(f"   ⚠️ {contaminante} no disponible en firmas espectrales")
                print(f"       Disponibles: {len(entrenador.contaminantes_disponibles)} contaminantes")
    
    return resultados_benchmarks

def verificar_archivos_firmas_espectrales():
    """
    Verifica que existan los archivos de firmas espectrales necesarios
    (Compatible con el sistema SVM del proyecto)
    
    Returns:
        bool: True si todos los archivos están disponibles
    """
    
    print("🔍 Verificando archivos de firmas espectrales...")
    
    directorio_base = "todo/firmas_espectrales_csv"
    
    # Mapeo estándar del proyecto
    mapeo_carpetas = {
        'Doc_Mg_L': 'Doc',
        'Nh4_Mg_L': 'Nh4', 
        'Turbidity_Ntu': 'Turbidity',
        'Caffeine_Ng_L': 'Caffeine',
        'Acesulfame_Ng_L': 'Acesulfame',
        '4-&5-Methylbenzotriazole_Ng_L': 'Methylbenzotriazole',
        '6Ppd-Quinone_Ng_L': 'Quinone',
        '13-Diphenylguanidine_Ng_L': 'Diphenylguanidine',
        'Benzotriazole_Ng_L': 'Benzotriazole',
        'Candesartan_Ng_L': 'Candesartan',
        'Citalopram_Ng_L': 'Citalopram',
        'Cyclamate_Ng_L': 'Cyclamate',
        'Deet_Ng_L': 'Deet',
        'Diclofenac_Ng_L': 'Diclofenac',
        'Diuron_Ng_L': 'Diuron',
        'Hmmm_Ng_L': 'Hmmm',
        'Hydrochlorthiazide_Ng_L': 'Hydrochlorthiazide',
        'Mecoprop_Ng_L': 'Mecoprop',
        'Nsol_Mg_L': 'Nsol',
        'Oit_Ng_L': 'Oit',
        'Po4_Mg_L': 'Po4',
        'So4_Mg_L': 'So4'
    }
    
    if not os.path.exists(directorio_base):
        print(f"❌ No se encontró directorio base: {directorio_base}")
        return False
    
    archivos_encontrados = 0
    archivos_faltantes = []
    
    for contaminante, carpeta in mapeo_carpetas.items():
        ruta_carpeta = os.path.join(directorio_base, carpeta)
        
        if os.path.exists(ruta_carpeta):
            # Buscar específicamente el archivo _datos_espectrales.csv
            archivos_espectrales = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
            
            if archivos_espectrales:
                archivo_encontrado = archivos_espectrales[0]
                print(f"   ✅ {contaminante} -> {carpeta}/{archivo_encontrado}")
                archivos_encontrados += 1
            else:
                print(f"   ❌ {contaminante} -> {carpeta}/ (sin archivo *_datos_espectrales.csv)")
                archivos_faltantes.append(f"{carpeta}/*_datos_espectrales.csv")
        else:
            print(f"   ❌ {contaminante} -> {carpeta}/ (carpeta no existe)")
            archivos_faltantes.append(f"{carpeta}/")
    
    print(f"\n📊 Resumen: {archivos_encontrados}/{len(mapeo_carpetas)} archivos encontrados")
    
    if archivos_faltantes:
        print(f"❌ Archivos/carpetas faltantes: {len(archivos_faltantes)}")
        print("📁 Estructura esperada: todo/firmas_espectrales_csv/[Contaminante]/[Contaminante]_datos_espectrales.csv")
    else:
        print("✅ Todos los archivos necesarios están disponibles")
    
    return len(archivos_faltantes) == 0

def main():
    """Función principal con verificación de archivos"""
    
    print("🧬 XGBOOST PARA FIRMAS ESPECTRALES - MARÍA JOSÉ")
    print("="*60)
    print("🎯 Detección de Contaminantes en Aguas Superficiales")
    print("🏫 Universidad Diego Portales - Anteproyecto de Título")
    print("="*60)
    
    # Verificar archivos antes de comenzar
    archivos_ok = verificar_archivos_firmas_espectrales()
    
    if not archivos_ok:
        print("\n⚠️ Algunos archivos de firmas espectrales no están disponibles.")
        continuar = input("¿Deseas continuar de todas formas? (s/n): ").lower().strip()
        if continuar != 's':
            print("❌ Programa terminado.")
            return None
    
    entrenador = XGBoostFirmasEspectrales("todo/firmas_espectrales_csv")
    
    print(f"\n🔬 Contaminantes disponibles: {len(entrenador.contaminantes_disponibles)}")
    
    print("\n🎮 OPCIONES:")
    print("1. 🧪 Entrenar un contaminante específico")
    print("2. 🎯 Ejecutar benchmarks clave (recomendado)")
    print("3. 🔬 Entrenar todos los contaminantes")
    print("4. 🔍 Solo verificar archivos")
    
    try:
        opcion = input("\nSelecciona una opción (1-4): ").strip()
        
        if opcion == '1':
            print(f"\nContaminantes disponibles:")
            for i, cont in enumerate(entrenador.contaminantes_disponibles, 1):
                print(f"  {i}. {cont}")
            
            seleccion = input("\nEscribe el nombre exacto del contaminante: ").strip()
            if seleccion in entrenador.contaminantes_disponibles:
                resultado = entrenador.entrenar_xgboost_firmas(seleccion)
                return {seleccion: resultado}
            else:
                print(f"❌ Contaminante '{seleccion}' no encontrado")
                
        elif opcion == '2':
            print("\n🚀 Ejecutando benchmarks automáticamente...")
            resultados = ejecutar_benchmarks_xgboost_firmas()
            return resultados
            
        elif opcion == '3':
            print("\n🔬 Entrenando todos los contaminantes...")
            resultados = entrenador.entrenar_todos_contaminantes()
            return resultados
            
        elif opcion == '4':
            verificar_archivos_firmas_espectrales()
            return None
            
        else:
            print("❌ Opción inválida. Ejecutando benchmarks por defecto...")
            resultados = ejecutar_benchmarks_xgboost_firmas()
            return resultados
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario.")
        return None
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    import sys
    
    if len(sys.argv) > 1:
        contaminante = sys.argv[1]
        entrenador = XGBoostFirmasEspectrales("todo/firmas_espectrales_csv")
        resultado = entrenador.entrenar_xgboost_firmas(contaminante)
    else:
        main()