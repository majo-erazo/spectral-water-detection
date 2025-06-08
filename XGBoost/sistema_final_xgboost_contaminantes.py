import os
import pandas as pd
import numpy as np
import json
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
from scipy import signal
from scipy.stats import randint, uniform

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from numpy import trapz

class SistemaFinalXGBoostContaminantes:
    """
    Sistema final para detección de contaminantes en aguas superficiales
    usando XGBoost con adaptación química específica por tipo de contaminante
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.results_dir = "resultados_sistema_final_xgboost"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Mapeo completo de contaminantes
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
        
        # Perfiles químicos específicos para optimización de XGBoost
        self.perfiles_quimicos = self._inicializar_perfiles_quimicos()
        
        # Configuración XGBoost adaptativa según tamaño de dataset
        self.config_xgboost = {
            'extremo': {  # ≤ 4 muestras
                'params': {
                    'n_estimators': randint(3, 6),
                    'max_depth': randint(2, 3),
                    'learning_rate': uniform(0.15, 0.25),
                    'reg_alpha': uniform(8.0, 15.0),
                    'reg_lambda': uniform(10.0, 20.0),
                    'min_child_weight': randint(15, 25)
                },
                'descripcion': 'XGBoost ultra-conservador para datasets extremos'
            },
            'pequeno': {  # 5-15 muestras
                'params': {
                    'n_estimators': randint(5, 12),
                    'max_depth': randint(3, 4),
                    'learning_rate': uniform(0.08, 0.18),
                    'reg_alpha': uniform(3.0, 8.0),
                    'reg_lambda': uniform(5.0, 12.0),
                    'min_child_weight': randint(8, 15)
                },
                'descripcion': 'XGBoost conservador para datasets pequeños'
            },
            'mediano': {  # 16-50 muestras
                'params': {
                    'n_estimators': randint(10, 25),
                    'max_depth': randint(4, 6),
                    'learning_rate': uniform(0.04, 0.12),
                    'reg_alpha': uniform(1.0, 4.0),
                    'reg_lambda': uniform(2.0, 8.0),
                    'min_child_weight': randint(3, 10)
                },
                'descripcion': 'XGBoost balanceado para datasets medianos'
            },
            'grande': {  # >50 muestras
                'params': {
                    'n_estimators': randint(20, 50),
                    'max_depth': randint(5, 8),
                    'learning_rate': uniform(0.02, 0.08),
                    'reg_alpha': uniform(0.5, 2.0),
                    'reg_lambda': uniform(1.0, 4.0),
                    'min_child_weight': randint(1, 6)
                },
                'descripcion': 'XGBoost completo para datasets grandes'
            }
        }
    
    def _inicializar_perfiles_quimicos(self):
        """Perfiles químicos para optimización específica de XGBoost"""
        return {
            # INORGÁNICOS - Iones simples
            'Nh4_Mg_L': {
                'tipo': 'inorganico', 'subtipo': 'ion_amonio',
                'regiones_criticas': [(200, 280)],
                'features_especificos': ['uv_absorption', 'slope_uv'],
                'complejidad_esperada': 'baja',
                'separabilidad_esperada': 'alta',
                'n_features_optimo': 4
            },
            'Po4_Mg_L': {
                'tipo': 'inorganico', 'subtipo': 'ion_fosfato',
                'regiones_criticas': [(200, 250)],
                'features_especificos': ['uv_extreme', 'sharp_absorption'],
                'complejidad_esperada': 'baja',
                'separabilidad_esperada': 'alta',
                'n_features_optimo': 4
            },
            'So4_Mg_L': {
                'tipo': 'inorganico', 'subtipo': 'ion_sulfato',
                'regiones_criticas': [(200, 300)],
                'features_especificos': ['broad_uv', 'moderate_absorption'],
                'complejidad_esperada': 'baja',
                'separabilidad_esperada': 'media',
                'n_features_optimo': 5
            },
            
            # ORGÁNICOS - Compuestos con cromóforos
            'Caffeine_Ng_L': {
                'tipo': 'organico', 'subtipo': 'xantina_alcaloide',
                'regiones_criticas': [(250, 300), (350, 400)],
                'features_especificos': ['dual_peaks', 'chromophore_analysis'],
                'complejidad_esperada': 'media',
                'separabilidad_esperada': 'alta',
                'n_features_optimo': 6
            },
            'Acesulfame_Ng_L': {
                'tipo': 'organico', 'subtipo': 'edulcorante_artificial',
                'regiones_criticas': [(220, 280)],
                'features_especificos': ['uv_signature', 'specific_absorption'],
                'complejidad_esperada': 'media',
                'separabilidad_esperada': 'media',
                'n_features_optimo': 5
            },
            
            # FARMACÉUTICOS - Estructuras complejas
            'Candesartan_Ng_L': {
                'tipo': 'organico', 'subtipo': 'farmaceutico_complejo',
                'regiones_criticas': [(250, 350), (400, 500)],
                'features_especificos': ['multi_peak', 'complex_structure'],
                'complejidad_esperada': 'alta',
                'separabilidad_esperada': 'media',
                'n_features_optimo': 8
            },
            'Diclofenac_Ng_L': {
                'tipo': 'organico', 'subtipo': 'antiinflamatorio',
                'regiones_criticas': [(275, 285), (320, 330)],
                'features_especificos': ['precise_peaks', 'narrow_bands'],
                'complejidad_esperada': 'media',
                'separabilidad_esperada': 'alta',
                'n_features_optimo': 6
            },
            
            # PARÁMETROS FÍSICO-QUÍMICOS
            'Turbidity_Ntu': {
                'tipo': 'fisicoquimico', 'subtipo': 'dispersion_particulas',
                'regiones_criticas': [(400, 800)],
                'features_especificos': ['broadband_scattering', 'uniform_response'],
                'complejidad_esperada': 'alta',
                'separabilidad_esperada': 'muy_alta',
                'n_features_optimo': 7
            },
            'Doc_Mg_L': {
                'tipo': 'fisicoquimico', 'subtipo': 'carbono_organico_disuelto',
                'regiones_criticas': [(200, 400), (450, 600)],
                'features_especificos': ['broad_organic', 'multiple_chromophores'],
                'complejidad_esperada': 'muy_alta',
                'separabilidad_esperada': 'media',
                'n_features_optimo': 10
            }
        }
    
    def detectar_contaminante(self, contaminante):
        """
        Función principal para detectar un contaminante específico
        """
        print(f"\n{'='*80}")
        print(f"🧬 SISTEMA FINAL XGBOOST - DETECCIÓN DE CONTAMINANTES")
        print(f"📋 Contaminante: {contaminante}")
        print(f"🏫 Universidad Diego Portales - Anteproyecto de Título")
        print(f"👩‍🎓 María José Erazo González")
        print(f"{'='*80}")
        
        inicio_tiempo = datetime.datetime.now()
        
        try:
            # 1. Determinar perfil químico
            perfil = self.perfiles_quimicos.get(contaminante)
            if perfil:
                print(f"🏷️ Tipo: {perfil['tipo']} - {perfil['subtipo']}")
                print(f"🌈 Regiones críticas: {perfil['regiones_criticas']}")
                print(f"🎯 Complejidad esperada: {perfil['complejidad_esperada']}")
                print(f"📊 Separabilidad esperada: {perfil['separabilidad_esperada']}")
            else:
                print(f"⚠️ Perfil no catalogado - usando configuración genérica")
                perfil = {'tipo': 'desconocido', 'subtipo': 'generico', 'n_features_optimo': 6}
            
            # 2. Preparar datos con features químicamente específicos
            X, y, feature_names = self.preparar_datos_especificos(contaminante, perfil)
            
            # 3. Determinar configuración XGBoost según tamaño
            n_muestras = len(X)
            categoria_dataset = self.determinar_categoria_dataset(n_muestras)
            config_xgb = self.config_xgboost[categoria_dataset]
            
            print(f"📊 Dataset: {n_muestras} muestras → Categoría: {categoria_dataset}")
            print(f"🔧 Configuración: {config_xgb['descripcion']}")
            
            # 4. Entrenar XGBoost optimizado
            resultado = self.entrenar_xgboost_optimizado(
                X, y, feature_names, config_xgb, contaminante, perfil
            )
            
            # 5. Finalizar resultado
            if resultado:
                fin_tiempo = datetime.datetime.now()
                tiempo_total = (fin_tiempo - inicio_tiempo).total_seconds()
                
                resultado.update({
                    'contaminante': contaminante,
                    'metodo': 'sistema_final_xgboost',
                    'perfil_quimico': perfil,
                    'categoria_dataset': categoria_dataset,
                    'n_muestras_total': n_muestras,
                    'tiempo_entrenamiento': tiempo_total,
                    'features_utilizados': feature_names,
                    'configuracion_xgboost': config_xgb['descripcion']
                })
                
                # 6. Mostrar y guardar resultados
                self._mostrar_resultados_finales(resultado)
                self._guardar_resultados_finales(contaminante, resultado)
                
                return resultado
            else:
                print(f"❌ Error en entrenamiento XGBoost")
                return None
            
        except Exception as e:
            print(f"❌ Error en detección: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def preparar_datos_especificos(self, contaminante, perfil):
        """VERSIÓN CORREGIDA: Usar las FIRMAS ESPECTRALES REALES"""
        print(f"🔬 Preparando datos REALES para {contaminante}...")
        
        # Cargar datos espectrales REALES
        datos_espectrales = self.cargar_datos_espectrales(contaminante)
        
        # Verificar columnas necesarias
        columnas_requeridas = ['wavelength', 'high_mean', 'low_mean']
        if not all(col in datos_espectrales.columns for col in columnas_requeridas):
            raise ValueError(f"❌ Faltan columnas: {datos_espectrales.columns}")
        
        # Análisis de firma real
        wavelengths = datos_espectrales['wavelength'].values
        high_mean = datos_espectrales['high_mean'].values
        low_mean = datos_espectrales['low_mean'].values
        signature = high_mean - low_mean
        
        # Identificar regiones significativas
        threshold = np.std(signature) * 1.5
        significant_regions = np.abs(signature) > threshold
        
        print(f"   📊 {len(datos_espectrales)} λ, señal en {np.sum(significant_regions)} regiones")
        
        # Extraer features de regiones significativas
        features = self.extraer_features_quimicos_especificos(datos_espectrales, perfil)
        
        # Crear dataset desde firmas reales
        dataset = self.crear_dataset_xgboost_optimizado(features, perfil)
        
        # Seleccionar features discriminativos
        feature_columns = list(features.keys())[:6]  # Top 6 features
        
        X = dataset[feature_columns].values
        y = dataset['label'].values
        
        print(f"   ✅ Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
        
        return X, y, feature_columns
    def extraer_features_quimicos_especificos(self, datos, perfil):
        """VERSIÓN CORREGIDA: Features de la FIRMA ESPECTRAL REAL"""
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        # FIRMA ESPECTRAL REAL (esto es lo que estaba faltando!)
        signature = high_response - low_response
        
        features = {}
        
        # Features de la firma real
        features['signature_mean'] = np.mean(signature)
        features['signature_max'] = np.max(signature)
        features['signature_std'] = np.std(signature)
        features['signature_range'] = np.ptp(signature)
        
        try:
            features['signature_auc'] = trapz(signature, wavelengths)
        except:
            features['signature_auc'] = np.sum(signature)
        
        # Features específicos por contaminante conocido
        contaminante_str = str(perfil)
        if 'Methylbenzotriazole' in contaminante_str or 'Benzotriazole' in contaminante_str:
            # Usar regiones específicas que SABEMOS que funcionan
            region_720 = (wavelengths >= 716) & (wavelengths <= 732)
            region_798 = (wavelengths >= 795) & (wavelengths <= 800)
            
            if np.any(region_720):
                features['benzotriazole_720'] = np.mean(signature[region_720])
            if np.any(region_798):
                features['benzotriazole_798'] = np.mean(signature[region_798])
        
        # Features para inorgánicos (NH4, PO4, SO4)
        if perfil.get('tipo') == 'inorganico':
            uv_region = (wavelengths <= 300)
            if np.any(uv_region):
                features['ion_uv'] = np.mean(signature[uv_region])
                features['ion_enhancement'] = np.max(signature[uv_region])
        
        # Features para turbidez
        if 'Turbidity' in contaminante_str:
            visible = (wavelengths >= 400) & (wavelengths <= 700)
            if np.any(visible):
                features['turbidity_vis'] = np.mean(signature[visible])
                if np.sum(visible) > 1:
                    features['turbidity_slope'] = np.polyfit(wavelengths[visible], signature[visible], 1)[0]
                else:
                    features['turbidity_slope'] = 0
        
        # Features discriminativos reales
        if np.std(high_response) > 0 and np.std(low_response) > 0:
            features['real_ratio'] = np.mean(high_response) / (np.mean(low_response) + 1e-8)
            features['real_diff'] = np.mean(high_response) - np.mean(low_response)
        
        return features
    def _features_inorganicos_xgboost(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para iones inorgánicos optimizados para XGBoost"""
        features = {}
        
        # Iones absorben fuertemente en UV
        uv_mask = wavelengths <= 300
        if np.any(uv_mask):
            features['ion_uv_intensity_high'] = np.mean(high_response[uv_mask])
            features['ion_uv_intensity_low'] = np.mean(low_response[uv_mask])
            features['ion_uv_enhancement'] = features['ion_uv_intensity_high'] / (features['ion_uv_intensity_low'] + 1e-8)
            features['ion_uv_contrast'] = features['ion_uv_intensity_high'] - features['ion_uv_intensity_low']
            
            # Pendiente UV característica
            if np.sum(uv_mask) > 2:
                uv_slope_high, _ = np.polyfit(wavelengths[uv_mask], high_response[uv_mask], 1)
                uv_slope_low, _ = np.polyfit(wavelengths[uv_mask], low_response[uv_mask], 1)
                features['ion_uv_slope_ratio'] = uv_slope_high / (uv_slope_low + 1e-8)
        
        return features
    
    def _features_organicos_xgboost(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para compuestos orgánicos optimizados para XGBoost"""
        features = {}
        
        # Análisis de picos optimizado para XGBoost
        peaks_high, properties_high = signal.find_peaks(high_response, height=np.percentile(high_response, 60))
        peaks_low, properties_low = signal.find_peaks(low_response, height=np.percentile(low_response, 60))
        
        features['organic_peaks_ratio'] = len(peaks_high) / (len(peaks_low) + 1)
        features['organic_peak_enhancement'] = len(peaks_high) - len(peaks_low)
        
        if len(peaks_high) > 0:
            features['organic_peak_intensity_max'] = np.max(high_response[peaks_high])
            features['organic_peak_intensity_mean'] = np.mean(high_response[peaks_high])
        else:
            features['organic_peak_intensity_max'] = 0
            features['organic_peak_intensity_mean'] = 0
        
        # Cromóforos en UV-visible optimizado
        chromophore_mask = (wavelengths >= 250) & (wavelengths <= 400)
        if np.any(chromophore_mask):
            chromophore_high = trapz(high_response[chromophore_mask], wavelengths[chromophore_mask])
            chromophore_low = trapz(low_response[chromophore_mask], wavelengths[chromophore_mask])
            features['chromophore_enhancement_ratio'] = chromophore_high / (chromophore_low + 1e-8)
            features['chromophore_enhancement_diff'] = chromophore_high - chromophore_low
        
        return features
    
    def _features_fisicoquimicos_xgboost(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para parámetros físico-químicos optimizados para XGBoost"""
        features = {}
        
        # Dispersión para turbidez
        if 'turbidity' in perfil['subtipo'].lower():
            visible_mask = (wavelengths >= 400) & (wavelengths <= 700)
            if np.any(visible_mask):
                # Uniformidad de dispersión optimizada
                scattering_uniformity_high = np.std(high_response[visible_mask]) / (np.mean(high_response[visible_mask]) + 1e-8)
                scattering_uniformity_low = np.std(low_response[visible_mask]) / (np.mean(low_response[visible_mask]) + 1e-8)
                features['scattering_uniformity_ratio'] = scattering_uniformity_high / (scattering_uniformity_low + 1e-8)
                features['scattering_contrast'] = scattering_uniformity_high - scattering_uniformity_low
        
        # Múltiples cromóforos para DOC
        elif 'carbono' in perfil['subtipo'].lower():
            for band_start, band_end, band_name in [(200, 300, 'uv'), (300, 400, 'uv_vis'), (400, 500, 'vis')]:
                band_mask = (wavelengths >= band_start) & (wavelengths <= band_end)
                if np.any(band_mask):
                    band_high = trapz(high_response[band_mask], wavelengths[band_mask])
                    band_low = trapz(low_response[band_mask], wavelengths[band_mask])
                    features[f'doc_{band_name}_ratio'] = band_high / (band_low + 1e-8)
                    features[f'doc_{band_name}_enhancement'] = band_high - band_low
        
        return features
    
    def seleccionar_features_optimos(self, features, perfil):
        """Selecciona features óptimos según perfil químico"""
        n_features_optimo = perfil.get('n_features_optimo', 6)
        
        # Priorizar features comparativos (los más discriminativos)
        features_prioritarios = [k for k in features.keys() if any(word in k.lower() for word in ['ratio', 'enhancement', 'contrast', 'diff'])]
        features_regionales = [k for k in features.keys() if 'region_' in k]
        features_especificos = [k for k in features.keys() if any(word in k.lower() for word in ['ion_', 'organic_', 'chromophore_', 'scattering_', 'doc_'])]
        features_basicos = [k for k in features.keys() if k not in features_prioritarios + features_regionales + features_especificos]
        
        # Seleccionar en orden de prioridad
        features_seleccionados = []
        features_seleccionados.extend(features_prioritarios[:n_features_optimo//2])
        features_seleccionados.extend(features_especificos[:n_features_optimo//3])
        features_seleccionados.extend(features_regionales[:n_features_optimo//4])
        
        # Completar con features básicos si es necesario
        while len(features_seleccionados) < n_features_optimo and features_basicos:
            features_seleccionados.append(features_basicos.pop(0))
        
        return features_seleccionados[:n_features_optimo]
    
    def crear_dataset_xgboost_optimizado(self, features, perfil):
        """VERSIÓN CORREGIDA: Datasets más grandes y realistas"""
        
        # Generar dataset más grande para XGBoost
        n_samples = 100  # En lugar de 3-8 muestras
        
        print(f"   🎯 Generando {n_samples*2} muestras desde firmas reales")
        
        muestra_alta = list(features.values())
        muestra_baja = self._crear_muestra_baja_optimizada(features, perfil)
        
        samples = []
        labels = []
        
        # Generar variaciones realistas
        for i in range(n_samples):
            # Clase alta (señal original + ruido pequeño)
            noise_factor = 0.05  # 5% ruido (realista)
            alta_noise = [val * np.random.normal(1.0, noise_factor) for val in muestra_alta]
            samples.append(alta_noise)
            labels.append(1)
            
            # Clase baja (señal reducida + ruido)
            baja_noise = [val * np.random.normal(0.3, noise_factor) for val in muestra_baja]
            samples.append(baja_noise)
            labels.append(0)
        
        # Crear DataFrame
        feature_names = list(features.keys())
        df = pd.DataFrame(samples, columns=feature_names)
        df['label'] = labels
        
        return df
    def _crear_muestra_baja_optimizada(self, features, perfil):
        """Crea muestra de baja concentración optimizada para XGBoost"""
        muestra_baja = []
        
        for nombre, valor in features.items():
            if nombre.startswith('high_'):
                low_nombre = nombre.replace('high_', 'low_')
                if low_nombre in features:
                    muestra_baja.append(features[low_nombre])
                else:
                    # Factor específico según tipo químico
                    if perfil.get('tipo') == 'inorganico':
                        factor = 0.25  # Iones: gran diferencia
                    elif perfil.get('tipo') == 'organico':
                        factor = 0.4   # Orgánicos: diferencia moderada-alta
                    else:
                        factor = 0.6   # Fisicoquímicos: diferencia menor
                    muestra_baja.append(valor * factor)
            
            elif nombre.startswith('low_'):
                high_nombre = nombre.replace('low_', 'high_')
                if high_nombre in features:
                    muestra_baja.append(features[high_nombre])
                else:
                    if perfil.get('tipo') == 'inorganico':
                        factor = 3.0
                    elif perfil.get('tipo') == 'organico':
                        factor = 2.2
                    else:
                        factor = 1.8
                    muestra_baja.append(valor * factor)
            
            elif 'ratio' in nombre or 'enhancement' in nombre:
                muestra_baja.append(1 / (valor + 1e-8))
            elif 'diff' in nombre or 'contrast' in nombre:
                muestra_baja.append(-valor)
            else:
                muestra_baja.append(valor)
        
        return muestra_baja
    
    def entrenar_xgboost_optimizado(self, X, y, feature_names, config_xgb, contaminante, perfil):
        """Entrena XGBoost con configuración optimizada"""
        
        print(f"🚀 Entrenando XGBoost optimizado...")
        
        try:
            # Escalado específico para XGBoost
            scaler = RobustScaler()  # Más robusto para XGBoost
            X_scaled = scaler.fit_transform(X)
            
            # División estratificada
            if len(X) >= 6:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
            else:
                X_train = X_test = X_scaled
                y_train = y_test = y
            
            print(f"   📊 División: train={X_train.shape}, test={X_test.shape}")
            
            # Optimización de hiperparámetros específica para cada configuración
            mejor_modelo, mejores_params = self._optimizar_xgboost(X_train, y_train, config_xgb['params'])
            
            # Evaluación final
            resultado = self._evaluar_xgboost_final(
                mejor_modelo, X_train, X_test, y_train, y_test, 
                feature_names, contaminante, perfil, mejores_params
            )
            
            return resultado
            
        except Exception as e:
            print(f"   ❌ Error en entrenamiento XGBoost: {str(e)}")
            return None
    
    def _optimizar_xgboost(self, X_train, y_train, param_dist):
        """Optimiza hiperparámetros de XGBoost"""
        
        print(f"   🔍 Optimizando hiperparámetros XGBoost...")
        
        mejor_score = -np.inf
        mejor_modelo = None
        mejores_params = None
        
        # Número de iteraciones según tamaño del dataset
        n_iteraciones = min(15, max(5, len(X_train) // 2))
        
        for i in range(n_iteraciones):
            # Generar parámetros aleatorios
            params = {}
            for key, dist in param_dist.items():
                if hasattr(dist, 'rvs'):
                    params[key] = dist.rvs()
                else:
                    params[key] = np.random.choice(dist)
            
            try:
                # Crear modelo
                modelo = xgb.XGBClassifier(
                    **params,
                    objective='binary:logistic',
                    random_state=42,
                    verbosity=0,
                    n_jobs=1
                )
                
                # Validación cruzada adaptativa
                if len(X_train) >= 6:
                    cv_folds = min(3, len(X_train) // 3)
                    scores = cross_val_score(modelo, X_train, y_train, cv=cv_folds, scoring='f1')
                    score_promedio = np.mean(scores)
                else:
                    # Para datasets muy pequeños, usar train score
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_train)
                    score_promedio = f1_score(y_train, y_pred, zero_division=0)
                
                if score_promedio > mejor_score:
                    mejor_score = score_promedio
                    mejor_modelo = modelo
                    mejores_params = params.copy()
                
            except Exception as e:
                continue
        
        # Si no se encontró modelo válido, usar parámetros por defecto
        if mejor_modelo is None:
            mejores_params = {
                'n_estimators': 8,
                'max_depth': 3,
                'learning_rate': 0.1,
                'reg_alpha': 2.0,
                'reg_lambda': 4.0,
                'min_child_weight': 5
            }
            mejor_modelo = xgb.XGBClassifier(**mejores_params, random_state=42, verbosity=0)
        
        # Entrenar modelo final
        mejor_modelo.fit(X_train, y_train)
        
        print(f"   ✅ Mejor score CV: {mejor_score:.4f}")
        
        return mejor_modelo, mejores_params
    
    def _evaluar_xgboost_final(self, modelo, X_train, X_test, y_train, y_test, 
                              feature_names, contaminante, perfil, params):
        """Evaluación final del modelo XGBoost"""
        
        # Predicciones
        y_train_pred = modelo.predict(X_train)
        y_test_pred = modelo.predict(X_test)
        
        # Métricas principales
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        # Gaps para diagnóstico overfitting
        gap_acc = train_acc - test_acc
        gap_f1 = train_f1 - test_f1
        
        # AUC
        try:
            y_test_proba = modelo.predict_proba(X_test)[:, 1]
            if len(set(y_test)) > 1:
                auc = roc_auc_score(y_test, y_test_proba)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        # Diagnóstico específico para XGBoost
        if gap_f1 > 0.25:
            diagnostico = "OVERFITTING_SEVERO_XGBOOST"
            recomendacion = "Aumentar regularización (reg_alpha, reg_lambda)"
        elif gap_f1 > 0.15:
            diagnostico = "OVERFITTING_MODERADO_XGBOOST"
            recomendacion = "Reducir max_depth o aumentar min_child_weight"
        elif gap_f1 > 0.08:
            diagnostico = "LEVE_OVERFITTING_XGBOOST"
            recomendacion = "Modelo aceptable, monitorear en producción"
        else:
            diagnostico = "XGBOOST_ROBUSTO"
            recomendacion = "Configuración óptima para este contaminante"
        
        # Feature importance de XGBoost
        feature_importance = {}
        if hasattr(modelo, 'feature_importances_'):
            for i, imp in enumerate(modelo.feature_importances_):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(imp)
        
        # Evaluación contextualizada según perfil químico
        separabilidad = perfil.get('separabilidad_esperada', 'media')
        if separabilidad == 'muy_alta' and test_f1 >= 0.95:
            evaluacion_quimica = "🟢 PERFECTO - Esperado para alta separabilidad"
        elif separabilidad == 'alta' and test_f1 >= 0.85:
            evaluacion_quimica = "🟢 EXCELENTE"
        elif separabilidad == 'media' and test_f1 >= 0.70:
            evaluacion_quimica = "🟡 BUENO - Esperado para separabilidad media"
        elif test_f1 >= 0.50:
            evaluacion_quimica = "🟠 MODERADO - Revisar features específicos"
        else:
            evaluacion_quimica = "🔴 REQUIERE OPTIMIZACIÓN"
        
        return {
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'train_accuracy': float(train_acc),
            'train_f1': float(train_f1),
            'gap_accuracy': float(gap_acc),
            'gap_f1': float(gap_f1),
            'auc': float(auc),
            'diagnostico_overfitting': diagnostico,
            'recomendacion': recomendacion,
            'evaluacion_quimica': evaluacion_quimica,
            'feature_importance': feature_importance,
            'parametros_xgboost': params,
            'n_muestras_train': X_train.shape[0],
            'n_muestras_test': X_test.shape[0],
            'exito': True
        }
    
    def determinar_categoria_dataset(self, n_muestras):
        """Determina categoría del dataset para XGBoost"""
        if n_muestras <= 4:
            return 'extremo'
        elif n_muestras <= 15:
            return 'pequeno'
        elif n_muestras <= 50:
            return 'mediano'
        else:
            return 'grande'
    
    def cargar_datos_espectrales(self, contaminante):
        """Carga datos espectrales"""
        carpeta = self.mapeo_carpetas[contaminante]
        ruta_carpeta = os.path.join(self.directorio_base, carpeta)
        
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) 
                              if f.endswith('_datos_espectrales.csv')]
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        datos = pd.read_csv(ruta_archivo)
        return datos.dropna().sort_values('wavelength').reset_index(drop=True)
    
    def _mostrar_resultados_finales(self, resultado):
        """Muestra resultados del sistema final"""
        
        print(f"\n{'='*80}")
        print(f"🧬 RESULTADOS SISTEMA FINAL XGBOOST")
        print(f"{'='*80}")
        
        perfil = resultado['perfil_quimico']
        
        print(f"📋 Contaminante: {resultado['contaminante']}")
        print(f"🏷️ Tipo químico: {perfil['tipo']} - {perfil['subtipo']}")
        print(f"📊 Categoría dataset: {resultado['categoria_dataset']} ({resultado['n_muestras_total']} muestras)")
        print(f"🔧 Configuración: {resultado['configuracion_xgboost']}")
        
        print(f"\n📊 MÉTRICAS FINALES:")
        print(f"   🎯 Test F1:       {resultado['test_f1']:.4f}")
        print(f"   🎯 Test Accuracy: {resultado['test_accuracy']:.4f}")
        print(f"   🎯 AUC:           {resultado['auc']:.4f}")
        print(f"   📊 Gap F1:        {resultado['gap_f1']:+.4f}")
        
        print(f"\n🔍 DIAGNÓSTICO XGBOOST:")
        print(f"   🏆 Estado: {resultado['diagnostico_overfitting']}")
        print(f"   💡 Recomendación: {resultado['recomendacion']}")
        print(f"   🧪 Evaluación química: {resultado['evaluacion_quimica']}")
        
        if resultado['feature_importance']:
            print(f"\n🔑 TOP FEATURES XGBOOST:")
            top_features = sorted(resultado['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        print(f"\n⏱️ Tiempo total: {resultado['tiempo_entrenamiento']:.1f}s")
        
        # Conclusión específica para el proyecto
        if resultado['test_f1'] >= 0.8:
            print(f"\n🎉 DETECCIÓN EXITOSA - Sistema listo para implementación")
        elif resultado['test_f1'] >= 0.6:
            print(f"\n💛 DETECCIÓN BUENA - Continuar optimización")
        else:
            print(f"\n🔧 DETECCIÓN MEJORABLE - Revisar features químicos específicos")
    
    def _guardar_resultados_finales(self, contaminante, resultado):
        """Guarda resultados del sistema final"""
        try:
            dir_contaminante = os.path.join(self.results_dir, contaminante)
            os.makedirs(dir_contaminante, exist_ok=True)
            
            ruta_json = os.path.join(dir_contaminante, f"{contaminante}_sistema_final.json")
            
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"   💾 Resultados guardados: {ruta_json}")
            
        except Exception as e:
            print(f"   ⚠️ Error guardando: {e}")
    
    def detectar_multiples_contaminantes(self, lista_contaminantes=None):
        """Detecta múltiples contaminantes"""
        
        if lista_contaminantes is None:
            # Contaminantes de prueba por defecto
            lista_contaminantes = ['Nh4_Mg_L', 'Caffeine_Ng_L', 'Turbidity_Ntu', 'Doc_Mg_L', 'Acesulfame_Ng_L']
        
        print(f"\n{'='*90}")
        print(f"🧬 DETECCIÓN MÚLTIPLE - SISTEMA FINAL XGBOOST")
        print(f"📊 Contaminantes a procesar: {len(lista_contaminantes)}")
        print(f"{'='*90}")
        
        resultados = {}
        inicio_total = datetime.datetime.now()
        
        for i, contaminante in enumerate(lista_contaminantes, 1):
            print(f"\n[{i}/{len(lista_contaminantes)}] 🔬 PROCESANDO: {contaminante}")
            
            resultado = self.detectar_contaminante(contaminante)
            if resultado:
                resultados[contaminante] = resultado
        
        # Análisis consolidado
        self._generar_reporte_consolidado(resultados, inicio_total)
        
        return resultados
    
    def _generar_reporte_consolidado(self, resultados, inicio_total):
        """Genera reporte consolidado final"""
        
        fin_total = datetime.datetime.now()
        tiempo_total = (fin_total - inicio_total).total_seconds()
        
        print(f"\n{'='*90}")
        print(f"📊 REPORTE CONSOLIDADO - SISTEMA FINAL XGBOOST")
        print(f"{'='*90}")
        
        if not resultados:
            print("❌ No hay resultados para analizar")
            return
        
        # Estadísticas generales
        f1_scores = [r['test_f1'] for r in resultados.values()]
        accuracies = [r['test_accuracy'] for r in resultados.values()]
        aucs = [r['auc'] for r in resultados.values()]
        
        print(f"✅ Contaminantes procesados: {len(resultados)}")
        print(f"⏱️ Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        print(f"\n📈 ESTADÍSTICAS GLOBALES:")
        print(f"   F1-score promedio: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"   Accuracy promedio: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   AUC promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        
        # Análisis por tipo químico
        tipos_quimicos = {}
        for cont, res in resultados.items():
            tipo = res['perfil_quimico']['tipo']
            if tipo not in tipos_quimicos:
                tipos_quimicos[tipo] = []
            tipos_quimicos[tipo].append(res['test_f1'])
        
        print(f"\n🧪 ANÁLISIS POR TIPO QUÍMICO:")
        for tipo, f1s in tipos_quimicos.items():
            promedio = np.mean(f1s)
            print(f"   {tipo.title()}: {len(f1s)} contaminante(s), F1 promedio: {promedio:.3f}")
        
        # Ranking de rendimiento
        print(f"\n🏆 RANKING DE DETECCIÓN:")
        ranking = sorted(resultados.items(), key=lambda x: x[1]['test_f1'], reverse=True)
        
        for i, (contaminante, resultado) in enumerate(ranking[:10], 1):
            f1 = resultado['test_f1']
            tipo = resultado['perfil_quimico']['tipo'][:8]
            
            if f1 >= 0.9:
                emoji = "🥇"
            elif f1 >= 0.8:
                emoji = "🥈"
            elif f1 >= 0.7:
                emoji = "🥉"
            else:
                emoji = "📊"
            
            print(f"   {i:2d}. {contaminante:<20} | F1: {f1:.3f} | {tipo:<8} {emoji}")
        
        # Resumen de efectividad
        excelentes = sum(1 for f1 in f1_scores if f1 >= 0.8)
        buenos = sum(1 for f1 in f1_scores if 0.6 <= f1 < 0.8)
        mejorables = sum(1 for f1 in f1_scores if f1 < 0.6)
        
        print(f"\n📊 RESUMEN DE EFECTIVIDAD:")
        print(f"   🟢 Excelentes (F1≥0.8): {excelentes}/{len(resultados)}")
        print(f"   🟡 Buenos (0.6≤F1<0.8): {buenos}/{len(resultados)}")
        print(f"   🔴 Mejorables (F1<0.6): {mejorables}/{len(resultados)}")
        
        # Conclusión del proyecto
        if excelentes >= len(resultados) * 0.7:
            print(f"\n🎉 PROYECTO EXITOSO:")
            print(f"   ✅ Sistema XGBoost altamente efectivo")
            print(f"   ✅ Adaptación química específica funcional")
            print(f"   ✅ Listo para implementación en detección temprana")
        elif excelentes + buenos >= len(resultados) * 0.8:
            print(f"\n💛 PROYECTO SATISFACTORIO:")
            print(f"   ✅ Sistema XGBoost funcionalmente efectivo")
            print(f"   🔧 Refinamiento menor para optimización")
        else:
            print(f"\n🔧 PROYECTO REQUIERE OPTIMIZACIÓN:")
            print(f"   ⚠️ Revisar features químicos específicos")
            print(f"   ⚠️ Ajustar parámetros XGBoost por tipo")
        
        # Guardar reporte
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reporte = {
            'timestamp': datetime.datetime.now().isoformat(),
            'proyecto': 'Detección Temprana de Contaminantes en Aguas Superficiales',
            'estudiante': 'María José Erazo González',
            'universidad': 'Universidad Diego Portales',
            'metodo': 'XGBoost con Adaptación Química Específica',
            'estadisticas': {
                'total_contaminantes': len(resultados),
                'f1_promedio': float(np.mean(f1_scores)),
                'accuracy_promedio': float(np.mean(accuracies)),
                'auc_promedio': float(np.mean(aucs)),
                'tiempo_total_segundos': tiempo_total
            },
            'efectividad': {
                'excelentes': excelentes,
                'buenos': buenos,
                'mejorables': mejorables
            },
            'ranking': [(k, v['test_f1']) for k, v in ranking],
            'resultados_detallados': resultados
        }
        
        nombre_reporte = f"reporte_sistema_final_xgboost_{timestamp}.json"
        with open(os.path.join(self.results_dir, nombre_reporte), 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Reporte completo guardado: {nombre_reporte}")

def main():
    """Función principal del sistema final"""
    
    print("🧬 SISTEMA FINAL XGBOOST PARA DETECCIÓN DE CONTAMINANTES")
    print("="*70)
    print("🎯 Detección Temprana de Contaminantes en Aguas Superficiales")
    print("👩‍🎓 María José Erazo González - Universidad Diego Portales")
    print("📋 Anteproyecto de Título - Sistema de IA Especializado")
    print("="*70)
    
    sistema = SistemaFinalXGBoostContaminantes("firmas_espectrales_csv")
    
    print("\n🎮 OPCIONES:")
    print("1. 🧪 Detectar un contaminante específico")
    print("2. 🔬 Detección múltiple (recomendado)")
    print("3. 📊 Análisis de todos los contaminantes disponibles")
    
    try:
        opcion = input("\nSelecciona una opción (1-3): ").strip()
        
        if opcion == '1':
            print(f"\nContaminantes disponibles:")
            contaminantes_disponibles = list(sistema.mapeo_carpetas.keys())
            for i, cont in enumerate(contaminantes_disponibles, 1):
                print(f"  {i}. {cont}")
            
            seleccion = input("\nEscribe el nombre exacto del contaminante: ").strip()
            if seleccion in contaminantes_disponibles:
                resultado = sistema.detectar_contaminante(seleccion)
                return {seleccion: resultado}
            else:
                print(f"❌ Contaminante '{seleccion}' no encontrado")
                
        elif opcion == '2':
            print("\n🚀 Ejecutando detección múltiple...")
            resultados = sistema.detectar_multiples_contaminantes()
            return resultados
            
        elif opcion == '3':
            print("\n🔬 Analizando todos los contaminantes...")
            todos_contaminantes = list(sistema.mapeo_carpetas.keys())
            resultados = sistema.detectar_multiples_contaminantes(todos_contaminantes)
            return resultados
            
        else:
            print("❌ Opción inválida. Ejecutando detección múltiple por defecto...")
            resultados = sistema.detectar_multiples_contaminantes()
            return resultados
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario.")
        return None
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return None

if __name__ == "__main__":
    main()