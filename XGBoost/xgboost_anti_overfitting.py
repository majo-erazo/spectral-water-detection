# entrenamiento_adaptativo_contaminantes.py
# Sistema de entrenamiento personalizado según características químicas y espectrales
# María José Erazo González - Proyecto UDP

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from scipy.stats import randint, uniform
from scipy import signal

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from numpy import trapz

class EntrenadorAdaptativoContaminantes:
    """
    Sistema de entrenamiento que adapta automáticamente la estrategia según 
    las características químicas y espectrales específicas de cada contaminante
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.results_dir = "resultados_adaptativos_por_contaminante"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Base de conocimiento química y espectral
        self.perfiles_contaminantes = self._inicializar_perfiles_contaminantes()
        
        # Mapeo estándar del proyecto
        self.mapeo_carpetas = {
            'Doc_Mg_L': 'Doc', 'Nh4_Mg_L': 'Nh4', 'Turbidity_Ntu': 'Turbidity',
            'Caffeine_Ng_L': 'Caffeine', 'Acesulfame_Ng_L': 'Acesulfame',
            '4-&5-Methylbenzotriazole_Ng_L': 'Methylbenzotriazole',
            '6Ppd-Quinone_Ng_L': 'Quinone', '13-Diphenylguanidine_Ng_L': 'Diphenylguanidine',
            'Benzotriazole_Ng_L': 'Benzotriazole', 'Candesartan_Ng_L': 'Candesartan',
            'Citalopram_Ng_L': 'Citalopram', 'Cyclamate_Ng_L': 'Cyclamate',
            'Deet_Ng_L': 'Deet', 'Diclofenac_Ng_L': 'Diclofenac',
            'Diuron_Ng_L': 'Diuron', 'Hmmm_Ng_L': 'Hmmm',
            'Hydrochlorthiazide_Ng_L': 'Hydrochlorthiazide', 'Mecoprop_Ng_L': 'Mecoprop',
            'Nsol_Mg_L': 'Nsol', 'Oit_Ng_L': 'Oit', 'Po4_Mg_L': 'Po4', 'So4_Mg_L': 'So4'
        }
    
    def _inicializar_perfiles_contaminantes(self):
        """
        Base de conocimiento con características químicas y espectrales específicas
        """
        return {
            # INORGÁNICOS: Iones simples con absorción específica en UV
            'Nh4_Mg_L': {
                'tipo': 'inorganico',
                'categoria_quimica': 'ion_amonio',
                'regiones_criticas': [(200, 280)],  # UV fuerte
                'features_prioritarios': ['uv_absorption', 'slope_uv', 'peak_uv'],
                'model_complexity': 'minimo',  # Ion simple
                'features_especificos': ['nh4_specific'],
                'validation_strategy': 'leave_one_out',
                'xgb_params': {
                    'n_estimators': randint(3, 8),
                    'max_depth': randint(2, 3),
                    'learning_rate': uniform(0.1, 0.2),
                    'reg_alpha': uniform(5.0, 10.0),
                    'reg_lambda': uniform(8.0, 15.0),
                    'min_child_weight': randint(10, 20)
                }
            },
            
            'Po4_Mg_L': {
                'tipo': 'inorganico',
                'categoria_quimica': 'ion_fosfato',
                'regiones_criticas': [(200, 250)],  # UV extremo
                'features_prioritarios': ['uv_extreme', 'mean_uv', 'auc_uv'],
                'model_complexity': 'minimo',
                'validation_strategy': 'leave_one_out',
                'xgb_params': {
                    'n_estimators': randint(3, 6),
                    'max_depth': randint(2, 3),
                    'learning_rate': uniform(0.15, 0.25),
                    'reg_alpha': uniform(6.0, 12.0),
                    'reg_lambda': uniform(10.0, 18.0)
                }
            },
            
            'So4_Mg_L': {
                'tipo': 'inorganico',
                'categoria_quimica': 'ion_sulfato',
                'regiones_criticas': [(200, 300)],
                'features_prioritarios': ['broad_uv', 'slope_global'],
                'model_complexity': 'minimo',
                'validation_strategy': 'leave_one_out',
                'xgb_params': {
                    'n_estimators': randint(3, 7),
                    'max_depth': randint(2, 3),
                    'learning_rate': uniform(0.1, 0.2),
                    'reg_alpha': uniform(4.0, 8.0),
                    'reg_lambda': uniform(6.0, 12.0)
                }
            },
            
            # ORGÁNICOS SIMPLES: Compuestos con cromóforos específicos
            'Caffeine_Ng_L': {
                'tipo': 'organico',
                'categoria_quimica': 'xantina_alcaloide',
                'regiones_criticas': [(250, 300), (350, 400)],  # Dos picos característicos
                'features_prioritarios': ['peak_analysis', 'dual_absorption', 'ratio_regions'],
                'model_complexity': 'bajo',
                'features_especificos': ['caffeine_signature'],
                'validation_strategy': 'stratified_cv',
                'xgb_params': {
                    'n_estimators': randint(5, 12),
                    'max_depth': randint(3, 4),
                    'learning_rate': uniform(0.08, 0.15),
                    'reg_alpha': uniform(2.0, 5.0),
                    'reg_lambda': uniform(3.0, 8.0),
                    'min_child_weight': randint(5, 12)
                }
            },
            
            'Acesulfame_Ng_L': {
                'tipo': 'organico',
                'categoria_quimica': 'edulcorante_artificial',
                'regiones_criticas': [(220, 280)],  # Absorción característica UV
                'features_prioritarios': ['uv_signature', 'peak_specific', 'narrow_absorption'],
                'model_complexity': 'bajo',
                'validation_strategy': 'holdout_robusto',
                'xgb_params': {
                    'n_estimators': randint(4, 10),
                    'max_depth': randint(2, 3),
                    'learning_rate': uniform(0.1, 0.18),
                    'reg_alpha': uniform(3.0, 7.0),
                    'reg_lambda': uniform(4.0, 9.0)
                }
            },
            
            # COMPUESTOS FARMACÉUTICOS: Estructuras complejas
            'Candesartan_Ng_L': {
                'tipo': 'organico',
                'categoria_quimica': 'farmaceutico_complejo',
                'regiones_criticas': [(250, 350), (400, 500)],
                'features_prioritarios': ['multi_peak', 'complex_shape', 'derivative_features'],
                'model_complexity': 'medio',
                'validation_strategy': 'stratified_cv',
                'xgb_params': {
                    'n_estimators': randint(8, 20),
                    'max_depth': randint(3, 5),
                    'learning_rate': uniform(0.05, 0.12),
                    'reg_alpha': uniform(1.5, 4.0),
                    'reg_lambda': uniform(2.0, 6.0),
                    'min_child_weight': randint(3, 8)
                }
            },
            
            'Diclofenac_Ng_L': {
                'tipo': 'organico',
                'categoria_quimica': 'farmaceutico_antiinflamatorio',
                'regiones_criticas': [(275, 285), (320, 330)],  # Picos muy específicos
                'features_prioritarios': ['precise_peaks', 'narrow_bands', 'ratio_analysis'],
                'model_complexity': 'medio',
                'validation_strategy': 'stratified_cv',
                'xgb_params': {
                    'n_estimators': randint(6, 15),
                    'max_depth': randint(3, 4),
                    'learning_rate': uniform(0.06, 0.14),
                    'reg_alpha': uniform(2.0, 5.0),
                    'reg_lambda': uniform(3.0, 7.0)
                }
            },
            
            # PARÁMETROS FÍSICO-QUÍMICOS: Comportamiento diferente
            'Turbidity_Ntu': {
                'tipo': 'fisicoquimico',
                'categoria_quimica': 'dispersion_particulas',
                'regiones_criticas': [(400, 800)],  # Todo el visible
                'features_prioritarios': ['broadband_scattering', 'slope_analysis', 'uniform_response'],
                'model_complexity': 'medio',
                'features_especificos': ['scattering_pattern'],
                'validation_strategy': 'stratified_cv',
                'xgb_params': {
                    'n_estimators': randint(10, 25),
                    'max_depth': randint(4, 6),
                    'learning_rate': uniform(0.04, 0.10),
                    'reg_alpha': uniform(1.0, 3.0),
                    'reg_lambda': uniform(1.5, 4.0),
                    'min_child_weight': randint(2, 6)
                }
            },
            
            'Doc_Mg_L': {
                'tipo': 'fisicoquimico',
                'categoria_quimica': 'carbono_organico_disuelto',
                'regiones_criticas': [(200, 400), (450, 600)],
                'features_prioritarios': ['broad_organic', 'multiple_chromophores', 'integration'],
                'model_complexity': 'alto',  # Mezcla compleja
                'validation_strategy': 'stratified_cv',
                'xgb_params': {
                    'n_estimators': randint(15, 35),
                    'max_depth': randint(4, 7),
                    'learning_rate': uniform(0.03, 0.08),
                    'reg_alpha': uniform(0.5, 2.0),
                    'reg_lambda': uniform(1.0, 3.0),
                    'min_child_weight': randint(1, 4)
                }
            }
        }
    
    def extraer_features_adaptativos(self, datos, contaminante):
        """
        Extrae features específicos según el tipo de contaminante
        """
        perfil = self.perfiles_contaminantes.get(contaminante)
        if not perfil:
            # Contaminante no catalogado, usar features genéricos
            return self._extraer_features_genericos(datos)
        
        print(f"🧬 Extrayendo features para {perfil['categoria_quimica']}")
        
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        features = {}
        
        # 1. FEATURES BÁSICOS UNIVERSALES
        for conc, response in [('high', high_response), ('low', low_response)]:
            features[f'{conc}_mean'] = np.mean(response)
            features[f'{conc}_std'] = np.std(response)
            features[f'{conc}_max'] = np.max(response)
            features[f'{conc}_min'] = np.min(response)
        
        # 2. FEATURES ESPECÍFICOS POR TIPO
        if perfil['tipo'] == 'inorganico':
            features.update(self._features_inorganicos(wavelengths, high_response, low_response, perfil))
        
        elif perfil['tipo'] == 'organico':
            features.update(self._features_organicos(wavelengths, high_response, low_response, perfil))
        
        elif perfil['tipo'] == 'fisicoquimico':
            features.update(self._features_fisicoquimicos(wavelengths, high_response, low_response, perfil))
        
        # 3. FEATURES DE REGIONES CRÍTICAS
        for region_start, region_end in perfil['regiones_criticas']:
            mask = (wavelengths >= region_start) & (wavelengths <= region_end)
            if np.any(mask):
                region_name = f"region_{region_start}_{region_end}"
                features[f'{region_name}_high_mean'] = np.mean(high_response[mask])
                features[f'{region_name}_low_mean'] = np.mean(low_response[mask])
                features[f'{region_name}_ratio'] = (features[f'{region_name}_high_mean'] / 
                                                  (features[f'{region_name}_low_mean'] + 1e-8))
        
        # 4. RATIOS Y COMPARACIONES
        features['ratio_global'] = features['high_mean'] / (features['low_mean'] + 1e-8)
        features['diff_global'] = features['high_mean'] - features['low_mean']
        
        print(f"   ✅ Features adaptativos: {len(features)} específicos para {perfil['categoria_quimica']}")
        
        return features
    
    def _features_inorganicos(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para contaminantes inorgánicos (iones)"""
        features = {}
        
        # Los iones tienen absorción fuerte en UV
        uv_mask = wavelengths <= 300
        if np.any(uv_mask):
            features['uv_intensity_high'] = np.mean(high_response[uv_mask])
            features['uv_intensity_low'] = np.mean(low_response[uv_mask])
            features['uv_enhancement'] = features['uv_intensity_high'] / (features['uv_intensity_low'] + 1e-8)
            
            # Pendiente en UV (característica de iones)
            if np.sum(uv_mask) > 2:
                uv_slope_high, _ = np.polyfit(wavelengths[uv_mask], high_response[uv_mask], 1)
                uv_slope_low, _ = np.polyfit(wavelengths[uv_mask], low_response[uv_mask], 1)
                features['uv_slope_high'] = uv_slope_high
                features['uv_slope_low'] = uv_slope_low
        
        # Características específicas según ion
        if 'nh4' in perfil['categoria_quimica'].lower():
            # NH4+ tiene absorción muy característica
            extreme_uv = wavelengths <= 250
            if np.any(extreme_uv):
                features['nh4_signature'] = np.max(high_response[extreme_uv]) - np.max(low_response[extreme_uv])
        
        return features
    
    def _features_organicos(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para compuestos orgánicos"""
        features = {}
        
        # Análisis de picos (importante para orgánicos)
        peaks_high, _ = signal.find_peaks(high_response, height=np.percentile(high_response, 60))
        peaks_low, _ = signal.find_peaks(low_response, height=np.percentile(low_response, 60))
        
        features['n_peaks_high'] = len(peaks_high)
        features['n_peaks_low'] = len(peaks_low)
        features['peak_difference'] = features['n_peaks_high'] - features['n_peaks_low']
        
        if len(peaks_high) > 0:
            features['main_peak_high'] = np.max(high_response[peaks_high])
            features['main_peak_wavelength'] = wavelengths[peaks_high[np.argmax(high_response[peaks_high])]]
        
        # Cromóforos orgánicos (250-400 nm típicamente)
        chromophore_mask = (wavelengths >= 250) & (wavelengths <= 400)
        if np.any(chromophore_mask):
            features['chromophore_area_high'] = trapz(high_response[chromophore_mask], wavelengths[chromophore_mask])
            features['chromophore_area_low'] = trapz(low_response[chromophore_mask], wavelengths[chromophore_mask])
            features['chromophore_enhancement'] = (features['chromophore_area_high'] / 
                                                 (features['chromophore_area_low'] + 1e-8))
        
        # Features específicos según tipo orgánico
        if 'caffeine' in perfil['categoria_quimica'].lower():
            # Cafeína tiene dos regiones de absorción características
            peak1_mask = (wavelengths >= 260) & (wavelengths <= 280)
            peak2_mask = (wavelengths >= 350) & (wavelengths <= 380)
            
            if np.any(peak1_mask) and np.any(peak2_mask):
                peak1_high = np.mean(high_response[peak1_mask])
                peak2_high = np.mean(high_response[peak2_mask])
                features['caffeine_signature'] = peak1_high / (peak2_high + 1e-8)
        
        return features
    
    def _features_fisicoquimicos(self, wavelengths, high_response, low_response, perfil):
        """Features específicos para parámetros físico-químicos"""
        features = {}
        
        # Análisis de dispersión (para turbidez)
        if 'turbidity' in perfil['categoria_quimica'].lower():
            # La turbidez afecta todo el espectro visible uniformemente
            visible_mask = (wavelengths >= 400) & (wavelengths <= 700)
            if np.any(visible_mask):
                # Uniformidad de la respuesta (característica de dispersión)
                visible_std_high = np.std(high_response[visible_mask])
                visible_std_low = np.std(low_response[visible_mask])
                features['scattering_uniformity'] = visible_std_high / (visible_std_low + 1e-8)
                
                # Pendiente de dispersión (debería ser negativa para turbidez)
                if np.sum(visible_mask) > 2:
                    scatter_slope, _ = np.polyfit(wavelengths[visible_mask], high_response[visible_mask], 1)
                    features['scattering_slope'] = scatter_slope
        
        # DOC: múltiples cromóforos
        elif 'carbono' in perfil['categoria_quimica'].lower():
            # DOC tiene absorción amplia con múltiples componentes
            # Análisis por bandas espectrales
            for band_start, band_end, band_name in [(200, 300, 'uv'), (300, 400, 'uv_vis'), (400, 500, 'vis')]:
                band_mask = (wavelengths >= band_start) & (wavelengths <= band_end)
                if np.any(band_mask):
                    features[f'doc_{band_name}_high'] = trapz(high_response[band_mask], wavelengths[band_mask])
                    features[f'doc_{band_name}_low'] = trapz(low_response[band_mask], wavelengths[band_mask])
        
        return features
    
    def _extraer_features_genericos(self, datos):
        """Features genéricos para contaminantes no catalogados"""
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        features = {}
        
        for conc, response in [('high', high_response), ('low', low_response)]:
            features[f'{conc}_mean'] = np.mean(response)
            features[f'{conc}_std'] = np.std(response)
            features[f'{conc}_max'] = np.max(response)
            features[f'{conc}_min'] = np.min(response)
            features[f'{conc}_auc'] = trapz(response, wavelengths)
        
        features['ratio_global'] = features['high_mean'] / (features['low_mean'] + 1e-8)
        features['diff_global'] = features['high_mean'] - features['low_mean']
        
        return features
    
    def seleccionar_features_adaptativos(self, features_dict, contaminante):
        """
        Selecciona features según las prioridades del contaminante
        """
        perfil = self.perfiles_contaminantes.get(contaminante)
        if not perfil:
            # Sin perfil, usar selección genérica
            return list(features_dict.keys())[:8]
        
        # Priorizar features según el perfil
        features_prioritarios = []
        features_secundarios = []
        
        for feature_name in features_dict.keys():
            es_prioritario = False
            
            # Buscar features prioritarios definidos en el perfil
            for patron_prioritario in perfil['features_prioritarios']:
                if patron_prioritario.lower() in feature_name.lower():
                    features_prioritarios.append(feature_name)
                    es_prioritario = True
                    break
            
            if not es_prioritario:
                features_secundarios.append(feature_name)
        
        # Determinar número de features según complejidad del modelo
        complexity = perfil['model_complexity']
        if complexity == 'minimo':
            n_features = min(3, len(features_dict))
        elif complexity == 'bajo':
            n_features = min(5, len(features_dict))
        elif complexity == 'medio':
            n_features = min(8, len(features_dict))
        else:  # alto
            n_features = min(12, len(features_dict))
        
        # Combinar prioritarios + secundarios hasta el límite
        features_seleccionados = features_prioritarios[:n_features]
        if len(features_seleccionados) < n_features:
            features_restantes = n_features - len(features_seleccionados)
            features_seleccionados.extend(features_secundarios[:features_restantes])
        
        print(f"   🎯 Seleccionados {len(features_seleccionados)} features para {perfil['categoria_quimica']}")
        print(f"   📊 Prioritarios: {len(features_prioritarios)}, Secundarios: {len(features_secundarios)}")
        
        return features_seleccionados
    
    def entrenar_adaptativo(self, contaminante):
        """
        Entrenamiento completamente adaptado al contaminante específico
        """
        print(f"\n{'='*70}")
        print(f"🧬 ENTRENAMIENTO ADAPTATIVO ESPECÍFICO")
        print(f"📋 Contaminante: {contaminante}")
        
        perfil = self.perfiles_contaminantes.get(contaminante)
        if perfil:
            print(f"🏷️ Tipo: {perfil['tipo']} - {perfil['categoria_quimica']}")
            print(f"🎯 Complejidad: {perfil['model_complexity']}")
            print(f"🌈 Regiones críticas: {perfil['regiones_criticas']}")
        else:
            print(f"⚠️ Contaminante no catalogado - usando configuración genérica")
        
        print(f"{'='*70}")
        
        inicio_tiempo = datetime.datetime.now()
        
        try:
            # 1. Cargar datos
            datos_espectrales = self.cargar_firma_espectral(contaminante)
            
            # 2. Extraer features adaptativos
            features_adaptativos = self.extraer_features_adaptativos(datos_espectrales, contaminante)
            
            # 3. Crear dataset mínimo (sin over-augmentation)
            dataset = self.crear_dataset_adaptativo(features_adaptativos, contaminante)
            
            # 4. Seleccionar features específicos
            feature_columns = list(features_adaptativos.keys())
            features_seleccionados = self.seleccionar_features_adaptativos(features_adaptativos, contaminante)
            
            # 5. Preparar datos
            X = dataset[features_seleccionados].values
            y = dataset['label'].values
            
            print(f"📊 Dataset adaptativo: {X.shape}")
            print(f"🔢 Features específicos: {len(features_seleccionados)}")
            
            # 6. Estrategia de validación adaptativa
            resultados = self.validar_y_entrenar_adaptativo(X, y, features_seleccionados, contaminante)
            
            # 7. Guardar resultados
            self._guardar_resultados_adaptativos(contaminante, resultados)
            
            fin_tiempo = datetime.datetime.now()
            tiempo_total = (fin_tiempo - inicio_tiempo).total_seconds()
            resultados['tiempo_entrenamiento'] = tiempo_total
            
            # 8. Mostrar resultados
            self._mostrar_resultados_adaptativos(resultados)
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error en entrenamiento adaptativo: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def crear_dataset_adaptativo(self, features_base, contaminante):
        """
        Crea dataset según la estrategia específica del contaminante
        """
        perfil = self.perfiles_contaminantes.get(contaminante, {'model_complexity': 'medio'})
        
        # Estrategia según complejidad
        if perfil['model_complexity'] == 'minimo':
            factor_aug = 0  # Solo muestras originales
            variabilidad = 0.1  # Mínima variabilidad
        elif perfil['model_complexity'] == 'bajo':
            factor_aug = 1  # 1 muestra adicional por clase
            variabilidad = 0.2
        elif perfil['model_complexity'] == 'medio':
            factor_aug = 2  # 2 muestras adicionales por clase
            variabilidad = 0.3
        else:  # alto
            factor_aug = 3  # 3 muestras adicionales por clase
            variabilidad = 0.4
        
        print(f"📊 Estrategia dataset: {perfil['model_complexity']} (aug={factor_aug}, var=±{variabilidad*100:.0f}%)")
        
        samples = []
        labels = []
        
        # 1. Muestra original alta concentración
        samples.append(list(features_base.values()))
        labels.append(1)
        
        # 2. Muestra original baja concentración (invertida)
        features_low = self._invertir_features_contaminante(features_base, contaminante)
        samples.append(features_low)
        labels.append(0)
        
        # 3. Augmentation específico
        for i in range(factor_aug):
            # Alta concentración con variabilidad controlada
            aug_high = []
            for val in features_base.values():
                noise = np.random.normal(1.0, variabilidad)
                noise = max(0.5, min(2.0, noise))
                aug_high.append(val * noise)
            
            samples.append(aug_high)
            labels.append(1)
            
            # Baja concentración con variabilidad controlada
            aug_low = []
            for val in features_low:
                noise = np.random.normal(1.0, variabilidad)
                noise = max(0.5, min(2.0, noise))
                aug_low.append(val * noise)
            
            samples.append(aug_low)
            labels.append(0)
        
        # Crear DataFrame
        feature_names = list(features_base.keys())
        df = pd.DataFrame(samples, columns=feature_names)
        df['label'] = labels
        
        print(f"   ✅ Dataset creado: {df.shape}, distribución: {pd.Series(labels).value_counts().to_dict()}")
        
        return df
    
    def _invertir_features_contaminante(self, features_base, contaminante):
        """Invierte features considerando las características del contaminante"""
        
        features_invertidos = []
        perfil = self.perfiles_contaminantes.get(contaminante)
        
        for key, value in features_base.items():
            if key.startswith('high_'):
                # Buscar equivalente low_
                low_key = key.replace('high_', 'low_')
                if low_key in features_base:
                    features_invertidos.append(features_base[low_key])
                else:
                    # Factor de reducción según tipo de contaminante
                    if perfil and perfil['tipo'] == 'inorganico':
                        factor = 0.3  # Iones muestran gran diferencia
                    elif perfil and perfil['tipo'] == 'organico':
                        factor = 0.5  # Orgánicos diferencia moderada
                    else:
                        factor = 0.7  # Fisicoquímicos diferencia menor
                    features_invertidos.append(value * factor)
            
            elif key.startswith('low_'):
                # Buscar equivalente high_
                high_key = key.replace('low_', 'high_')
                if high_key in features_base:
                    features_invertidos.append(features_base[high_key])
                else:
                    # Factor de aumento según tipo
                    if perfil and perfil['tipo'] == 'inorganico':
                        factor = 2.5  # Gran aumento para iones
                    elif perfil and perfil['tipo'] == 'organico':
                        factor = 1.8  # Aumento moderado
                    else:
                        factor = 1.3  # Aumento menor
                    features_invertidos.append(value * factor)
            
            elif 'ratio' in key:
                features_invertidos.append(1 / (value + 1e-8))
            elif 'diff' in key:
                features_invertidos.append(-value)
            else:
                features_invertidos.append(value)
        
        return features_invertidos
    
    def validar_y_entrenar_adaptativo(self, X, y, features_seleccionados, contaminante):
        """
        Validación y entrenamiento según estrategia específica del contaminante
        """
        perfil = self.perfiles_contaminantes.get(contaminante)
        
        # Parámetros específicos del contaminante
        if perfil:
            param_dist = perfil['xgb_params']
            validation_strategy = perfil['validation_strategy']
        else:
            # Parámetros genéricos conservadores
            param_dist = {
                'n_estimators': randint(5, 15),
                'max_depth': randint(2, 4),
                'learning_rate': uniform(0.05, 0.15),
                'reg_alpha': uniform(2.0, 6.0),
                'reg_lambda': uniform(3.0, 8.0),
                'min_child_weight': randint(5, 12)
            }
            validation_strategy = 'holdout_robusto'
        
        print(f"🔍 Estrategia validación: {validation_strategy}")
        
        # Ejecutar validación según estrategia
        if validation_strategy == 'leave_one_out' and len(X) <= 10:
            return self._validacion_leave_one_out(X, y, param_dist, features_seleccionados, contaminante)
        elif validation_strategy == 'stratified_cv':
            return self._validacion_stratified_cv(X, y, param_dist, features_seleccionados, contaminante)
        else:  # holdout_robusto
            return self._validacion_holdout_robusto(X, y, param_dist, features_seleccionados, contaminante)
    
    def _validacion_holdout_robusto(self, X, y, param_dist, features, contaminante):
        """Validación holdout robusta para datasets pequeños"""
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        print(f"   📊 División: train={X_train.shape}, test={X_test.shape}")
        
        # Escalado robusto
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Buscar mejores parámetros con pocas iteraciones
        mejor_score = -np.inf
        mejores_params = None
        
        for i in range(8):  # Solo 8 iteraciones
            # Generar parámetros aleatorios
            params = {}
            for key, dist in param_dist.items():
                if hasattr(dist, 'rvs'):
                    params[key] = dist.rvs()
                else:
                    params[key] = np.random.choice(dist)
            
            # Entrenar y evaluar
            modelo = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
            modelo.fit(X_train_scaled, y_train)
            
            # Validación cruzada simple en train
            try:
                scores = cross_val_score(modelo, X_train_scaled, y_train, cv=2, scoring='f1')
                score_promedio = np.mean(scores)
                
                if score_promedio > mejor_score:
                    mejor_score = score_promedio
                    mejores_params = params.copy()
                    
            except:
                continue
        
        # Entrenar modelo final
        modelo_final = xgb.XGBClassifier(**mejores_params, random_state=42, verbosity=0)
        modelo_final.fit(X_train_scaled, y_train)
        
        # Evaluación final
        return self._evaluar_modelo_final(modelo_final, X_train_scaled, X_test_scaled, 
                                        y_train, y_test, features, contaminante, mejores_params)
    
    def _evaluar_modelo_final(self, modelo, X_train, X_test, y_train, y_test, features, contaminante, params):
        """Evaluación final del modelo adaptativo"""
        
        # Predicciones
        y_train_pred = modelo.predict(X_train)
        y_test_pred = modelo.predict(X_test)
        
        # Métricas
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        # Gaps
        gap_acc = train_acc - test_acc
        gap_f1 = train_f1 - test_f1
        
        # AUC si es posible
        try:
            y_test_proba = modelo.predict_proba(X_test)[:, 1]
            if len(set(y_test)) > 1:
                auc = roc_auc_score(y_test, y_test_proba)
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        # Diagnóstico adaptativo
        if gap_f1 > 0.2:
            diagnostico = "OVERFITTING_SEVERO"
        elif gap_f1 > 0.1:
            diagnostico = "OVERFITTING_MODERADO"
        elif gap_f1 > 0.05:
            diagnostico = "LEVE_OVERFITTING"
        else:
            diagnostico = "ROBUSTO"
        
        # Feature importance
        importance = {}
        if hasattr(modelo, 'feature_importances_'):
            for i, imp in enumerate(modelo.feature_importances_):
                if i < len(features):
                    importance[features[i]] = float(imp)
        
        perfil = self.perfiles_contaminantes.get(contaminante, {})
        
        return {
            'contaminante': contaminante,
            'metodo': 'entrenamiento_adaptativo',
            'tipo_quimico': perfil.get('tipo', 'desconocido'),
            'categoria_quimica': perfil.get('categoria_quimica', 'desconocido'),
            'complejidad_modelo': perfil.get('model_complexity', 'medio'),
            
            # Métricas principales
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'auc': float(auc),
            
            # Diagnóstico overfitting
            'train_accuracy': float(train_acc),
            'train_f1': float(train_f1),
            'gap_accuracy': float(gap_acc),
            'gap_f1': float(gap_f1),
            'diagnostico_overfitting': diagnostico,
            
            # Configuración específica
            'parametros_especificos': params,
            'features_utilizados': features,
            'n_features': len(features),
            'feature_importance': importance,
            'regiones_criticas': perfil.get('regiones_criticas', []),
            'estrategia_validacion': perfil.get('validation_strategy', 'holdout_robusto'),
            
            # Metadatos
            'n_muestras_train': X_train.shape[0],
            'n_muestras_test': X_test.shape[0]
        }
    
    def _mostrar_resultados_adaptativos(self, resultados):
        """Muestra resultados del entrenamiento adaptativo"""
        
        print(f"\n{'='*70}")
        print(f"🧬 RESULTADOS ENTRENAMIENTO ADAPTATIVO")
        print(f"{'='*70}")
        
        # Información del contaminante
        print(f"📋 Contaminante: {resultados['contaminante']}")
        print(f"🏷️ Tipo químico: {resultados['tipo_quimico']} - {resultados['categoria_quimica']}")
        print(f"🎯 Complejidad: {resultados['complejidad_modelo']}")
        print(f"🔍 Validación: {resultados['estrategia_validacion']}")
        
        # Métricas principales
        print(f"\n📊 MÉTRICAS:")
        print(f"   🎯 Test F1:       {resultados['test_f1']:.4f}")
        print(f"   🎯 Test Accuracy: {resultados['test_accuracy']:.4f}")
        print(f"   🎯 AUC:           {resultados['auc']:.4f}")
        
        # Diagnóstico
        diagnostico = resultados['diagnostico_overfitting']
        gap_f1 = resultados['gap_f1']
        
        emoji_diag = {
            'ROBUSTO': '✅',
            'LEVE_OVERFITTING': '💛', 
            'OVERFITTING_MODERADO': '⚠️',
            'OVERFITTING_SEVERO': '🚨'
        }.get(diagnostico, '❓')
        
        print(f"\n🔍 DIAGNÓSTICO:")
        print(f"   {emoji_diag} Estado: {diagnostico}")
        print(f"   📊 Gap F1: {gap_f1:+.4f}")
        print(f"   🔢 Features: {resultados['n_features']}")
        
        # Features importantes
        if resultados['feature_importance']:
            print(f"\n🔑 TOP FEATURES:")
            top_features = sorted(resultados['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        # Recomendaciones específicas
        if diagnostico in ['OVERFITTING_SEVERO', 'OVERFITTING_MODERADO']:
            print(f"\n💡 RECOMENDACIONES ESPECÍFICAS:")
            if resultados['tipo_quimico'] == 'inorganico':
                print(f"   - Los iones requieren features UV específicos")
                print(f"   - Considerar solo regiones 200-300 nm")
            elif resultados['tipo_quimico'] == 'organico':
                print(f"   - Enfocar en análisis de picos característicos")
                print(f"   - Usar features de cromóforos específicos")
            else:
                print(f"   - Parámetros físico-químicos necesitan features de dispersión")
        elif diagnostico == 'ROBUSTO':
            print(f"\n✅ MODELO ROBUSTO - Configuración óptima para este contaminante")
        
        print(f"\n⏱️ Tiempo: {resultados.get('tiempo_entrenamiento', 0):.1f}s")
    
    def cargar_firma_espectral(self, contaminante):
        """Carga firma espectral usando mapeo estándar"""
        carpeta = self.mapeo_carpetas[contaminante]
        ruta_carpeta = os.path.join(self.directorio_base, carpeta)
        
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) 
                              if f.endswith('_datos_espectrales.csv')]
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        datos = pd.read_csv(ruta_archivo)
        datos = datos.dropna().sort_values('wavelength').reset_index(drop=True)
        
        return datos
    
    def _guardar_resultados_adaptativos(self, contaminante, resultados):
        """Guarda resultados del entrenamiento adaptativo"""
        try:
            dir_contaminante = os.path.join(self.results_dir, contaminante)
            os.makedirs(dir_contaminante, exist_ok=True)
            
            ruta_json = os.path.join(dir_contaminante, f"{contaminante}_adaptativo.json")
            
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"   💾 Resultados guardados: {ruta_json}")
            
        except Exception as e:
            print(f"   ⚠️ Error guardando: {e}")

def main_entrenamiento_adaptativo():
    """Función principal para entrenamiento adaptativo"""
    
    print("🧬 SISTEMA DE ENTRENAMIENTO ADAPTATIVO POR CONTAMINANTE")
    print("="*65)
    print("🎯 Cada contaminante usa configuración específica según sus propiedades químicas")
    
    entrenador = EntrenadorAdaptativoContaminantes("todo/firmas_espectrales_csv")
    
    # Contaminantes de prueba con diferentes tipos
    contaminantes_prueba = [
        'Nh4_Mg_L',        # Inorgánico simple
        'Caffeine_Ng_L',   # Orgánico con picos característicos
        'Turbidity_Ntu',   # Físico-químico de dispersión
        'Candesartan_Ng_L' # Farmacéutico complejo
    ]
    
    resultados = {}
    
    for contaminante in contaminantes_prueba:
        print(f"\n🔄 Procesando: {contaminante}")
        resultado = entrenador.entrenar_adaptativo(contaminante)
        if resultado:
            resultados[contaminante] = resultado
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN COMPARATIVO ENTRENAMIENTO ADAPTATIVO")
    print(f"{'='*70}")
    
    for cont, res in resultados.items():
        if res:
            tipo = res['tipo_quimico']
            f1 = res['test_f1']
            diagnostico = res['diagnostico_overfitting']
            
            emoji = '✅' if diagnostico == 'ROBUSTO' else ('💛' if 'LEVE' in diagnostico else '🚨')
            
            print(f"{cont:20} | {tipo:12} | F1: {f1:.3f} | {emoji} {diagnostico}")
    
    return resultados

if __name__ == "__main__":
    main_entrenamiento_adaptativo()