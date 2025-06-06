# sistema_adaptativo_3_modelos.py
# Sistema Adaptativo Inteligente para Detección de Contaminantes
# Selección automática de modelos basada en características químico-espectrales
# María José Erazo González - UDP

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import signal
from scipy.integrate import trapezoid

class AnalizadorFirmasEspectrales:
    """
    Analizador automático de características químico-espectrales
    Determina automáticamente el tipo químico y propiedades de cada contaminante
    """
    
    def __init__(self):
        self.caracteristicas = {}
        
        # Umbrales para clasificación automática
        self.umbrales = {
            'separabilidad_alta': 15.0,     # % diferencia high vs low
            'separabilidad_media': 8.0,
            'complejidad_alta': 8,          # número de picos
            'complejidad_media': 4,
            'ruido_alto': 0.15,            # std/mean ratio
            'ruido_medio': 0.08
        }
        
        # Bandas espectrales importantes
        self.bandas = {
            'uv_extremo': (400, 450),
            'uv_cercano': (450, 500), 
            'visible_azul': (500, 550),
            'visible_verde': (550, 600),
            'visible_rojo': (600, 700),
            'nir_cercano': (700, 800)
        }
    
    def analizar_firma_espectral(self, datos, nombre_contaminante):
        """
        Análisis automático completo de una firma espectral
        
        Returns:
            dict: Características detectadas y clasificación química
        """
        print(f"\n🔬 Analizando firma espectral: {nombre_contaminante}")
        
        wavelengths = datos['wavelength'].values
        high_mean = datos['high_mean'].values
        low_mean = datos['low_mean'].values
        signature = datos['signature'].values
        high_std = datos['high_std'].values
        low_std = datos['low_std'].values
        
        # 1. Análisis de separabilidad
        separabilidad = self._calcular_separabilidad(high_mean, low_mean)
        
        # 2. Análisis de complejidad espectral
        complejidad = self._calcular_complejidad(signature)
        
        # 3. Análisis de ruido
        ruido = self._calcular_ruido(high_mean, low_mean, high_std, low_std)
        
        # 4. Análisis por bandas espectrales
        dominancia_bandas = self._analizar_bandas_espectrales(wavelengths, signature)
        
        # 5. Detección de picos característicos
        picos_info = self._detectar_picos_caracteristicos(wavelengths, signature)
        
        # 6. Clasificación química automática
        clasificacion = self._clasificar_quimicamente(nombre_contaminante, separabilidad, 
                                                    complejidad, dominancia_bandas, picos_info)
        
        # 7. Recomendación de modelos
        modelos_recomendados = self._recomendar_modelos(separabilidad, complejidad, ruido)
        
        caracteristicas = {
            'nombre': nombre_contaminante,
            'separabilidad_porcentaje': separabilidad,
            'complejidad_nivel': complejidad['nivel'],
            'complejidad_picos': complejidad['num_picos'],
            'ruido_nivel': ruido['nivel'],
            'ruido_ratio': ruido['ratio'],
            'banda_dominante': dominancia_bandas['dominante'],
            'bandas_activas': dominancia_bandas['activas'],
            'picos_principales': picos_info['principales'],
            'clasificacion_quimica': clasificacion,
            'modelos_recomendados': modelos_recomendados,
            'features_recomendados': self._recomendar_features(clasificacion, dominancia_bandas)
        }
        
        self._mostrar_analisis(caracteristicas)
        self.caracteristicas[nombre_contaminante] = caracteristicas
        
        return caracteristicas
    
    def _calcular_separabilidad(self, high_mean, low_mean):
        """Calcula la separabilidad entre concentraciones altas y bajas"""
        diff_mean = np.mean(np.abs(high_mean - low_mean))
        baseline = np.mean([np.mean(high_mean), np.mean(low_mean)])
        separabilidad = (diff_mean / baseline) * 100
        return separabilidad
    
    def _calcular_complejidad(self, signature):
        """Calcula la complejidad espectral basada en picos y variaciones"""
        # Detectar picos significativos
        threshold = np.std(signature) * 1.2
        picos_pos, _ = signal.find_peaks(signature, height=threshold)
        picos_neg, _ = signal.find_peaks(-signature, height=threshold)
        
        num_picos = len(picos_pos) + len(picos_neg)
        
        # Calcular variabilidad como medida de complejidad
        variabilidad = np.std(signature) / (np.abs(np.mean(signature)) + 1e-6)
        
        if num_picos >= self.umbrales['complejidad_alta']:
            nivel = 'alta'
        elif num_picos >= self.umbrales['complejidad_media']:
            nivel = 'media'
        else:
            nivel = 'baja'
        
        return {
            'num_picos': num_picos,
            'variabilidad': variabilidad,
            'nivel': nivel
        }
    
    def _calcular_ruido(self, high_mean, low_mean, high_std, low_std):
        """Calcula el nivel de ruido relativo"""
        signal_mean = np.mean([np.mean(high_mean), np.mean(low_mean)])
        noise_mean = np.mean([np.mean(high_std), np.mean(low_std)])
        
        snr_ratio = noise_mean / (signal_mean + 1e-6)
        
        if snr_ratio >= self.umbrales['ruido_alto']:
            nivel = 'alto'
        elif snr_ratio >= self.umbrales['ruido_medio']:
            nivel = 'medio'
        else:
            nivel = 'bajo'
        
        return {
            'ratio': snr_ratio,
            'nivel': nivel
        }
    
    def _analizar_bandas_espectrales(self, wavelengths, signature):
        """Analiza qué bandas espectrales son más activas"""
        bandas_energia = {}
        
        for nombre_banda, (wl_min, wl_max) in self.bandas.items():
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if np.any(mask):
                energia = np.sum(np.abs(signature[mask]))
                bandas_energia[nombre_banda] = energia
            else:
                bandas_energia[nombre_banda] = 0
        
        # Normalizar energías
        total_energia = sum(bandas_energia.values())
        if total_energia > 0:
            bandas_energia = {k: v/total_energia for k, v in bandas_energia.items()}
        
        # Encontrar banda dominante
        banda_dominante = max(bandas_energia.items(), key=lambda x: x[1])
        bandas_activas = [k for k, v in bandas_energia.items() if v > 0.1]
        
        return {
            'energias': bandas_energia,
            'dominante': banda_dominante[0],
            'dominante_valor': banda_dominante[1],
            'activas': bandas_activas
        }
    
    def _detectar_picos_caracteristicos(self, wavelengths, signature):
        """Detecta picos característicos importantes"""
        threshold = np.std(signature) * 1.5
        
        # Picos positivos (aumentos)
        picos_pos, props_pos = signal.find_peaks(signature, height=threshold, distance=10)
        
        # Picos negativos (disminuciones)  
        picos_neg, props_neg = signal.find_peaks(-signature, height=threshold, distance=10)
        
        # Compilar información de picos principales
        principales = []
        
        # Agregar picos positivos más significativos
        if len(picos_pos) > 0:
            for i, idx in enumerate(picos_pos):
                if i < 5:  # Top 5 picos
                    principales.append({
                        'wavelength': wavelengths[idx],
                        'intensidad': signature[idx],
                        'tipo': 'aumento'
                    })
        
        # Agregar picos negativos más significativos
        if len(picos_neg) > 0:
            for i, idx in enumerate(picos_neg):
                if i < 5:  # Top 5 picos
                    principales.append({
                        'wavelength': wavelengths[idx],
                        'intensidad': -signature[idx],
                        'tipo': 'disminucion'
                    })
        
        # Ordenar por intensidad absoluta
        principales.sort(key=lambda x: abs(x['intensidad']), reverse=True)
        
        return {
            'principales': principales[:10],  # Top 10
            'num_picos_pos': len(picos_pos),
            'num_picos_neg': len(picos_neg)
        }
    
    def _clasificar_quimicamente(self, nombre, separabilidad, complejidad, bandas, picos):
        """Clasificación química automática basada en características"""
        nombre_lower = nombre.lower()
        
        # Clasificación basada en nombre (conocimiento previo)
        if any(x in nombre_lower for x in ['doc', 'turbidity', 'tss']):
            tipo_base = 'fisicoquimico'
        elif any(x in nombre_lower for x in ['nh4', 'po4', 'so4', 'nsol']):
            tipo_base = 'inorganico'
        else:
            tipo_base = 'organico'
        
        # Refinar clasificación con características espectrales
        subtipo = self._determinar_subtipo_espectral(
            tipo_base, nombre_lower, separabilidad, complejidad, bandas, picos
        )
        
        return {
            'tipo': tipo_base,
            'subtipo': subtipo,
            'confianza': self._calcular_confianza_clasificacion(separabilidad, complejidad)
        }
    
    def _determinar_subtipo_espectral(self, tipo_base, nombre, sep, comp, bandas, picos):
        """Determina subtipo específico basado en características espectrales"""
        
        if tipo_base == 'organico':
            if 'caffeine' in nombre:
                return 'xantina_alcaloide'
            elif any(x in nombre for x in ['acesulfame', 'cyclamate']):
                return 'edulcorante_artificial'
            elif any(x in nombre for x in ['candesartan', 'citalopram', 'diclofenac']):
                return 'farmaceutico'
            elif 'deet' in nombre:
                return 'repelente_organico'
            elif any(x in nombre for x in ['diuron', 'mecoprop']):
                return 'pesticida_herbicida'
            elif 'benzotriazole' in nombre:
                return 'inhibidor_corrosion'
            elif any(x in nombre for x in ['diphenyl', 'hmmm']):
                return 'aditivo_industrial'
            else:
                return 'organico_general'
        
        elif tipo_base == 'inorganico':
            if 'nh4' in nombre:
                return 'ion_amonio'
            elif 'po4' in nombre:
                return 'ion_fosfato'
            elif 'so4' in nombre:
                return 'ion_sulfato'
            else:
                return 'ion_general'
        
        elif tipo_base == 'fisicoquimico':
            if 'doc' in nombre:
                return 'carbono_organico_disuelto'
            elif 'turbidity' in nombre:
                return 'dispersion_particulas'
            else:
                return 'parametro_fisico'
        
        return 'no_clasificado'
    
    def _calcular_confianza_clasificacion(self, separabilidad, complejidad):
        """Calcula confianza en la clasificación"""
        if separabilidad > 15 and complejidad['nivel'] in ['baja', 'media']:
            return 'alta'
        elif separabilidad > 8:
            return 'media'
        else:
            return 'baja'
    
    def _recomendar_modelos(self, separabilidad, complejidad, ruido):
        """Recomienda modelos basado en características detectadas"""
        modelos = []
        
        # Logistic Regression para casos simples y bien separados
        if separabilidad > self.umbrales['separabilidad_alta'] and complejidad['nivel'] == 'baja':
            modelos.append('logistic_regression')
        
        # Random Forest para mayoría de casos (comprobado exitoso)
        if separabilidad > self.umbrales['separabilidad_media']:
            modelos.append('random_forest')
        
        # SVM RBF para casos complejos o mal separados
        if complejidad['nivel'] == 'alta' or separabilidad < self.umbrales['separabilidad_media']:
            modelos.append('svm_rbf')
        
        # Asegurar que siempre hay al menos un modelo
        if not modelos:
            modelos = ['random_forest']  # Fallback al exitoso comprobado
        
        return modelos
    
    def _recomendar_features(self, clasificacion, bandas):
        """Recomienda features específicos según tipo químico"""
        features = ['estadisticos_basicos', 'ratios_comparativos']  # Siempre incluidos
        
        tipo = clasificacion['tipo']
        subtipo = clasificacion['subtipo']
        
        if tipo == 'inorganico':
            features.extend(['uv_absorption', 'ionic_characteristics'])
        elif tipo == 'organico':
            features.extend(['chromophore_analysis', 'peak_analysis'])
            if 'farmaceutico' in subtipo:
                features.append('pharmaceutical_bands')
        elif tipo == 'fisicoquimico':
            if 'turbidity' in subtipo:
                features.extend(['scattering_analysis', 'broadband_response'])
            elif 'carbono' in subtipo:
                features.extend(['organic_carbon_bands', 'multiple_chromophores'])
        
        # Features específicos por banda dominante
        if bandas['dominante'] in ['uv_extremo', 'uv_cercano']:
            features.append('uv_specific_features')
        elif bandas['dominante'] in ['visible_azul', 'visible_verde', 'visible_rojo']:
            features.append('visible_specific_features')
        
        return list(set(features))  # Eliminar duplicados
    
    def _mostrar_analisis(self, caracteristicas):
        """Muestra resultados del análisis de forma clara"""
        print(f"   📊 Separabilidad: {caracteristicas['separabilidad_porcentaje']:.1f}%")
        print(f"   🔀 Complejidad: {caracteristicas['complejidad_nivel']} ({caracteristicas['complejidad_picos']} picos)")
        print(f"   📡 Ruido: {caracteristicas['ruido_nivel']} (ratio: {caracteristicas['ruido_ratio']:.3f})")
        print(f"   🌈 Banda dominante: {caracteristicas['banda_dominante']}")
        print(f"   🏷️ Clasificación: {caracteristicas['clasificacion_quimica']['tipo']} - {caracteristicas['clasificacion_quimica']['subtipo']}")
        print(f"   🔧 Modelos recomendados: {caracteristicas['modelos_recomendados']}")


class ModeloAdaptativoQuimico:
    """
    Sistema de modelado adaptativo que selecciona y optimiza modelos
    según las características químico-espectrales detectadas
    """
    
    def __init__(self):
        self.analizador = AnalizadorFirmasEspectrales()
        self.resultados = {}
        
    def extraer_features_adaptativos(self, datos, caracteristicas):
        """
        Extrae features adaptativamente según tipo químico y características
        """
        wavelengths = datos['wavelength'].values
        high_mean = datos['high_mean'].values
        low_mean = datos['low_mean'].values
        signature = datos['signature'].values
        high_std = datos['high_std'].values
        low_std = datos['low_std'].values
        
        features = {}
        
        # 1. Features estadísticos básicos (siempre incluidos)
        features.update(self._features_estadisticos_basicos(
            wavelengths, high_mean, low_mean, high_std, low_std
        ))
        
        # 2. Features ratios comparativos (siempre incluidos)
        features.update(self._features_ratios_comparativos(features))
        
        # 3. Features específicos por tipo químico
        tipo = caracteristicas['clasificacion_quimica']['tipo']
        if tipo == 'inorganico':
            features.update(self._features_inorganicos(wavelengths, high_mean, low_mean))
        elif tipo == 'organico':
            features.update(self._features_organicos(wavelengths, high_mean, low_mean, signature))
        elif tipo == 'fisicoquimico':
            features.update(self._features_fisicoquimicos(wavelengths, high_mean, low_mean, caracteristicas))
        
        # 4. Features específicos por banda dominante
        banda_dom = caracteristicas['banda_dominante']
        features.update(self._features_por_banda(wavelengths, high_mean, low_mean, banda_dom))
        
        # 5. Features de picos característicos
        if caracteristicas['complejidad_picos'] > 2:
            features.update(self._features_picos(wavelengths, signature))
        
        return features
    
    def _features_estadisticos_basicos(self, wavelengths, high_mean, low_mean, high_std, low_std):
        """Features estadísticos fundamentales"""
        features = {}
        
        for conc, values, stds in [('high', high_mean, high_std), ('low', low_mean, low_std)]:
            features[f'{conc}_mean'] = np.mean(values)
            features[f'{conc}_std'] = np.mean(stds)
            features[f'{conc}_max'] = np.max(values)
            features[f'{conc}_min'] = np.min(values)
            features[f'{conc}_auc'] = trapezoid(values, wavelengths)
            features[f'{conc}_range'] = np.max(values) - np.min(values)
            
            if len(wavelengths) > 1:
                slope, _ = np.polyfit(wavelengths, values, 1)
                features[f'{conc}_slope'] = slope
        
        return features
    
    def _features_ratios_comparativos(self, features):
        """Features de comparación entre concentraciones"""
        ratios = {}
        
        # Ratios principales (los más exitosos según diagnóstico previo)
        ratios['ratio_mean'] = features['high_mean'] / (features['low_mean'] + 1e-8)
        ratios['diff_mean'] = features['high_mean'] - features['low_mean']
        ratios['ratio_auc'] = features['high_auc'] / (features['low_auc'] + 1e-8)
        ratios['ratio_max'] = features['high_max'] / (features['low_max'] + 1e-8)
        ratios['ratio_std'] = features['high_std'] / (features['low_std'] + 1e-8)
        ratios['ratio_range'] = features['high_range'] / (features['low_range'] + 1e-8)
        
        return ratios
    
    def _features_inorganicos(self, wavelengths, high_mean, low_mean):
        """Features específicos para iones inorgánicos"""
        features = {}
        
        # Región UV donde absorben fuertemente los iones
        uv_mask = wavelengths <= 500
        if np.any(uv_mask):
            features['ion_uv_high'] = np.mean(high_mean[uv_mask])
            features['ion_uv_low'] = np.mean(low_mean[uv_mask])
            features['ion_uv_ratio'] = features['ion_uv_high'] / (features['ion_uv_low'] + 1e-8)
            features['ion_uv_auc'] = trapezoid(high_mean[uv_mask], wavelengths[uv_mask])
        
        return features
    
    def _features_organicos(self, wavelengths, high_mean, low_mean, signature):
        """Features específicos para compuestos orgánicos"""
        features = {}
        
        # Análisis de picos (cromóforos)
        peaks_high, _ = signal.find_peaks(high_mean, height=np.percentile(high_mean, 70))
        peaks_low, _ = signal.find_peaks(low_mean, height=np.percentile(low_mean, 70))
        
        features['organic_peaks_high'] = len(peaks_high)
        features['organic_peaks_low'] = len(peaks_low)
        features['organic_peak_enhancement'] = len(peaks_high) - len(peaks_low)
        
        # Cromóforos en UV-Visible
        chromophore_mask = (wavelengths >= 450) & (wavelengths <= 600)
        if np.any(chromophore_mask):
            features['chromophore_auc_high'] = trapezoid(high_mean[chromophore_mask], wavelengths[chromophore_mask])
            features['chromophore_auc_low'] = trapezoid(low_mean[chromophore_mask], wavelengths[chromophore_mask])
            features['chromophore_enhancement'] = features['chromophore_auc_high'] / (features['chromophore_auc_low'] + 1e-8)
        
        return features
    
    def _features_fisicoquimicos(self, wavelengths, high_mean, low_mean, caracteristicas):
        """Features específicos para parámetros físico-químicos"""
        features = {}
        
        subtipo = caracteristicas['clasificacion_quimica']['subtipo']
        
        if 'turbidity' in subtipo or 'dispersion' in subtipo:
            # Dispersión uniforme para turbidez
            visible_mask = (wavelengths >= 500) & (wavelengths <= 700)
            if np.any(visible_mask):
                features['scattering_uniformity_high'] = np.std(high_mean[visible_mask])
                features['scattering_uniformity_low'] = np.std(low_mean[visible_mask])
                features['scattering_ratio'] = features['scattering_uniformity_high'] / (features['scattering_uniformity_low'] + 1e-8)
        
        elif 'carbono' in subtipo:
            # Múltiples cromóforos para DOC
            for band_start, band_end, band_name in [(400, 500, 'uv'), (500, 600, 'vis'), (600, 700, 'red')]:
                band_mask = (wavelengths >= band_start) & (wavelengths <= band_end)
                if np.any(band_mask):
                    features[f'doc_{band_name}_high'] = trapezoid(high_mean[band_mask], wavelengths[band_mask])
                    features[f'doc_{band_name}_low'] = trapezoid(low_mean[band_mask], wavelengths[band_mask])
                    features[f'doc_{band_name}_ratio'] = features[f'doc_{band_name}_high'] / (features[f'doc_{band_name}_low'] + 1e-8)
        
        return features
    
    def _features_por_banda(self, wavelengths, high_mean, low_mean, banda_dominante):
        """Features específicos para la banda espectral dominante"""
        features = {}
        
        banda_ranges = {
            'uv_extremo': (400, 450),
            'uv_cercano': (450, 500),
            'visible_azul': (500, 550),
            'visible_verde': (550, 600),
            'visible_rojo': (600, 700),
            'nir_cercano': (700, 800)
        }
        
        if banda_dominante in banda_ranges:
            start, end = banda_ranges[banda_dominante]
            mask = (wavelengths >= start) & (wavelengths <= end)
            
            if np.any(mask):
                features[f'banda_dom_high'] = np.mean(high_mean[mask])
                features[f'banda_dom_low'] = np.mean(low_mean[mask])
                features[f'banda_dom_ratio'] = features['banda_dom_high'] / (features['banda_dom_low'] + 1e-8)
                features[f'banda_dom_auc'] = trapezoid(high_mean[mask], wavelengths[mask])
        
        return features
    
    def _features_picos(self, wavelengths, signature):
        """Features basados en picos característicos"""
        features = {}
        
        # Detectar picos principales
        threshold = np.std(signature) * 1.2
        peaks_pos, props_pos = signal.find_peaks(signature, height=threshold)
        peaks_neg, props_neg = signal.find_peaks(-signature, height=threshold)
        
        features['num_peaks_positive'] = len(peaks_pos)
        features['num_peaks_negative'] = len(peaks_neg)
        features['total_peaks'] = len(peaks_pos) + len(peaks_neg)
        
        # Intensidad promedio de picos
        if len(peaks_pos) > 0:
            features['avg_peak_intensity_pos'] = np.mean(signature[peaks_pos])
            features['max_peak_intensity_pos'] = np.max(signature[peaks_pos])
        else:
            features['avg_peak_intensity_pos'] = 0
            features['max_peak_intensity_pos'] = 0
        
        if len(peaks_neg) > 0:
            features['avg_peak_intensity_neg'] = np.mean(-signature[peaks_neg])
            features['max_peak_intensity_neg'] = np.max(-signature[peaks_neg])
        else:
            features['avg_peak_intensity_neg'] = 0
            features['max_peak_intensity_neg'] = 0
        
        return features
    
    def crear_dataset_adaptativo(self, features, caracteristicas):
        """
        Crea dataset con número de muestras adaptado a la separabilidad esperada
        """
        separabilidad = caracteristicas['separabilidad_porcentaje']
        
        # Número de muestras según separabilidad (más samples para casos difíciles)
        if separabilidad > 15:
            n_samples = 3  # Casos muy separables
        elif separabilidad > 10:
            n_samples = 4  # Casos moderadamente separables
        elif separabilidad > 5:
            n_samples = 6  # Casos difíciles
        else:
            n_samples = 8  # Casos muy difíciles
        
        # Crear muestras base
        muestra_alta = list(features.values())
        muestra_baja = self._generar_muestra_baja(features)
        
        samples = [muestra_alta, muestra_baja]
        labels = [1, 0]
        
        # Generar muestras adicionales con variabilidad controlada
        variabilidad = 0.08 if separabilidad > 10 else 0.12  # Menos variabilidad para casos bien separados
        
        for _ in range(n_samples):
            # Muestra alta con variabilidad
            noise_alta = [val * np.random.normal(1.0, variabilidad) for val in muestra_alta]
            samples.append(noise_alta)
            labels.append(1)
            
            # Muestra baja con variabilidad
            noise_baja = [val * np.random.normal(1.0, variabilidad) for val in muestra_baja]
            samples.append(noise_baja)
            labels.append(0)
        
        # Crear DataFrame
        feature_names = list(features.keys())
        df = pd.DataFrame(samples, columns=feature_names)
        df['label'] = labels
        
        print(f"   📊 Dataset creado: {len(samples)} muestras ({n_samples+1} por clase)")
        
        return df, feature_names
    
    def _generar_muestra_baja(self, features):
        """Genera muestra de baja concentración basada en la de alta"""
        muestra_baja = []
        
        for nombre, valor in features.items():
            if nombre.startswith('high_'):
                # Buscar correspondiente 'low_'
                low_nombre = nombre.replace('high_', 'low_')
                if low_nombre in features:
                    muestra_baja.append(features[low_nombre])
                else:
                    muestra_baja.append(valor * 0.6)  # Reducir si no hay correspondiente
            elif nombre.startswith('low_'):
                # Buscar correspondiente 'high_'
                high_nombre = nombre.replace('low_', 'high_')
                if high_nombre in features:
                    muestra_baja.append(features[high_nombre])
                else:
                    muestra_baja.append(valor * 1.4)  # Aumentar si no hay correspondiente
            elif 'ratio' in nombre:
                muestra_baja.append(1 / (valor + 1e-8))  # Invertir ratio
            elif 'diff' in nombre:
                muestra_baja.append(-valor)  # Invertir diferencia
            elif 'enhancement' in nombre:
                muestra_baja.append(-valor)  # Invertir enhancement
            else:
                muestra_baja.append(valor)  # Mantener igual para features neutrales
        
        return muestra_baja
    
    def entrenar_modelos_adaptativos(self, datos, nombre_contaminante):
        """
        Entrena los 3 modelos y selecciona el mejor según características detectadas
        """
        print(f"\n{'='*70}")
        print(f"🧬 SISTEMA ADAPTATIVO: {nombre_contaminante}")
        print(f"{'='*70}")
        
        # 1. Analizar características químico-espectrales
        caracteristicas = self.analizador.analizar_firma_espectral(datos, nombre_contaminante)
        
        # 2. Extraer features adaptativos
        print(f"\n🔧 Extrayendo features adaptativos...")
        features = self.extraer_features_adaptativos(datos, caracteristicas)
        print(f"   ✅ {len(features)} features extraídos")
        
        # 3. Crear dataset adaptativo
        dataset, feature_names = self.crear_dataset_adaptativo(features, caracteristicas)
        
        # 4. Preparar datos para ML
        X = dataset[feature_names].values
        y = dataset['label'].values
        
        # 5. Probar los modelos recomendados
        modelos_recomendados = caracteristicas['modelos_recomendados']
        print(f"\n🔧 Probando modelos recomendados: {modelos_recomendados}")
        
        resultados_modelos = {}
        
        # Siempre probar los 3 modelos para comparación completa
        modelos_config = {
            'logistic_regression': self._crear_logistic_regression(caracteristicas),
            'random_forest': self._crear_random_forest(caracteristicas),
            'svm_rbf': self._crear_svm_rbf(caracteristicas)
        }
        
        for nombre_modelo, modelo in modelos_config.items():
            print(f"\n   🔧 Entrenando: {nombre_modelo}")
            resultado = self._entrenar_modelo_individual(X, y, modelo, nombre_modelo, caracteristicas)
            resultados_modelos[nombre_modelo] = resultado
            
            # Marcar si es recomendado
            if nombre_modelo in modelos_recomendados:
                resultado['recomendado'] = True
                print(f"      ⭐ RECOMENDADO por análisis espectral")
            else:
                resultado['recomendado'] = False
        
        # 6. Seleccionar mejor modelo
        mejor_modelo = self._seleccionar_mejor_modelo(resultados_modelos, caracteristicas)
        
        # 7. Compilar resultados finales
        resultado_final = {
            'contaminante': nombre_contaminante,
            'caracteristicas_espectrales': caracteristicas,
            'features_extraidos': len(features),
            'dataset_size': len(dataset),
            'modelos_probados': resultados_modelos,
            'mejor_modelo': mejor_modelo,
            'timestamp': datetime.now().isoformat()
        }
        
        self._mostrar_resultados_finales(resultado_final)
        self.resultados[nombre_contaminante] = resultado_final
        
        return resultado_final
    
    def _crear_logistic_regression(self, caracteristicas):
        """Crea Logistic Regression adaptado"""
        # Regularización según separabilidad
        separabilidad = caracteristicas['separabilidad_porcentaje']
        C = 1.0 if separabilidad > 10 else 0.5
        
        return LogisticRegression(random_state=42, max_iter=1000, C=C)
    
    def _crear_random_forest(self, caracteristicas):
        """Crea Random Forest adaptado"""
        complejidad = caracteristicas['complejidad_nivel']
        
        if complejidad == 'alta':
            n_estimators, max_depth = 20, 5
        elif complejidad == 'media':
            n_estimators, max_depth = 15, 4
        else:
            n_estimators, max_depth = 10, 3
        
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1
        )
    
    def _crear_svm_rbf(self, caracteristicas):
        """Crea SVM RBF adaptado"""
        separabilidad = caracteristicas['separabilidad_porcentaje']
        ruido = caracteristicas['ruido_nivel']
        
        # C según separabilidad, gamma según ruido
        C = 1.0 if separabilidad > 8 else 0.5
        gamma = 'scale' if ruido == 'bajo' else 'auto'
        
        return SVC(kernel='rbf', random_state=42, C=C, gamma=gamma, probability=True)
    
    def _entrenar_modelo_individual(self, X, y, modelo, nombre_modelo, caracteristicas):
        """Entrena un modelo individual con validación adaptativa"""
        
        # Escalado adaptativo
        if caracteristicas['ruido_nivel'] == 'alto':
            scaler = RobustScaler()  # Más robusto al ruido
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        # Validación adaptativa según tamaño del dataset
        if len(X) <= 6:
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
        else:
            cv = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
            cv_name = f"{min(5, len(X)//2)}-Fold"
        
        # Validación cruzada
        try:
            cv_scores = cross_val_score(modelo, X_scaled, y, cv=cv, scoring='f1', n_jobs=1)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
        except Exception as e:
            print(f"      ⚠️ Error en CV: {e}")
            cv_mean, cv_std = 0.0, 0.0
        
        # Entrenamiento completo
        try:
            modelo.fit(X_scaled, y)
            y_pred = modelo.predict(X_scaled)
            
            train_accuracy = accuracy_score(y, y_pred)
            train_f1 = f1_score(y, y_pred, zero_division=0)
            
            # AUC si es posible
            try:
                if hasattr(modelo, 'predict_proba') and len(set(y)) > 1:
                    y_proba = modelo.predict_proba(X_scaled)[:, 1]
                    auc = roc_auc_score(y, y_proba)
                else:
                    auc = 0.5
            except:
                auc = 0.5
            
        except Exception as e:
            print(f"      ❌ Error en entrenamiento: {e}")
            train_accuracy = train_f1 = auc = 0.0
        
        resultado = {
            'modelo': nombre_modelo,
            'train_accuracy': float(train_accuracy),
            'train_f1': float(train_f1),
            'cv_f1_mean': float(cv_mean),
            'cv_f1_std': float(cv_std),
            'auc': float(auc),
            'cv_method': cv_name,
            'gap_f1': float(train_f1 - cv_mean),
            'exito': train_f1 > 0.6  # Umbral de éxito
        }
        
        print(f"      📊 F1: {train_f1:.3f} | CV: {cv_mean:.3f}±{cv_std:.3f} | AUC: {auc:.3f}")
        
        return resultado
    
    def _seleccionar_mejor_modelo(self, resultados_modelos, caracteristicas):
        """Selecciona el mejor modelo considerando recomendaciones y rendimiento"""
        
        # Filtrar modelos exitosos
        exitosos = {k: v for k, v in resultados_modelos.items() if v['exito']}
        
        if not exitosos:
            # Si ninguno es exitoso, elegir el menos malo
            mejor = max(resultados_modelos.items(), key=lambda x: x[1]['train_f1'])
            return mejor[0]
        
        # Priorizar modelos recomendados entre los exitosos
        recomendados_exitosos = {k: v for k, v in exitosos.items() if v['recomendado']}
        
        if recomendados_exitosos:
            # Elegir el mejor entre los recomendados exitosos
            mejor = max(recomendados_exitosos.items(), key=lambda x: x[1]['cv_f1_mean'])
            return mejor[0]
        else:
            # Elegir el mejor entre todos los exitosos
            mejor = max(exitosos.items(), key=lambda x: x[1]['cv_f1_mean'])
            return mejor[0]
    
    def _mostrar_resultados_finales(self, resultado):
        """Muestra resultados finales del análisis"""
        print(f"\n{'='*70}")
        print(f"📊 RESULTADOS FINALES: {resultado['contaminante']}")
        print(f"{'='*70}")
        
        # Características detectadas
        carac = resultado['caracteristicas_espectrales']
        print(f"🔬 ANÁLISIS ESPECTRAL:")
        print(f"   📊 Separabilidad: {carac['separabilidad_porcentaje']:.1f}%")
        print(f"   🔀 Complejidad: {carac['complejidad_nivel']} ({carac['complejidad_picos']} picos)")
        print(f"   🏷️ Tipo químico: {carac['clasificacion_quimica']['tipo']} - {carac['clasificacion_quimica']['subtipo']}")
        print(f"   🌈 Banda dominante: {carac['banda_dominante']}")
        
        # Comparación de modelos
        print(f"\n🔧 COMPARACIÓN DE MODELOS:")
        print(f"{'Modelo':<20} | {'F1 Train':<8} | {'F1 CV':<12} | {'AUC':<6} | {'Recom':<5} | {'Estado'}")
        print("-" * 70)
        
        mejor = resultado['mejor_modelo']
        for nombre, res in resultado['modelos_probados'].items():
            estado = "✅ MEJOR" if nombre == mejor else ("✅" if res['exito'] else "❌")
            recom = "⭐" if res['recomendado'] else " "
            
            print(f"{nombre:<20} | {res['train_f1']:<8.3f} | {res['cv_f1_mean']:<6.3f}±{res['cv_f1_std']:<4.3f} | {res['auc']:<6.3f} | {recom:<5} | {estado}")
        
        # Evaluación final
        mejor_resultado = resultado['modelos_probados'][mejor]
        f1_final = mejor_resultado['cv_f1_mean']
        
        if f1_final >= 0.9:
            evaluacion = "🟢 EXCELENTE"
        elif f1_final >= 0.7:
            evaluacion = "🟡 BUENO"
        elif f1_final >= 0.5:
            evaluacion = "🟠 ACEPTABLE"
        else:
            evaluacion = "🔴 PROBLEMÁTICO"
        
        print(f"\n🏆 MODELO SELECCIONADO: {mejor}")
        print(f"📈 RENDIMIENTO FINAL: {evaluacion} (F1 CV: {f1_final:.3f})")
        
        # Explicación de la selección
        if mejor_resultado['recomendado']:
            print(f"💡 Seleccionado por: Recomendación espectral + Mejor rendimiento")
        else:
            print(f"💡 Seleccionado por: Mejor rendimiento (no recomendado espectralmente)")


class GestorContaminantesCompleto:
    """
    Gestor que procesa todos los contaminantes disponibles automáticamente
    Adaptado para estructura de carpetas organizadas
    """
    
    def __init__(self, directorio_base="firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.modelo_adaptativo = ModeloAdaptativoQuimico()
        self.resultados_completos = {}
        
        # Mapeo de nombres de carpetas a nombres técnicos
        self.mapeo_contaminantes = {
            'Acesulfame': 'Acesulfame_Ng_L',
            'Benzotriazole': 'Benzotriazole_Ng_L', 
            'Caffeine': 'Caffeine_Ng_L',
            'Candesartan': 'Candesartan_Ng_L',
            'Citalopram': 'Citalopram_Ng_L',
            'Cyclamate': 'Cyclamate_Ng_L',
            'Deet': 'Deet_Ng_L',
            'Diclofenac': 'Diclofenac_Ng_L',
            'Diphenylguanidine': '13Diphenylguanidine_Ng_L',
            'Diuron': 'Diuron_Ng_L',
            'Doc': 'Doc_Mg_L',
            'Hmmm': 'Hmmm_Ng_L',
            'Hydrochlorthiazide': 'Hydrochlorthiazide_Ng_L',
            'Mecoprop': 'Mecoprop_Ng_L',
            'Methylbenzotriazole': '45Methylbenzotriazole_Ng_L',
            'Nh4': 'Nh4_Mg_L',
            'Nsol': 'Nsol_Mg_L',
            'Oit': 'Oit_Ng_L',
            'Po4': 'Po4_Mg_L',
            'Quinone': '6PpdQuinone_Ng_L',
            'So4': 'So4_Mg_L',
            'Turbidity': 'Turbidity_Ntu'
        }
    
    def detectar_contaminantes_disponibles(self):
        """Detecta contaminantes en estructura de carpetas organizadas"""
        contaminantes_encontrados = []
        
        if not os.path.exists(self.directorio_base):
            print(f"❌ No se encontró directorio: {self.directorio_base}")
            return []
        
        # Buscar en cada subcarpeta
        for carpeta in os.listdir(self.directorio_base):
            ruta_carpeta = os.path.join(self.directorio_base, carpeta)
            
            if os.path.isdir(ruta_carpeta):
                # Buscar archivo CSV en la carpeta
                archivos_csv = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
                
                if archivos_csv:
                    # Determinar nombre técnico
                    if carpeta in self.mapeo_contaminantes:
                        nombre_tecnico = self.mapeo_contaminantes[carpeta]
                    else:
                        # Inferir del nombre del archivo
                        nombre_tecnico = archivos_csv[0].replace('_datos_espectrales.csv', '')
                    
                    contaminantes_encontrados.append({
                        'carpeta': carpeta,
                        'nombre_tecnico': nombre_tecnico,
                        'archivo_csv': archivos_csv[0],
                        'ruta_completa': os.path.join(ruta_carpeta, archivos_csv[0])
                    })
        
        return sorted(contaminantes_encontrados, key=lambda x: x['nombre_tecnico'])
    
    def procesar_todos_contaminantes(self, max_contaminantes=None):
        """
        Procesa todos los contaminantes disponibles con el sistema adaptativo
        Adaptado para estructura de carpetas organizadas
        """
        contaminantes_info = self.detectar_contaminantes_disponibles()
        
        if not contaminantes_info:
            print(f"❌ No se encontraron contaminantes en {self.directorio_base}")
            print(f"💡 Verifica que la estructura sea: {self.directorio_base}/[Contaminante]/[Contaminante]_datos_espectrales.csv")
            return {}
        
        if max_contaminantes:
            contaminantes_info = contaminantes_info[:max_contaminantes]
        
        print(f"🧬 SISTEMA ADAPTATIVO COMPLETO")
        print(f"="*60)
        print(f"📁 Directorio base: {self.directorio_base}")
        print(f"📋 Contaminantes detectados: {len(contaminantes_info)}")
        print(f"🔧 Modelos a probar: Logistic Regression, Random Forest, SVM RBF")
        print(f"🎯 Selección automática según características químico-espectrales")
        
        # Mostrar lista
        print(f"\n📊 CONTAMINANTES ENCONTRADOS:")
        for i, info in enumerate(contaminantes_info, 1):
            print(f"   {i:2d}. {info['carpeta']:<15} → {info['nombre_tecnico']}")
        
        inicio_total = datetime.now()
        
        # Procesar cada contaminante
        for i, info in enumerate(contaminantes_info, 1):
            nombre_carpeta = info['carpeta']
            nombre_tecnico = info['nombre_tecnico']
            ruta_csv = info['ruta_completa']
            
            print(f"\n[{i}/{len(contaminantes_info)}] 🔄 PROCESANDO: {nombre_carpeta}")
            print(f"   📁 Archivo: {info['archivo_csv']}")
            
            try:
                # Cargar datos desde ruta específica
                datos = pd.read_csv(ruta_csv)
                
                # Verificar estructura de datos
                columnas_requeridas = ['wavelength', 'high_mean', 'low_mean', 'high_std', 'low_std', 'signature']
                columnas_faltantes = [col for col in columnas_requeridas if col not in datos.columns]
                
                if columnas_faltantes:
                    print(f"   ⚠️ Columnas faltantes: {columnas_faltantes}")
                    print(f"   📋 Columnas disponibles: {list(datos.columns)}")
                    continue
                
                print(f"   ✅ Datos cargados: {datos.shape}")
                
                # Procesar con sistema adaptativo usando nombre técnico
                resultado = self.modelo_adaptativo.entrenar_modelos_adaptativos(datos, nombre_tecnico)
                
                if resultado:
                    # Agregar información de carpeta
                    resultado['carpeta_origen'] = nombre_carpeta
                    resultado['archivo_origen'] = info['archivo_csv']
                    self.resultados_completos[nombre_tecnico] = resultado
                    print(f"   ✅ COMPLETADO")
                else:
                    print(f"   ❌ FALLÓ EL PROCESAMIENTO")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                import traceback
                print(f"   📝 Detalle: {traceback.format_exc()}")
                self.resultados_completos[nombre_tecnico] = None
        
        # Generar reporte final
        fin_total = datetime.now()
        tiempo_total = (fin_total - inicio_total).total_seconds()
        
        self.generar_reporte_final_completo(tiempo_total)
        
        return self.resultados_completos
    
    def generar_reporte_final_completo(self, tiempo_total):
        """Genera reporte final con análisis completo del sistema"""
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE FINAL SISTEMA ADAPTATIVO COMPLETO")
        print(f"{'='*80}")
        
        # Estadísticas generales
        exitosos = {k: v for k, v in self.resultados_completos.items() if v is not None}
        fallidos = [k for k, v in self.resultados_completos.items() if v is None]
        
        print(f"✅ Contaminantes procesados exitosamente: {len(exitosos)}")
        print(f"❌ Contaminantes fallidos: {len(fallidos)}")
        print(f"⏱️ Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        
        if not exitosos:
            print("\n❌ No hay resultados para analizar")
            return
        
        # Análisis por tipo químico
        tipos_quimicos = {}
        modelos_seleccionados = {}
        rendimientos = []
        
        for nombre, resultado in exitosos.items():
            tipo = resultado['caracteristicas_espectrales']['clasificacion_quimica']['tipo']
            mejor_modelo = resultado['mejor_modelo']
            f1_cv = resultado['modelos_probados'][mejor_modelo]['cv_f1_mean']
            
            if tipo not in tipos_quimicos:
                tipos_quimicos[tipo] = []
            tipos_quimicos[tipo].append((nombre, f1_cv))
            
            if mejor_modelo not in modelos_seleccionados:
                modelos_seleccionados[mejor_modelo] = 0
            modelos_seleccionados[mejor_modelo] += 1
            
            rendimientos.append(f1_cv)
        
        # Estadísticas de rendimiento
        f1_promedio = np.mean(rendimientos)
        f1_std = np.std(rendimientos)
        
        print(f"\n📈 ESTADÍSTICAS DE RENDIMIENTO:")
        print(f"   🎯 F1-Score promedio: {f1_promedio:.3f} ± {f1_std:.3f}")
        print(f"   🏆 Mejor rendimiento: {max(rendimientos):.3f}")
        print(f"   📉 Peor rendimiento: {min(rendimientos):.3f}")
        
        # Análisis por tipo químico
        print(f"\n🧪 ANÁLISIS POR TIPO QUÍMICO:")
        for tipo, contaminantes in tipos_quimicos.items():
            f1_scores = [f1 for _, f1 in contaminantes]
            print(f"   {tipo.upper()}: {len(contaminantes)} contaminantes")
            print(f"      F1 promedio: {np.mean(f1_scores):.3f}")
            print(f"      Mejor: {max(contaminantes, key=lambda x: x[1])[0]} ({max(f1_scores):.3f})")
        
        # Efectividad de la selección de modelos
        print(f"\n🔧 EFECTIVIDAD DE SELECCIÓN DE MODELOS:")
        for modelo, count in sorted(modelos_seleccionados.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (count / len(exitosos)) * 100
            print(f"   {modelo}: {count} selecciones ({porcentaje:.1f}%)")
        
        # Top performers
        print(f"\n🏆 TOP 10 CONTAMINANTES:")
        ranking = sorted(exitosos.items(), key=lambda x: x[1]['modelos_probados'][x[1]['mejor_modelo']]['cv_f1_mean'], reverse=True)
        
        for i, (nombre, resultado) in enumerate(ranking[:10], 1):
            mejor_modelo = resultado['mejor_modelo']
            f1_cv = resultado['modelos_probados'][mejor_modelo]['cv_f1_mean']
            tipo = resultado['caracteristicas_espectrales']['clasificacion_quimica']['tipo']
            separabilidad = resultado['caracteristicas_espectrales']['separabilidad_porcentaje']
            
            print(f"   {i:2d}. {nombre:<25} | F1: {f1_cv:.3f} | {mejor_modelo:<15} | {tipo} ({separabilidad:.1f}%)")
        
        # Casos problemáticos
        problematicos = [(n, r) for n, r in exitosos.items() 
                        if r['modelos_probados'][r['mejor_modelo']]['cv_f1_mean'] < 0.6]
        
        if problematicos:
            print(f"\n⚠️ CONTAMINANTES PROBLEMÁTICOS (F1 < 0.6):")
            for nombre, resultado in problematicos:
                mejor_modelo = resultado['mejor_modelo']
                f1_cv = resultado['modelos_probados'][mejor_modelo]['cv_f1_mean']
                separabilidad = resultado['caracteristicas_espectrales']['separabilidad_porcentaje']
                print(f"   • {nombre}: F1={f1_cv:.3f}, Separabilidad={separabilidad:.1f}%")
        
        # Guardar resultados
        self.guardar_resultados_json()
        
        # Evaluación final del sistema
        excelentes = len([r for r in rendimientos if r >= 0.9])
        buenos = len([r for r in rendimientos if 0.7 <= r < 0.9])
        aceptables = len([r for r in rendimientos if 0.5 <= r < 0.7])
        
        print(f"\n🎯 EVALUACIÓN FINAL DEL SISTEMA:")
        print(f"   🟢 Excelentes (F1≥0.9): {excelentes}/{len(exitosos)} ({excelentes/len(exitosos)*100:.1f}%)")
        print(f"   🟡 Buenos (F1≥0.7): {buenos}/{len(exitosos)} ({buenos/len(exitosos)*100:.1f}%)")
        print(f"   🟠 Aceptables (F1≥0.5): {aceptables}/{len(exitosos)} ({aceptables/len(exitosos)*100:.1f}%)")
        
        if f1_promedio >= 0.8:
            evaluacion_global = "🟢 SISTEMA EXCELENTE"
        elif f1_promedio >= 0.6:
            evaluacion_global = "🟡 SISTEMA BUENO"
        elif f1_promedio >= 0.4:
            evaluacion_global = "🟠 SISTEMA ACEPTABLE"
        else:
            evaluacion_global = "🔴 SISTEMA REQUIERE MEJORAS"
        
        print(f"\n🏆 EVALUACIÓN GLOBAL: {evaluacion_global}")
        print(f"💡 El sistema adaptativo logró F1 promedio de {f1_promedio:.3f}")
        print(f"🎯 Selección automática de modelos funcionó en {len(exitosos)} casos")
    
    def guardar_resultados_json(self):
        """Guarda resultados completos en JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_json = f"resultados_sistema_adaptativo_{timestamp}.json"
        
        # Preparar datos para JSON (convertir numpy types)
        datos_json = {}
        for nombre, resultado in self.resultados_completos.items():
            if resultado is not None:
                datos_json[nombre] = self._convertir_para_json(resultado)
        
        try:
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(datos_json, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados guardados: {archivo_json}")
        except Exception as e:
            print(f"\n⚠️ Error guardando JSON: {e}")
    
    def _convertir_para_json(self, obj):
        """Convierte tipos numpy/pandas a tipos nativos de Python"""
        if isinstance(obj, dict):
            return {k: self._convertir_para_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convertir_para_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj


# FUNCIÓN PRINCIPAL PARA EJECUTAR EL SISTEMA COMPLETO
def ejecutar_sistema_adaptativo_completo(directorio_base="todo/firmas_espectrales_csv", max_contaminantes=None):
    """
    Función principal que ejecuta el sistema adaptativo completo
    Adaptada para estructura de carpetas organizadas
    
    Args:
        directorio_base: Directorio base con subcarpetas de contaminantes
        max_contaminantes: Límite de contaminantes a procesar (None = todos)
    
    Returns:
        dict: Resultados completos de todos los contaminantes
    """
    
    print("🧬 SISTEMA ADAPTATIVO QUÍMICO-ESPECTRAL")
    print("="*60)
    print("🎯 Selección automática de modelos basada en características químicas")
    print("🔧 Modelos: Logistic Regression, Random Forest, SVM RBF")
    print("📊 Análisis espectral automático por cada contaminante")
    print(f"📁 Estructura esperada: {directorio_base}/[Contaminante]/[Contaminante]_datos_espectrales.csv")
    print()
    
    gestor = GestorContaminantesCompleto(directorio_base)
    resultados = gestor.procesar_todos_contaminantes(max_contaminantes)
    
    return resultados


if __name__ == "__main__":
    # CONFIGURACIÓN PARA TU ESTRUCTURA DE CARPETAS
    print("🎯 SISTEMA ADAPTATIVO - CONFIGURACIÓN AUTOMÁTICA")
    print("="*60)
    print()
    
    # Detectar contaminantes disponibles primero
    gestor_temp = GestorContaminantesCompleto("todo/firmas_espectrales_csv")
    contaminantes_disponibles = gestor_temp.detectar_contaminantes_disponibles()
    
    if not contaminantes_disponibles:
        print("❌ NO SE ENCONTRARON CONTAMINANTES")
        print("💡 Verifica que:")
        print("   1. La carpeta 'firmas_espectrales_csv' existe")
        print("   2. Cada contaminante tiene su propia subcarpeta")
        print("   3. Cada subcarpeta contiene un archivo *_datos_espectrales.csv")
        exit()
    
    print(f"✅ CONTAMINANTES DETECTADOS: {len(contaminantes_disponibles)}")
    for i, info in enumerate(contaminantes_disponibles[:10], 1):  # Mostrar primeros 10
        print(f"   {i:2d}. {info['carpeta']:<15} → {info['nombre_tecnico']}")
    
    if len(contaminantes_disponibles) > 10:
        print(f"   ... y {len(contaminantes_disponibles) - 10} más")
    
    print(f"\n🚀 OPCIONES DE EJECUCIÓN:")
    print(f"1. 🧪 Prueba rápida (3 contaminantes representativos)")
    print(f"2. 🔬 Sistema completo ({len(contaminantes_disponibles)} contaminantes)")
    print(f"3. 📊 Cantidad personalizada")
    
    # EJECUCIÓN AUTOMÁTICA DE PRUEBA
    print(f"\n⚡ EJECUTANDO PRUEBA AUTOMÁTICA (3 contaminantes)...")
    
    try:
        # Seleccionar 3 representativos: orgánico, inorgánico, físico-químico
        contaminantes_prueba = []
        tipos_deseados = ['Caffeine', 'Nh4', 'Turbidity']  # Representativos
        
        for tipo in tipos_deseados:
            encontrado = next((c for c in contaminantes_disponibles if tipo in c['carpeta']), None)
            if encontrado:
                contaminantes_prueba.append(encontrado)
        
        # Si no encontramos los específicos, usar los primeros 3
        if len(contaminantes_prueba) < 3:
            contaminantes_prueba = contaminantes_disponibles[:3]
        
        print(f"📋 Contaminantes de prueba seleccionados:")
        for i, info in enumerate(contaminantes_prueba, 1):
            print(f"   {i}. {info['carpeta']} → {info['nombre_tecnico']}")
        
        # Ejecutar sistema con contaminantes seleccionados
        resultados = ejecutar_sistema_adaptativo_completo(
            directorio_base="todo/firmas_espectrales_csv",
            max_contaminantes=22  # Limitar a 10 para prueba rápida
        )
        
        if resultados:
            print(f"\n🎉 ¡PRUEBA EXITOSA!")
            print(f"✅ Sistema funcionando correctamente")
            print(f"📊 Resultados generados para {len([r for r in resultados.values() if r is not None])} contaminantes")
            print()
            print(f"🔄 PARA EJECUTAR SISTEMA COMPLETO:")
            print(f"   Cambiar: max_contaminantes=None")
            print(f"   Esto procesará automáticamente todos los {len(contaminantes_disponibles)} contaminantes")
            print()
            print(f"💾 Archivos generados:")
            print(f"   - resultados_sistema_adaptativo_[timestamp].json")
            print(f"   - Reportes detallados en consola")
        else:
            print(f"\n❌ No se generaron resultados")
            
    except Exception as e:
        print(f"\n❌ Error en ejecución: {e}")
        print(f"💡 Verifica la estructura de archivos y vuelve a intentar")
        import traceback
        print(f"📝 Detalle del error:")
        traceback.print_exc()