"""
Spectral Feature Engineering para Calidad de Agua
=================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

class SpectralFeatureEngineer:
    """
    Clase principal para extracción de features espectrales
    """
    
    def __init__(self, wavelengths: Union[List, np.ndarray]):
        """
        Inicializar el extractor de features espectrales
        
        Args:
            wavelengths: Lista o array de wavelengths en nanómetros
        """
        self.wavelengths = np.array(wavelengths)
        self.n_wavelengths = len(self.wavelengths)
        
        # Validar wavelengths
        if self.n_wavelengths < 3:
            raise ValueError("Se necesitan al menos 3 wavelengths para extracción de features")
        
        # Configurar índices espectrales conocidos para calidad de agua
        self.water_quality_indices = {
            'NDVI': [(850, 660), 'normalized_difference'],
            'NDWI': [(857, 1241), 'normalized_difference'],
            'MNDWI': [(560, 1640), 'normalized_difference'],
            'Turbidity_Index': [(660, 560), 'ratio'],
            'Chlorophyll_Index': [(670, 700), 'ratio'],
            'CDOM_Index': [(412, 440), 'ratio'],
            'Suspended_Solids': [(665, 709), 'ratio'],
            'Blue_Green_Ratio': [(490, 560), 'ratio'],
            'Red_NIR_Ratio': [(670, 800), 'ratio']
        }
        
        # Rangos espectrales estándar
        self.spectral_ranges = {
            'UV_C': (200, 280),
            'UV_B': (280, 315),
            'UV_A': (315, 400),
            'Violet': (380, 450),
            'Blue': (450, 495),
            'Green': (495, 570),
            'Yellow': (570, 590),
            'Orange': (590, 620),
            'Red': (620, 750),
            'NIR': (750, 1000),
            'SWIR1': (1000, 1800),
            'SWIR2': (1800, 2500)
        }
        
        print(f" SpectralFeatureEngineer inicializado:")
        print(f"   Wavelengths: {self.n_wavelengths} ({self.wavelengths.min():.0f}-{self.wavelengths.max():.0f} nm)")
        print(f"   Resolución promedio: {np.median(np.diff(self.wavelengths)):.1f} nm")
    
    def extract_all_features(self, spectra: np.ndarray) -> pd.DataFrame:
        """
        Extraer todas las features espectrales de un conjunto de espectros
        
        Args:
            spectra: Array de espectros (n_samples, n_wavelengths)
            
        Returns:
            DataFrame con todas las features espectrales extraídas
        """
        
        if len(spectra.shape) != 2:
            raise ValueError(f"Spectra debe ser 2D (n_samples, n_wavelengths), recibido: {spectra.shape}")
        
        if spectra.shape[1] != self.n_wavelengths:
            raise ValueError(f"Número de wavelengths no coincide: esperado {self.n_wavelengths}, recibido {spectra.shape[1]}")
        
        # Lista para almacenar todas las features
        all_features = []
        all_feature_names = []
        
        print(f" Extrayendo features espectrales de {spectra.shape[0]} espectros...")
        
        # 1. Features estadísticas básicas
        try:
            stat_features, stat_names = self._extract_statistical_features(spectra)
            all_features.append(stat_features)
            all_feature_names.extend(stat_names)
            print(f"    Features estadísticas: {len(stat_names)}")
        except Exception as e:
            print(f"    Error en features estadísticas: {e}")
        
        # 2. Índices espectrales
        try:
            index_features, index_names = self._extract_spectral_indices(spectra)
            all_features.append(index_features)
            all_feature_names.extend(index_names)
            print(f"    Índices espectrales: {len(index_names)}")
        except Exception as e:
            print(f"    Error en índices espectrales: {e}")
        
        # 3. Features de forma espectral
        try:
            shape_features, shape_names = self._extract_shape_features(spectra)
            all_features.append(shape_features)
            all_feature_names.extend(shape_names)
            print(f"    Features de forma: {len(shape_names)}")
        except Exception as e:
            print(f"    Error en features de forma: {e}")
        
        # 4. Features por rangos espectrales
        try:
            range_features, range_names = self._extract_range_features(spectra)
            all_features.append(range_features)
            all_feature_names.extend(range_names)
            print(f"    Features por rangos: {len(range_names)}")
        except Exception as e:
            print(f"    Error en features por rangos: {e}")
        
        # 5. Features de absorción y picos
        try:
            peak_features, peak_names = self._extract_peak_features(spectra)
            all_features.append(peak_features)
            all_feature_names.extend(peak_names)
            print(f"    Features de picos: {len(peak_names)}")
        except Exception as e:
            print(f"    Error en features de picos: {e}")
        
        # 6. Features derivadas
        try:
            deriv_features, deriv_names = self._extract_derivative_features(spectra)
            all_features.append(deriv_features)
            all_feature_names.extend(deriv_names)
            print(f"    Features derivadas: {len(deriv_names)}")
        except Exception as e:
            print(f"    Error en features derivadas: {e}")
        
        # Combinar todas las features
        if all_features:
            combined_features = np.column_stack(all_features)
            feature_df = pd.DataFrame(combined_features, columns=all_feature_names)
            
            print(f"    Total features extraídas: {len(all_feature_names)}")
            
            # Verificar que no hay NaN o inf
            feature_df = self._clean_features(feature_df)
            
            return feature_df
        else:
            print(f"    No se pudieron extraer features")
            # Retornar features básicas como fallback
            return self._fallback_features(spectra)
    
    def _extract_statistical_features(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer features estadísticas básicas"""
        
        features = []
        names = []
        
        # Estadísticas básicas
        features.append(np.mean(spectra, axis=1))
        names.append('mean_reflectance')
        
        features.append(np.std(spectra, axis=1))
        names.append('std_reflectance')
        
        features.append(np.median(spectra, axis=1))
        names.append('median_reflectance')
        
        features.append(np.min(spectra, axis=1))
        names.append('min_reflectance')
        
        features.append(np.max(spectra, axis=1))
        names.append('max_reflectance')
        
        # Rango y percentiles
        features.append(np.max(spectra, axis=1) - np.min(spectra, axis=1))
        names.append('range_reflectance')
        
        features.append(np.percentile(spectra, 25, axis=1))
        names.append('q25_reflectance')
        
        features.append(np.percentile(spectra, 75, axis=1))
        names.append('q75_reflectance')
        
        # Momentos estadísticos
        try:
            features.append(skew(spectra, axis=1))
            names.append('skewness_reflectance')
        except:
            features.append(np.zeros(len(spectra)))
            names.append('skewness_reflectance')
        
        try:
            features.append(kurtosis(spectra, axis=1))
            names.append('kurtosis_reflectance')
        except:
            features.append(np.zeros(len(spectra)))
            names.append('kurtosis_reflectance')
        
        # Área bajo la curva
        features.append(np.trapz(spectra, self.wavelengths, axis=1))
        names.append('area_under_curve')
        
        # Coeficiente de variación
        mean_vals = np.mean(spectra, axis=1)
        std_vals = np.std(spectra, axis=1)
        cv = np.divide(std_vals, mean_vals + 1e-8)
        features.append(cv)
        names.append('coefficient_variation')
        
        return np.column_stack(features), names
    
    def _extract_spectral_indices(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer índices espectrales conocidos para calidad de agua"""
        
        features = []
        names = []
        
        for index_name, (wavelength_pair, calc_type) in self.water_quality_indices.items():
            try:
                # Encontrar índices más cercanos a las wavelengths objetivo
                idx1 = np.argmin(np.abs(self.wavelengths - wavelength_pair[0]))
                idx2 = np.argmin(np.abs(self.wavelengths - wavelength_pair[1]))
                
                # Extraer valores espectrales
                val1 = spectra[:, idx1]
                val2 = spectra[:, idx2]
                
                # Calcular índice según el tipo
                if calc_type == 'normalized_difference':
                    # NDVI-style: (band1 - band2) / (band1 + band2)
                    index_values = np.divide(val1 - val2, val1 + val2 + 1e-8)
                elif calc_type == 'ratio':
                    # Simple ratio: band1 / band2
                    index_values = np.divide(val1, val2 + 1e-8)
                else:
                    index_values = np.zeros(len(spectra))
                
                features.append(index_values)
                names.append(index_name)
                
            except Exception:
                # Si hay error, añadir zeros como fallback
                features.append(np.zeros(len(spectra)))
                names.append(index_name)
        
        # Índices adicionales específicos para contaminantes
        try:
            # Blue/Red ratio (indicador de sedimentos)
            blue_idx = np.argmin(np.abs(self.wavelengths - 490))
            red_idx = np.argmin(np.abs(self.wavelengths - 670))
            blue_red_ratio = np.divide(spectra[:, blue_idx], spectra[:, red_idx] + 1e-8)
            features.append(blue_red_ratio)
            names.append('Blue_Red_Ratio')
            
            # Green/NIR ratio (indicador de vegetación acuática)
            green_idx = np.argmin(np.abs(self.wavelengths - 560))
            nir_idx = np.argmin(np.abs(self.wavelengths - 800))
            green_nir_ratio = np.divide(spectra[:, green_idx], spectra[:, nir_idx] + 1e-8)
            features.append(green_nir_ratio)
            names.append('Green_NIR_Ratio')
            
        except Exception:
            pass
        
        return np.column_stack(features), names
    
    def _extract_shape_features(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer features relacionadas con la forma espectral"""
        
        features = []
        names = []
        
        # Derivadas espectrales
        try:
            # Primera derivada
            first_deriv = np.gradient(spectra, axis=1)
            
            features.append(np.mean(first_deriv, axis=1))
            names.append('mean_first_derivative')
            
            features.append(np.std(first_deriv, axis=1))
            names.append('std_first_derivative')
            
            features.append(np.max(np.abs(first_deriv), axis=1))
            names.append('max_abs_first_derivative')
            
            # Segunda derivada (aproximada)
            second_deriv = np.gradient(first_deriv, axis=1)
            
            features.append(np.mean(np.abs(second_deriv), axis=1))
            names.append('mean_abs_second_derivative')
            
        except Exception:
            # Fallback si las derivadas fallan
            for name in ['mean_first_derivative', 'std_first_derivative', 
                        'max_abs_first_derivative', 'mean_abs_second_derivative']:
                features.append(np.zeros(len(spectra)))
                names.append(name)
        
        # Análisis de pendientes
        try:
            # Pendiente general (regresión lineal simple)
            x_vals = np.arange(len(self.wavelengths))
            slopes = []
            
            for spectrum in spectra:
                slope = np.polyfit(x_vals, spectrum, 1)[0]
                slopes.append(slope)
            
            features.append(np.array(slopes))
            names.append('overall_slope')
            
        except Exception:
            features.append(np.zeros(len(spectra)))
            names.append('overall_slope')
        
        # Features de curvatura
        try:
            # Curvatura promedio (segunda derivada normalizada)
            curvatures = []
            
            for spectrum in spectra:
                if len(spectrum) > 4:
                    # Suavizar primero para reducir ruido
                    smooth_spectrum = savgol_filter(spectrum, min(5, len(spectrum)//2*2+1), 2)
                    # Calcular curvatura aproximada
                    second_deriv = np.gradient(np.gradient(smooth_spectrum))
                    mean_curvature = np.mean(np.abs(second_deriv))
                    curvatures.append(mean_curvature)
                else:
                    curvatures.append(0.0)
            
            features.append(np.array(curvatures))
            names.append('mean_curvature')
            
        except Exception:
            features.append(np.zeros(len(spectra)))
            names.append('mean_curvature')
        
        return np.column_stack(features), names
    
    def _extract_range_features(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer features estadísticas por rangos espectrales"""
        
        features = []
        names = []
        
        for range_name, (min_wl, max_wl) in self.spectral_ranges.items():
            # Encontrar índices de wavelengths en este rango
            range_mask = (self.wavelengths >= min_wl) & (self.wavelengths <= max_wl)
            range_indices = np.where(range_mask)[0]
            
            if len(range_indices) > 0:
                # Extraer espectros de este rango
                range_spectra = spectra[:, range_indices]
                
                # Features estadísticas del rango
                features.append(np.mean(range_spectra, axis=1))
                names.append(f'{range_name}_mean')
                
                features.append(np.std(range_spectra, axis=1))
                names.append(f'{range_name}_std')
                
                features.append(np.max(range_spectra, axis=1))
                names.append(f'{range_name}_max')
                
                features.append(np.min(range_spectra, axis=1))
                names.append(f'{range_name}_min')
                
                # Rango espectral
                features.append(np.max(range_spectra, axis=1) - np.min(range_spectra, axis=1))
                names.append(f'{range_name}_range')
                
            else:
                # Si no hay wavelengths en el rango, añadir zeros
                for suffix in ['_mean', '_std', '_max', '_min', '_range']:
                    features.append(np.zeros(len(spectra)))
                    names.append(f'{range_name}{suffix}')
        
        return np.column_stack(features), names
    
    def _extract_peak_features(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer features relacionadas con picos y valles espectrales"""
        
        features = []
        names = []
        
        # Análisis de picos
        n_peaks_list = []
        peak_heights_mean = []
        peak_prominences_mean = []
        
        for spectrum in spectra:
            try:
                # Suavizar espectro para reducir ruido
                if len(spectrum) > 5:
                    smooth_spectrum = savgol_filter(spectrum, min(5, len(spectrum)//3*2+1), 2)
                else:
                    smooth_spectrum = spectrum
                
                # Encontrar picos
                peaks, properties = find_peaks(
                    smooth_spectrum, 
                    prominence=np.std(smooth_spectrum) * 0.1,  # Prominencia adaptativa
                    distance=max(1, len(spectrum) // 20)  # Distancia mínima entre picos
                )
                
                n_peaks_list.append(len(peaks))
                
                if len(peaks) > 0:
                    peak_heights_mean.append(np.mean(smooth_spectrum[peaks]))
                    if 'prominences' in properties:
                        peak_prominences_mean.append(np.mean(properties['prominences']))
                    else:
                        peak_prominences_mean.append(0.0)
                else:
                    peak_heights_mean.append(0.0)
                    peak_prominences_mean.append(0.0)
                    
            except Exception:
                n_peaks_list.append(0)
                peak_heights_mean.append(0.0)
                peak_prominences_mean.append(0.0)
        
        features.append(np.array(n_peaks_list))
        names.append('n_peaks')
        
        features.append(np.array(peak_heights_mean))
        names.append('mean_peak_height')
        
        features.append(np.array(peak_prominences_mean))
        names.append('mean_peak_prominence')
        
        # Análisis de valles (picos invertidos)
        n_valleys_list = []
        
        for spectrum in spectra:
            try:
                # Invertir espectro para encontrar valles
                inverted_spectrum = -spectrum
                
                if len(spectrum) > 5:
                    smooth_inverted = savgol_filter(inverted_spectrum, min(5, len(spectrum)//3*2+1), 2)
                else:
                    smooth_inverted = inverted_spectrum
                
                valleys, _ = find_peaks(
                    smooth_inverted,
                    prominence=np.std(smooth_inverted) * 0.1,
                    distance=max(1, len(spectrum) // 20)
                )
                
                n_valleys_list.append(len(valleys))
                
            except Exception:
                n_valleys_list.append(0)
        
        features.append(np.array(n_valleys_list))
        names.append('n_valleys')
        
        return np.column_stack(features), names
    
    def _extract_derivative_features(self, spectra: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extraer features basadas en análisis de derivadas espectrales"""
        
        features = []
        names = []
        
        try:
            # Primera derivada
            first_deriv = np.gradient(spectra, axis=1)
            
            # Puntos de inflexión (cambios de signo en primera derivada)
            inflection_points = []
            
            for deriv in first_deriv:
                # Contar cambios de signo
                sign_changes = np.sum(np.diff(np.sign(deriv)) != 0)
                inflection_points.append(sign_changes)
            
            features.append(np.array(inflection_points))
            names.append('n_inflection_points')
            
            # Máxima pendiente positiva y negativa
            max_positive_slope = np.max(first_deriv, axis=1)
            min_negative_slope = np.min(first_deriv, axis=1)
            
            features.append(max_positive_slope)
            names.append('max_positive_slope')
            
            features.append(np.abs(min_negative_slope))
            names.append('max_negative_slope')
            
            # Energía de la derivada (suma de cuadrados)
            derivative_energy = np.sum(first_deriv**2, axis=1)
            features.append(derivative_energy)
            names.append('derivative_energy')
            
        except Exception:
            # Fallback si las derivadas fallan
            for name in ['n_inflection_points', 'max_positive_slope', 
                        'max_negative_slope', 'derivative_energy']:
                features.append(np.zeros(len(spectra)))
                names.append(name)
        
        return np.column_stack(features), names
    
    def _clean_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar features: reemplazar NaN, inf, y valores extremos"""
        
        # Reemplazar inf con NaN
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        
        # Reemplazar NaN con la mediana de cada columna
        for col in feature_df.columns:
            if feature_df[col].isna().any():
                median_val = feature_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                feature_df[col] = feature_df[col].fillna(median_val)
        
        # Detectar y limitar valores extremos (outliers)
        for col in feature_df.columns:
            Q1 = feature_df[col].quantile(0.25)
            Q3 = feature_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Definir límites (más conservadores)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Recortar valores extremos
            feature_df[col] = feature_df[col].clip(lower_bound, upper_bound)
        
        return feature_df
    
    def _fallback_features(self, spectra: np.ndarray) -> pd.DataFrame:
        """Crear features básicas como fallback si todo falla"""
        
        print("    Usando features básicas como fallback...")
        
        features = []
        names = []
        
        # Solo estadísticas muy básicas
        features.append(np.mean(spectra, axis=1))
        names.append('mean_reflectance')
        
        features.append(np.std(spectra, axis=1))
        names.append('std_reflectance')
        
        features.append(np.min(spectra, axis=1))
        names.append('min_reflectance')
        
        features.append(np.max(spectra, axis=1))
        names.append('max_reflectance')
        
        # Algunas bandas específicas si están disponibles
        if len(self.wavelengths) > 10:
            # Primera banda
            features.append(spectra[:, 0])
            names.append(f'band_{self.wavelengths[0]:.0f}nm')
            
            # Banda media
            mid_idx = len(self.wavelengths) // 2
            features.append(spectra[:, mid_idx])
            names.append(f'band_{self.wavelengths[mid_idx]:.0f}nm')
            
            # Última banda
            features.append(spectra[:, -1])
            names.append(f'band_{self.wavelengths[-1]:.0f}nm')
        
        combined_features = np.column_stack(features)
        return pd.DataFrame(combined_features, columns=names)
    
    def get_feature_importance_analysis(self, features_df: pd.DataFrame, 
                                      target: np.ndarray) -> Dict:
        """
        Análisis de importancia de features espectrales
        
        Args:
            features_df: DataFrame con features extraídas
            target: Array con valores objetivo (concentraciones)
            
        Returns:
            Dict con análisis de importancia
        """
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.ensemble import RandomForestRegressor
            
            # Información mutua
            mi_scores = mutual_info_regression(features_df, target, random_state=42)
            
            # Importancia de Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features_df, target)
            rf_importance = rf.feature_importances_
            
            # Correlación de Pearson
            correlations = []
            for col in features_df.columns:
                try:
                    corr = np.corrcoef(features_df[col], target)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            
            # Crear ranking combinado
            feature_importance = pd.DataFrame({
                'feature': features_df.columns,
                'mutual_info': mi_scores,
                'rf_importance': rf_importance,
                'correlation': correlations
            })
            
            # Score combinado (normalizado)
            feature_importance['combined_score'] = (
                (feature_importance['mutual_info'] / (feature_importance['mutual_info'].max() + 1e-8)) * 0.4 +
                (feature_importance['rf_importance'] / (feature_importance['rf_importance'].max() + 1e-8)) * 0.4 +
                (feature_importance['correlation'] / (feature_importance['correlation'].max() + 1e-8)) * 0.2
            )
            
            # Ordenar por importancia
            feature_importance = feature_importance.sort_values('combined_score', ascending=False)
            
            return {
                'feature_ranking': feature_importance,
                'top_features': feature_importance.head(20)['feature'].tolist(),
                'analysis_summary': {
                    'total_features': len(features_df.columns),
                    'avg_mutual_info': np.mean(mi_scores),
                    'avg_correlation': np.mean(correlations),
                    'top_feature': feature_importance.iloc[0]['feature'],
                    'top_score': feature_importance.iloc[0]['combined_score']
                }
            }
            
        except Exception as e:
            print(f"    Error en análisis de importancia: {e}")
            return {
                'feature_ranking': pd.DataFrame(),
                'top_features': features_df.columns.tolist()[:20],
                'analysis_summary': {'error': str(e)}
            }

# ============================================================================
# FUNCIÓN AUXILIAR PARA COMPATIBILIDAD CON CÓDIGO EXISTENTE
# ============================================================================

def extract_spectral_features(spectra: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Función auxiliar para compatibilidad con código existente
    que usa extract_spectral_features directamente
    
    Args:
        spectra: Array de espectros (n_samples, n_wavelengths)
        wavelengths: Array de wavelengths (opcional)
        
    Returns:
        Array con features espectrales extraídas
    """
    
    if wavelengths is None:
        # Crear wavelengths por defecto si no se proporcionan
        wavelengths = np.arange(400, 400 + spectra.shape[1] * 2, 2)
    
    try:
        # Usar SpectralFeatureEngineer
        engineer = SpectralFeatureEngineer(wavelengths)
        features_df = engineer.extract_all_features(spectra)
        return features_df.values
        
    except Exception as e:
        print(f" Error en extract_spectral_features: {e}")
        print("   Usando features básicas como fallback...")
        
        # Fallback a features muy básicas
        features = []
        features.append(np.mean(spectra, axis=1))  # Media
        features.append(np.std(spectra, axis=1))   # Desviación estándar
        features.append(np.min(spectra, axis=1))   # Mínimo
        features.append(np.max(spectra, axis=1))   # Máximo
        
        # Añadir algunas bandas específicas si hay suficientes wavelengths
        if spectra.shape[1] > 10:
            # Primera, media y última banda
            features.append(spectra[:, 0])
            features.append(spectra[:, spectra.shape[1]//2])
            features.append(spectra[:, -1])
        
        return np.column_stack(features)

# ============================================================================
# FUNCIONES DE ANÁLISIS DE FIRMAS ESPECTRALES
# ============================================================================

class SpectralSignatureAnalyzer:
    """
    Analizador de firmas espectrales para identificación de contaminantes
    """
    
    def __init__(self, wavelengths: np.ndarray):
        self.wavelengths = wavelengths
        self.feature_engineer = SpectralFeatureEngineer(wavelengths)
        
    def create_contaminant_signature(self, spectra: np.ndarray, 
                                   concentrations: np.ndarray,
                                   contaminant_name: str) -> Dict:
        """
        Crear firma espectral característica para un contaminante
        
        Args:
            spectra: Espectros del contaminante (n_samples, n_wavelengths)
            concentrations: Concentraciones correspondientes
            contaminant_name: Nombre del contaminante
            
        Returns:
            Dict con la firma espectral
        """
        
        try:
            # Filtrar datos válidos
            valid_mask = ~np.any(np.isnan(spectra), axis=1) & ~np.isnan(concentrations)
            clean_spectra = spectra[valid_mask]
            clean_concentrations = concentrations[valid_mask]
            
            if len(clean_spectra) < 3:
                return None
            
            # Suavizar espectros
            smoothed_spectra = []
            for spectrum in clean_spectra:
                if len(spectrum) > 5:
                    smooth = savgol_filter(spectrum, min(5, len(spectrum)//3*2+1), 2)
                    smoothed_spectra.append(smooth)
                else:
                    smoothed_spectra.append(spectrum)
            
            smoothed_spectra = np.array(smoothed_spectra)
            
            # Estadísticas espectrales
            mean_spectrum = np.mean(smoothed_spectra, axis=0)
            std_spectrum = np.std(smoothed_spectra, axis=0)
            median_spectrum = np.median(smoothed_spectra, axis=0)
            
            # Encontrar picos característicos
            peaks, properties = find_peaks(
                mean_spectrum, 
                prominence=np.std(mean_spectrum) * 0.1,
                distance=max(1, len(mean_spectrum) // 20)
            )
            
            characteristic_peaks = []
            for i, peak_idx in enumerate(peaks):
                if i < len(properties['prominences']):
                    wavelength = self.wavelengths[peak_idx]
                    intensity = mean_spectrum[peak_idx]
                    prominence = properties['prominences'][i]
                    
                    characteristic_peaks.append({
                        'wavelength': float(wavelength),
                        'intensity': float(intensity),
                        'prominence': float(prominence),
                        'index': int(peak_idx)
                    })
            
            # Ordenar picos por prominencia
            characteristic_peaks.sort(key=lambda x: x['prominence'], reverse=True)
            
            # Wavelengths discriminantes (correlación con concentración)
            discriminant_wavelengths = []
            
            for i, wl in enumerate(self.wavelengths):
                try:
                    corr_coef = np.corrcoef(smoothed_spectra[:, i], clean_concentrations)[0, 1]
                    if not np.isnan(corr_coef) and abs(corr_coef) > 0.1:
                        discriminant_wavelengths.append({
                            'wavelength': float(wl),
                            'correlation': float(abs(corr_coef)),
                            'index': int(i)
                        })
                except:
                    continue
            
            # Ordenar por correlación
            discriminant_wavelengths.sort(key=lambda x: x['correlation'], reverse=True)
            
            # Extraer features espectrales avanzadas
            features_df = self.feature_engineer.extract_all_features(smoothed_spectra)
            
            # Análisis de importancia de features
            importance_analysis = self.feature_engineer.get_feature_importance_analysis(
                features_df, clean_concentrations
            )
            
            # Crear firma espectral completa
            signature = {
                'contaminant_name': contaminant_name,
                'creation_date': datetime.now().isoformat(),
                'n_samples': len(clean_spectra),
                'wavelength_range': [float(self.wavelengths.min()), float(self.wavelengths.max())],
                'spectral_data': {
                    'wavelengths': self.wavelengths.tolist(),
                    'mean_spectrum': mean_spectrum.tolist(),
                    'std_spectrum': std_spectrum.tolist(),
                    'median_spectrum': median_spectrum.tolist()
                },
                'characteristic_peaks': characteristic_peaks[:10],  # Top 10 picos
                'discriminant_wavelengths': discriminant_wavelengths[:20],  # Top 20 wavelengths
                'concentration_stats': {
                    'min': float(clean_concentrations.min()),
                    'max': float(clean_concentrations.max()),
                    'mean': float(clean_concentrations.mean()),
                    'std': float(clean_concentrations.std()),
                    'median': float(np.median(clean_concentrations))
                },
                'spectral_features': {
                    'top_features': importance_analysis.get('top_features', [])[:15],
                    'feature_summary': importance_analysis.get('analysis_summary', {})
                },
                'quality_metrics': self._calculate_signature_quality(
                    smoothed_spectra, characteristic_peaks, discriminant_wavelengths
                )
            }
            
            return signature
            
        except Exception as e:
            print(f"Error creando firma para {contaminant_name}: {e}")
            return None
    
    def _calculate_signature_quality(self, spectra: np.ndarray, 
                                   peaks: List[Dict], 
                                   discriminants: List[Dict]) -> Dict:
        """Calcular métricas de calidad de la firma espectral"""
        
        try:
            # Consistencia espectral
            cv_spectrum = np.std(spectra, axis=0) / (np.mean(spectra, axis=0) + 1e-8)
            consistency_score = max(0, 100 - np.mean(cv_spectrum) * 100)
            
            # Score por número de muestras
            n_samples = len(spectra)
            samples_score = min(100, (n_samples / 20) * 100)
            
            # Score por características espectrales
            peaks_score = min(100, len(peaks) * 10)
            discriminants_score = min(100, len(discriminants) * 5)
            
            # Rango dinámico
            mean_spectrum = np.mean(spectra, axis=0)
            dynamic_range = (np.max(mean_spectrum) - np.min(mean_spectrum)) / np.max(mean_spectrum)
            range_score = min(100, dynamic_range * 200)
            
            # Score total ponderado
            total_score = (
                consistency_score * 0.3 +
                samples_score * 0.25 +
                peaks_score * 0.2 +
                discriminants_score * 0.15 +
                range_score * 0.1
            )
            
            return {
                'overall_quality': min(100, total_score),
                'consistency_score': consistency_score,
                'samples_score': samples_score,
                'peaks_score': peaks_score,
                'discriminants_score': discriminants_score,
                'dynamic_range': dynamic_range,
                'n_samples': n_samples,
                'n_peaks': len(peaks),
                'n_discriminants': len(discriminants)
            }
            
        except Exception:
            return {
                'overall_quality': 0.0,
                'error': 'Could not calculate quality metrics'
            }

def compare_spectral_signatures(signature1: Dict, signature2: Dict) -> Dict:
    """
    Comparar dos firmas espectrales y calcular similitud
    
    Args:
        signature1, signature2: Firmas espectrales a comparar
        
    Returns:
        Dict con métricas de similitud
    """
    
    try:
        # Extraer espectros promedio
        spectrum1 = np.array(signature1['spectral_data']['mean_spectrum'])
        spectrum2 = np.array(signature2['spectral_data']['mean_spectrum'])
        
        # Verificar que tienen la misma longitud
        if len(spectrum1) != len(spectrum2):
            return {'error': 'Spectral signatures have different lengths'}
        
        # Similitud coseno
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = cosine_similarity([spectrum1], [spectrum2])[0, 0]
        
        # Correlación de Pearson
        pearson_corr = np.corrcoef(spectrum1, spectrum2)[0, 1]
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
        
        # Distancia euclidiana normalizada
        euclidean_dist = np.linalg.norm(spectrum1 - spectrum2)
        max_possible_dist = np.linalg.norm(spectrum1) + np.linalg.norm(spectrum2)
        normalized_euclidean = 1 - (euclidean_dist / (max_possible_dist + 1e-8))
        
        # Comparar picos característicos
        peaks1 = signature1.get('characteristic_peaks', [])
        peaks2 = signature2.get('characteristic_peaks', [])
        
        peak_similarity = 0.0
        if peaks1 and peaks2:
            # Contar picos en rangos similares (±10 nm)
            matching_peaks = 0
            for peak1 in peaks1[:5]:  # Top 5 picos
                wl1 = peak1['wavelength']
                for peak2 in peaks2[:5]:
                    wl2 = peak2['wavelength']
                    if abs(wl1 - wl2) <= 10:  # Tolerancia de 10 nm
                        matching_peaks += 1
                        break
            
            peak_similarity = matching_peaks / min(len(peaks1[:5]), len(peaks2[:5]))
        
        # Score combinado de similitud
        combined_similarity = (
            cosine_sim * 0.4 +
            abs(pearson_corr) * 0.3 +
            normalized_euclidean * 0.2 +
            peak_similarity * 0.1
        )
        
        return {
            'cosine_similarity': float(cosine_sim),
            'pearson_correlation': float(pearson_corr),
            'normalized_euclidean_similarity': float(normalized_euclidean),
            'peak_similarity': float(peak_similarity),
            'combined_similarity': float(combined_similarity),
            'similarity_level': _classify_similarity(combined_similarity),
            'contaminant1': signature1['contaminant_name'],
            'contaminant2': signature2['contaminant_name']
        }
        
    except Exception as e:
        return {'error': f'Error comparing signatures: {str(e)}'}

def _classify_similarity(similarity_score: float) -> str:
    """Clasificar nivel de similitud"""
    if similarity_score >= 0.9:
        return 'VERY_HIGH'
    elif similarity_score >= 0.8:
        return 'HIGH'
    elif similarity_score >= 0.6:
        return 'MEDIUM'
    elif similarity_score >= 0.4:
        return 'LOW'
    else:
        return 'VERY_LOW'

# ============================================================================
# FUNCIONES DE UTILIDAD Y EXPORTACIÓN
# ============================================================================

def save_spectral_signature(signature: Dict, output_dir: str = "spectral_signatures"):
    """Guardar firma espectral en archivo JSON"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Crear nombre de archivo seguro
    safe_name = signature['contaminant_name'].replace('/', '_').replace(' ', '_')
    filename = f"signature_{safe_name}.json"
    filepath = output_path / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(signature, f, indent=2, ensure_ascii=False)
        
        print(f" Firma espectral guardada: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f" Error guardando firma: {e}")
        return None

def load_spectral_signature(filepath: str) -> Optional[Dict]:
    """Cargar firma espectral desde archivo JSON"""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            signature = json.load(f)
        
        print(f" Firma espectral cargada: {signature['contaminant_name']}")
        return signature
        
    except Exception as e:
        print(f" Error cargando firma: {e}")
        return None

def create_spectral_library(signatures_dir: str = "spectral_signatures") -> Dict:
    """Crear biblioteca de firmas espectrales desde un directorio"""
    
    signatures_path = Path(signatures_dir)
    
    if not signatures_path.exists():
        print(f" Directorio no encontrado: {signatures_dir}")
        return {}
    
    library = {
        'creation_date': datetime.now().isoformat(),
        'library_version': '1.0',
        'signatures': {},
        'metadata': {
            'total_signatures': 0,
            'wavelength_ranges': [],
            'contaminant_types': []
        }
    }
    
    # Cargar todas las firmas
    signature_files = list(signatures_path.glob("signature_*.json"))
    
    for file_path in signature_files:
        signature = load_spectral_signature(file_path)
        
        if signature:
            contaminant_name = signature['contaminant_name']
            library['signatures'][contaminant_name] = signature
            
            # Actualizar metadata
            wl_range = signature.get('wavelength_range', [0, 0])
            library['metadata']['wavelength_ranges'].append(wl_range)
            library['metadata']['contaminant_types'].append(contaminant_name)
    
    library['metadata']['total_signatures'] = len(library['signatures'])
    
    # Guardar biblioteca
    library_file = signatures_path / "spectral_library_complete.json"
    try:
        with open(library_file, 'w', encoding='utf-8') as f:
            json.dump(library, f, indent=2, ensure_ascii=False)
        
        print(f" Biblioteca espectral creada: {library_file}")
        print(f"   Total firmas: {library['metadata']['total_signatures']}")
        
    except Exception as e:
        print(f" Error creando biblioteca: {e}")
    
    return library

def generate_spectral_report(signatures: Dict[str, Dict], output_dir: str = "spectral_reports"):
    """Generar reporte comprehensivo de firmas espectrales"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_lines = []
    
    # Encabezado
    report_lines.extend([
        "# REPORTE DE ANÁLISIS DE FIRMAS ESPECTRALES",
        "=" * 60,
        f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total de firmas:** {len(signatures)}",
        ""
    ])
    
    # Resumen estadístico
    if signatures:
        qualities = [sig.get('quality_metrics', {}).get('overall_quality', 0) for sig in signatures.values()]
        n_samples = [sig.get('n_samples', 0) for sig in signatures.values()]
        n_peaks = [len(sig.get('characteristic_peaks', [])) for sig in signatures.values()]
        
        report_lines.extend([
            "## RESUMEN ESTADÍSTICO",
            "-" * 25,
            f"- **Calidad promedio:** {np.mean(qualities):.1f}/100",
            f"- **Muestras promedio por firma:** {np.mean(n_samples):.1f}",
            f"- **Picos promedio por firma:** {np.mean(n_peaks):.1f}",
            f"- **Mejor firma:** {max(signatures.items(), key=lambda x: x[1].get('quality_metrics', {}).get('overall_quality', 0))[0]}",
            ""
        ])
        
        # Detalle por contaminante
        report_lines.extend([
            "## FIRMAS ESPECTRALES DETALLADAS",
            "-" * 35
        ])
        
        # Ordenar por calidad
        sorted_signatures = sorted(
            signatures.items(),
            key=lambda x: x[1].get('quality_metrics', {}).get('overall_quality', 0),
            reverse=True
        )
        
        for i, (name, signature) in enumerate(sorted_signatures, 1):
            quality = signature.get('quality_metrics', {}).get('overall_quality', 0)
            n_samples = signature.get('n_samples', 0)
            n_peaks = len(signature.get('characteristic_peaks', []))
            
            report_lines.extend([
                f"### {i}. {name}",
                f"- **Calidad:** {quality:.1f}/100",
                f"- **Muestras:** {n_samples}",
                f"- **Picos característicos:** {n_peaks}",
                ""
            ])
            
            # Top wavelengths discriminantes
            discriminants = signature.get('discriminant_wavelengths', [])[:5]
            if discriminants:
                wl_list = [f"{d['wavelength']:.0f}nm" for d in discriminants]
                report_lines.append(f"- **Top wavelengths:** {', '.join(wl_list)}")
                report_lines.append("")
    
    # Guardar reporte
    report_file = output_path / "spectral_analysis_report.md"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f" Reporte generado: {report_file}")
        return str(report_file)
        
    except Exception as e:
        print(f" Error generando reporte: {e}")
        return None

# ============================================================================
# SCRIPT PRINCIPAL PARA TESTING
# ============================================================================

if __name__ == "__main__":
    
    print(" SISTEMA DE ANÁLISIS ESPECTRAL")
    print("=" * 40)
    
    # Test básico con datos sintéticos
    print("\n Ejecutando test básico...")
    
    try:
        # Crear datos sintéticos para demostración
        wavelengths = np.arange(400, 800, 2)  # 400-800 nm cada 2 nm
        n_samples = 50
        n_wavelengths = len(wavelengths)
        
        # Generar espectros sintéticos
        base_spectrum = 0.3 + 0.1 * np.sin(2 * np.pi * wavelengths / 100)
        
        synthetic_spectra = []
        synthetic_concentrations = []
        
        for i in range(n_samples):
            # Simular concentración
            concentration = np.random.exponential(2.0)
            synthetic_concentrations.append(concentration)
            
            # Crear espectro con efectos de concentración
            spectrum = base_spectrum.copy()
            
            # Simular absorción específica
            absorption_center = 550  # nm
            absorption_strength = concentration * 0.05
            
            for j, wl in enumerate(wavelengths):
                if abs(wl - absorption_center) < 20:
                    spectrum[j] -= absorption_strength * np.exp(-((wl - absorption_center) / 10) ** 2)
            
            # Añadir ruido
            spectrum += 0.02 * np.random.randn(len(spectrum))
            synthetic_spectra.append(spectrum)
        
        synthetic_spectra = np.array(synthetic_spectra)
        synthetic_concentrations = np.array(synthetic_concentrations)
        
        print(f"    Datos sintéticos creados: {synthetic_spectra.shape}")
        
        # Test de SpectralFeatureEngineer
        print("\n Testing SpectralFeatureEngineer...")
        
        engineer = SpectralFeatureEngineer(wavelengths)
        features_df = engineer.extract_all_features(synthetic_spectra)
        
        print(f"    Features extraídas: {features_df.shape}")
        print(f"    Ejemplos de features: {list(features_df.columns[:5])}")
        
        # Test de análisis de importancia
        importance_analysis = engineer.get_feature_importance_analysis(
            features_df, synthetic_concentrations
        )
        
        if 'top_features' in importance_analysis:
            print(f"    Top 5 features: {importance_analysis['top_features'][:5]}")
        
        # Test de SpectralSignatureAnalyzer
        print("\n Testing SpectralSignatureAnalyzer...")
        
        analyzer = SpectralSignatureAnalyzer(wavelengths)
        signature = analyzer.create_contaminant_signature(
            synthetic_spectra, synthetic_concentrations, "synthetic_contaminant"
        )
        
        if signature:
            quality = signature['quality_metrics']['overall_quality']
            n_peaks = len(signature['characteristic_peaks'])
            print(f"    Firma espectral creada: calidad {quality:.1f}/100, {n_peaks} picos")
            
            # Guardar firma de prueba
            saved_path = save_spectral_signature(signature, "test_signatures")
            if saved_path:
                print(f"    Firma guardada en: {saved_path}")
        
        # Test de función de compatibilidad
        print("\n Testing función de compatibilidad...")
        
        compat_features = extract_spectral_features(synthetic_spectra, wavelengths)
        print(f"    Features de compatibilidad: {compat_features.shape}")
        
        print(f"\n Todos los tests completados exitosamente!")
        print(f"    SpectralFeatureEngineer está listo para usar con ml_ready_dataset_generator.py")
        
    except Exception as e:
        print(f"\n Error en testing: {e}")
        import traceback
        traceback.print_exc()
    