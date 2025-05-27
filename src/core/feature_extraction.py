"""
Módulo de extracción de características espectrales.

Este módulo contiene todas las funciones para extraer características relevantes
de datos espectrales para machine learning.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.feature_selection import SelectKBest, f_classif


class SpectralFeatureExtractor:
    """Clase principal para extracción de características espectrales."""
    
    def __init__(self, config=None):
        """
        Inicializa el extractor con configuración específica.
        
        Args:
            config (dict): Configuración de extracción de características
        """
        self.config = config or {}
        self.top_features = self.config.get('top_features', 15)
        self.p_value_threshold = self.config.get('p_value_threshold', 0.05)
    
    def extract_spectral_features(self, spectrum, wavelengths):
        """
        Extrae características espectrales de un espectro.
        
        Args:
            spectrum (np.ndarray): Espectro a analizar
            wavelengths (list): Lista de longitudes de onda
            
        Returns:
            dict: Diccionario con características extraídas
        """
        features = {}
        
        # Estadísticos básicos
        features['mean'] = np.mean(spectrum)
        features['std'] = np.std(spectrum)
        features['max'] = np.max(spectrum)
        features['min'] = np.min(spectrum)
        features['range'] = np.max(spectrum) - np.min(spectrum)
        features['median'] = np.median(spectrum)
        
        # Calcular skewness y kurtosis
        if len(spectrum) > 2:
            std = np.std(spectrum)
            if std > 0:
                features['skewness'] = np.mean(((spectrum - np.mean(spectrum)) / std)**3)
                features['kurtosis'] = np.mean(((spectrum - np.mean(spectrum)) / std)**4) - 3
            else:
                features['skewness'] = 0
                features['kurtosis'] = 0
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0
        
        # Picos y valles
        if len(spectrum) > 3:
            peaks, _ = signal.find_peaks(spectrum, height=np.mean(spectrum), distance=5)
            valleys, _ = signal.find_peaks(-spectrum, height=-np.mean(spectrum), distance=5)
            
            features['n_peaks'] = len(peaks)
            features['n_valleys'] = len(valleys)
            
            if len(peaks) > 0:
                features['peak_max'] = np.max(spectrum[peaks])
                features['peak_mean'] = np.mean(spectrum[peaks])
                features['peak_std'] = np.std(spectrum[peaks])
                features['peak_wavelength_mean'] = np.mean([wavelengths[p] for p in peaks if p < len(wavelengths)])
            else:
                features['peak_max'] = 0
                features['peak_mean'] = 0
                features['peak_std'] = 0
                features['peak_wavelength_mean'] = 0
            
            if len(valleys) > 0:
                features['valley_min'] = np.min(spectrum[valleys])
                features['valley_mean'] = np.mean(spectrum[valleys])
                features['valley_std'] = np.std(spectrum[valleys])
                features['valley_wavelength_mean'] = np.mean([wavelengths[v] for v in valleys if v < len(wavelengths)])
            else:
                features['valley_min'] = 0
                features['valley_mean'] = 0
                features['valley_std'] = 0
                features['valley_wavelength_mean'] = 0
        else:
            features['n_peaks'] = 0
            features['n_valleys'] = 0
            features['peak_max'] = 0
            features['peak_mean'] = 0
            features['peak_std'] = 0
            features['peak_wavelength_mean'] = 0
            features['valley_min'] = 0
            features['valley_mean'] = 0
            features['valley_std'] = 0
            features['valley_wavelength_mean'] = 0
        
        # Pendientes en diferentes regiones del espectro
        features.update(self._calculate_regional_slopes(spectrum, wavelengths))
        
        # Áreas bajo la curva para diferentes regiones
        features.update(self._calculate_regional_areas(spectrum, wavelengths))
        
        return features

    def _calculate_regional_slopes(self, spectrum, wavelengths):
        """Calcula pendientes para diferentes regiones espectrales."""
        features = {}
        
        # Región UV (< 400 nm)
        uv_indices = [i for i, wl in enumerate(wavelengths) if wl < 400]
        if len(uv_indices) > 1:
            region_wavelengths = [wavelengths[i] for i in uv_indices]
            region_spectrum = [spectrum[i] for i in uv_indices]
            
            if len(region_wavelengths) == len(region_spectrum) and len(region_wavelengths) > 1:
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(region_wavelengths, region_spectrum)
                    features['uv_slope'] = slope
                    features['uv_intercept'] = intercept
                    features['uv_r_squared'] = r_value**2
                except:
                    features['uv_slope'] = 0
                    features['uv_intercept'] = 0
                    features['uv_r_squared'] = 0
            else:
                features['uv_slope'] = 0
                features['uv_intercept'] = 0
                features['uv_r_squared'] = 0
        else:
            features['uv_slope'] = 0
            features['uv_intercept'] = 0
            features['uv_r_squared'] = 0
        
        # Región visible (400-700 nm)
        vis_indices = [i for i, wl in enumerate(wavelengths) if 400 <= wl <= 700]
        if len(vis_indices) > 1:
            region_wavelengths = [wavelengths[i] for i in vis_indices]
            region_spectrum = [spectrum[i] for i in vis_indices]
            
            if len(region_wavelengths) == len(region_spectrum) and len(region_wavelengths) > 1:
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(region_wavelengths, region_spectrum)
                    features['vis_slope'] = slope
                    features['vis_intercept'] = intercept
                    features['vis_r_squared'] = r_value**2
                except:
                    features['vis_slope'] = 0
                    features['vis_intercept'] = 0
                    features['vis_r_squared'] = 0
            else:
                features['vis_slope'] = 0
                features['vis_intercept'] = 0
                features['vis_r_squared'] = 0
        else:
            features['vis_slope'] = 0
            features['vis_intercept'] = 0
            features['vis_r_squared'] = 0
        
        # Región NIR (> 700 nm)
        nir_indices = [i for i, wl in enumerate(wavelengths) if wl > 700]
        if len(nir_indices) > 1:
            region_wavelengths = [wavelengths[i] for i in nir_indices]
            region_spectrum = [spectrum[i] for i in nir_indices]
            
            if len(region_wavelengths) == len(region_spectrum) and len(region_wavelengths) > 1:
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(region_wavelengths, region_spectrum)
                    features['nir_slope'] = slope
                    features['nir_intercept'] = intercept
                    features['nir_r_squared'] = r_value**2
                except:
                    features['nir_slope'] = 0
                    features['nir_intercept'] = 0
                    features['nir_r_squared'] = 0
            else:
                features['nir_slope'] = 0
                features['nir_intercept'] = 0
                features['nir_r_squared'] = 0
        else:
            features['nir_slope'] = 0
            features['nir_intercept'] = 0
            features['nir_r_squared'] = 0
        
        return features

    def _calculate_regional_areas(self, spectrum, wavelengths):
        """Calcula áreas bajo la curva para diferentes regiones."""
        features = {}
        
        # Región UV (< 400 nm)
        uv_indices = [i for i, wl in enumerate(wavelengths) if wl < 400]
        if len(uv_indices) > 1:
            features['uv_area'] = np.trapz([spectrum[i] for i in uv_indices], [wavelengths[i] for i in uv_indices])
        else:
            features['uv_area'] = 0
        
        # Región visible (400-700 nm)
        vis_indices = [i for i, wl in enumerate(wavelengths) if 400 <= wl <= 700]
        if len(vis_indices) > 1:
            features['vis_area'] = np.trapz([spectrum[i] for i in vis_indices], [wavelengths[i] for i in vis_indices])
        else:
            features['vis_area'] = 0
        
        # Región NIR (> 700 nm)
        nir_indices = [i for i, wl in enumerate(wavelengths) if wl > 700]
        if len(nir_indices) > 1:
            features['nir_area'] = np.trapz([spectrum[i] for i in nir_indices], [wavelengths[i] for i in nir_indices])
        else:
            features['nir_area'] = 0
        
        return features

    def calculate_band_ratios(self, spectrum, wavelengths):
        """
        Calcula ratios entre bandas específicas del espectro.
        
        Args:
            spectrum (np.ndarray): Espectro a analizar
            wavelengths (list): Lista de longitudes de onda
            
        Returns:
            dict: Diccionario con ratios entre bandas
        """
        wavelengths_array = np.array(wavelengths)
        spectrum_array = np.array(spectrum)
        
        ratios = {}
        
        # Definir bandas clave
        key_bands = {
            'blue': (450, 490),
            'green': (520, 560),
            'red': (630, 670),
            'nir': (750, 800)
        }
        
        band_values = {}
        
        # Calcular valor medio para cada banda
        for band_name, (start_wl, end_wl) in key_bands.items():
            band_indices = np.where((wavelengths_array >= start_wl) & (wavelengths_array <= end_wl))[0]
            
            if len(band_indices) > 0:
                band_values[band_name] = np.mean(spectrum_array[band_indices])
            else:
                band_values[band_name] = 0
        
        # Calcular ratios entre bandas
        for band1 in band_values:
            for band2 in band_values:
                if band1 < band2:  # Evitar duplicados
                    if band_values[band2] != 0:  # Evitar división por cero
                        ratio_name = f'ratio_{band1}_{band2}'
                        ratios[ratio_name] = band_values[band1] / band_values[band2]
        
        # Calcular índices específicos para calidad de agua
        
        # Índice de vegetación (NDVI adaptado)
        if 'red' in band_values and 'nir' in band_values and (band_values['red'] + band_values['nir']) != 0:
            ratios['ndvi'] = (band_values['nir'] - band_values['red']) / (band_values['nir'] + band_values['red'])
        else:
            ratios['ndvi'] = 0
        
        # Índice de clorofila
        if 'red' in band_values and 'nir' in band_values and band_values['red'] != 0:
            ratios['chlorophyll_index'] = band_values['nir'] / band_values['red'] - 1
        else:
            ratios['chlorophyll_index'] = 0
        
        # Índice de turbidez
        if 'red' in band_values and 'green' in band_values and band_values['green'] != 0:
            ratios['turbidity_index'] = band_values['red'] / band_values['green']
        else:
            ratios['turbidity_index'] = 0
        
        # Calcular ratios entre longitudes de onda específicas
        specific_pairs = [
            (450, 550),  # Azul-Verde
            (550, 650),  # Verde-Rojo
            (650, 750),  # Rojo-NIR
            (450, 750)   # Azul-NIR
        ]
        
        for wl1, wl2 in specific_pairs:
            idx1 = np.argmin(np.abs(wavelengths_array - wl1))
            idx2 = np.argmin(np.abs(wavelengths_array - wl2))
            
            if spectrum_array[idx2] != 0:
                ratio_name = f'ratio_{wl1}_{wl2}'
                ratios[ratio_name] = spectrum_array[idx1] / spectrum_array[idx2]
        
        return ratios

    def calculate_derivative_features(self, spectrum):
        """
        Calcula características basadas en derivadas del espectro.
        
        Args:
            spectrum: Array del espectro
            
        Returns:
            dict: Características de derivadas
        """
        features = {}
        
        # Primera derivada
        first_derivative = np.gradient(spectrum)
        features['first_deriv_mean'] = np.mean(first_derivative)
        features['first_deriv_std'] = np.std(first_derivative)
        features['first_deriv_max'] = np.max(first_derivative)
        features['first_deriv_min'] = np.min(first_derivative)
        
        # Segunda derivada
        second_derivative = np.gradient(first_derivative)
        features['second_deriv_mean'] = np.mean(second_derivative)
        features['second_deriv_std'] = np.std(second_derivative)
        features['second_deriv_max'] = np.max(second_derivative)
        features['second_deriv_min'] = np.min(second_derivative)
        
        return features

    def calculate_moment_features(self, spectrum):
        """
        Calcula momentos estadísticos del espectro.
        
        Args:
            spectrum: Array del espectro
            
        Returns:
            dict: Características de momentos
        """
        features = {}
        
        # Momentos centrales
        mean_val = np.mean(spectrum)
        
        # Segundo momento (varianza)
        features['variance'] = np.var(spectrum)
        
        # Tercer momento (asimetría)
        if np.std(spectrum) > 0:
            features['moment_3'] = np.mean((spectrum - mean_val)**3) / (np.std(spectrum)**3)
        else:
            features['moment_3'] = 0
        
        # Cuarto momento (curtosis)
        if np.std(spectrum) > 0:
            features['moment_4'] = np.mean((spectrum - mean_val)**4) / (np.std(spectrum)**4)
        else:
            features['moment_4'] = 0
        
        return features

    def calculate_frequency_features(self, spectrum):
        """
        Calcula características en el dominio de la frecuencia.
        
        Args:
            spectrum: Array del espectro
            
        Returns:
            dict: Características de frecuencia
        """
        features = {}
        
        # Transformada de Fourier
        fft_vals = np.fft.fft(spectrum)
        fft_magnitude = np.abs(fft_vals)
        
        # Características básicas del espectro de frecuencia
        features['fft_mean'] = np.mean(fft_magnitude)
        features['fft_std'] = np.std(fft_magnitude)
        features['fft_max'] = np.max(fft_magnitude)
        
        # Energía espectral
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        
        # Centroide espectral
        freqs = np.fft.fftfreq(len(spectrum))
        if np.sum(fft_magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        else:
            features['spectral_centroid'] = 0
        
        return features

    def select_best_features(self, features_df, target, k=None):
        """
        Selecciona las k mejores características basándose en ANOVA F-value.
        
        Args:
            features_df: DataFrame con características extraídas
            target: Variable objetivo (etiquetas de contaminantes)
            k: Número de características a seleccionar
            
        Returns:
            tuple: (DataFrame con mejores características, scores, nombres seleccionados)
        """
        if k is None:
            k = self.top_features
        
        # Eliminar columnas con valores constantes o NaN
        features_df = features_df.dropna(axis=1)
        constant_cols = [col for col in features_df.columns if features_df[col].nunique() <= 1]
        features_df = features_df.drop(columns=constant_cols)
        
        # Seleccionar las mejores características
        selector = SelectKBest(f_classif, k=min(k, features_df.shape[1]))
        X_selected = selector.fit_transform(features_df, target)
        
        # Obtener los índices de las características seleccionadas
        selected_indices = selector.get_support(indices=True)
        selected_features = features_df.columns[selected_indices]
        
        # Crear DataFrame con características seleccionadas
        selected_df = pd.DataFrame(X_selected, columns=selected_features)
        
        # Scores de características
        feature_scores = pd.DataFrame({
            'Feature': features_df.columns,
            'Score': selector.scores_,
            'P-value': selector.pvalues_
        }).sort_values(by='Score', ascending=False)
        
        print(f"Seleccionadas {len(selected_features)} características de {features_df.shape[1]} originales")
        print("Top 10 características:")
        print(feature_scores.head(10))
        
        return selected_df, feature_scores, selected_features

    def extract_complete_feature_set(self, spectrum, wavelengths):
        """
        Extrae un conjunto completo de características de un espectro.
        
        Args:
            spectrum: Array del espectro
            wavelengths: Array de longitudes de onda
            
        Returns:
            dict: Diccionario con todas las características
        """
        all_features = {}
        
        # Características espectrales básicas
        all_features.update(self.extract_spectral_features(spectrum, wavelengths))
        
        # Ratios entre bandas
        all_features.update(self.calculate_band_ratios(spectrum, wavelengths))
        
        # Características de derivadas
        all_features.update(self.calculate_derivative_features(spectrum))
        
        # Momentos estadísticos
        all_features.update(self.calculate_moment_features(spectrum))
        
        # Características de frecuencia
        all_features.update(self.calculate_frequency_features(spectrum))
        
        return all_features

    def extract_features_from_dataset(self, spectra_matrix, wavelengths):
        """
        Extrae características de una matriz completa de espectros.
        
        Args:
            spectra_matrix: Matriz de espectros (filas=muestras, columnas=longitudes de onda)
            wavelengths: Array de longitudes de onda
            
        Returns:
            pd.DataFrame: DataFrame con características extraídas
        """
        print(f"Extrayendo características de {spectra_matrix.shape[0]} espectros...")
        
        all_features = []
        
        for i in range(spectra_matrix.shape[0]):
            spectrum = spectra_matrix[i, :]
            features = self.extract_complete_feature_set(spectrum, wavelengths)
            all_features.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"Procesados {i + 1}/{spectra_matrix.shape[0]} espectros")
        
        features_df = pd.DataFrame(all_features)
        print(f"Extracción completada. {features_df.shape[1]} características extraídas.")
        
        return features_df

    def create_spectral_signature(self, high_concentration_spectra, low_concentration_spectra, wavelengths):
        """
        Crea una firma espectral como diferencia entre alta y baja concentración.
        
        Args:
            high_concentration_spectra: Espectros de alta concentración
            low_concentration_spectra: Espectros de baja concentración
            wavelengths: Longitudes de onda
            
        Returns:
            dict: Firma espectral y metadatos
        """
        high_mean = np.mean(high_concentration_spectra, axis=0)
        low_mean = np.mean(low_concentration_spectra, axis=0)
        
        signature = high_mean - low_mean
        
        # Calcular estadísticas de la firma
        signature_stats = {
            'signature': signature,
            'high_mean': high_mean,
            'low_mean': low_mean,
            'wavelengths': wavelengths,
            'max_difference': np.max(signature),
            'min_difference': np.min(signature),
            'mean_difference': np.mean(signature),
            'std_difference': np.std(signature)
        }
        
        # Encontrar longitudes de onda con mayor diferencia
        max_diff_idx = np.argmax(np.abs(signature))
        signature_stats['peak_wavelength'] = wavelengths[max_diff_idx]
        signature_stats['peak_difference'] = signature[max_diff_idx]
        
        return signature_stats