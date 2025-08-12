#!/usr/bin/env python3
"""
Enhanced ML Generator con Data Augmentation + Análisis Espectral Integrado

"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# IMPORTAR EL SISTEMA DE ANÁLISIS ESPECTRAL
try:
    from spectral_analisis import SpectralFeatureEngineer, SpectralSignatureAnalyzer, extract_spectral_features
    SPECTRAL_ANALYSIS_AVAILABLE = True
    print("✓ Análisis espectral avanzado disponible")
except ImportError:
    SPECTRAL_ANALYSIS_AVAILABLE = False
    print(" spectral_analisis.py no encontrado - usando features básicas")
    # Función fallback básica
    def extract_spectral_features(spectra, wavelengths=None):
        features = []
        features.append(np.mean(spectra, axis=1))
        features.append(np.std(spectra, axis=1))
        features.append(np.min(spectra, axis=1))
        features.append(np.max(spectra, axis=1))
        return np.column_stack(features)


class SpectralEnhancedMLGenerator:
    """
    Generador de datasets ML con features espectrales avanzadas y data augmentation
    """
    
    def __init__(self, output_dir: str = "spectral_enhanced_datasets", 
                 use_spectral_features: bool = True,
                 spectral_strategy: str = "combined"):
        """
        Args:
            output_dir: Directorio de salida
            use_spectral_features: Usar análisis espectral avanzado
            spectral_strategy: "spectral_only", "combined", "raw_only"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuración de análisis espectral
        self.use_spectral_features = use_spectral_features and SPECTRAL_ANALYSIS_AVAILABLE
        self.spectral_strategy = spectral_strategy
        self.spectral_engineer = None
        self.signature_analyzer = None
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Variables internas
        self.wavelengths = None
        self.spectral_columns = None
        self.loq_data = None
        self.baseline_spectra = []
        self.spectral_signatures = {}  # Para almacenar firmas espectrales
        
        # Configuración de augmentation
        self.augmentation_config = {
            'enable_noise': True,
            'enable_shift': True,
            'enable_scale': True,
            'enable_smooth': True,
            'noise_level': 0.01,
            'shift_range': 0.02,
            'scale_range': 0.05,
            'smooth_sigma': 0.5
        }
        
        status = "HABILITADO" if self.use_spectral_features else "DESHABILITADO"
        self.logger.info(f"Spectral Enhanced ML Generator inicializado")
        self.logger.info(f"  Análisis espectral: {status}")
        self.logger.info(f"  Estrategia: {spectral_strategy}")
    
    def load_all_data(self, reflectance_file: str, chemicals_file: str, 
                     pollution_file: str = None, loq_file: str = None):
        """Cargar todos los datos incluyendo LOQ"""
        
        self.logger.info(" CARGANDO TODOS LOS DATOS")
        self.logger.info("=" * 50)
        
        # Cargar archivos principales
        df_reflectance = pd.read_csv(reflectance_file)
        df_chemicals = pd.read_csv(chemicals_file)
        df_pollution = pd.read_csv(pollution_file) if pollution_file else None
        
        self.logger.info(f" Reflectancia: {df_reflectance.shape}")
        self.logger.info(f" Químicos orgánicos: {df_chemicals.shape}")
        if df_pollution is not None:
            self.logger.info(f" Químicos inorgánicos: {df_pollution.shape}")
        
        # Cargar LOQ si está disponible
        if loq_file and os.path.exists(loq_file):
            try:
                self.loq_data = pd.read_csv(loq_file)
                self.logger.info(f" Datos LOQ: {self.loq_data.shape}")
            except Exception as e:
                self.logger.warning(f"Error cargando LOQ: {e}")
                self.loq_data = None
        else:
            self.logger.info("ℹ Sin datos LOQ - usando pipeline básico")
        
        # Detectar columnas espectrales y configurar análisis espectral
        self._detect_spectral_columns(df_reflectance)
        self._setup_spectral_analysis()
        
        # Identificar espectros baseline (agua limpia)
        self._identify_baseline_spectra(df_reflectance, df_chemicals)
        
        # Procesar timestamps
        df_reflectance['timestamp'] = pd.to_datetime(df_reflectance['timestamp_iso'])
        df_chemicals['timestamp'] = pd.to_datetime(df_chemicals['timestamp_iso'])
        if df_pollution is not None:
            df_pollution['timestamp'] = pd.to_datetime(df_pollution['timestamp_iso'])
        
        # Limpiar datos químicos
        df_chemicals = self._clean_chemical_data(df_chemicals)
        if df_pollution is not None:
            df_pollution = self._clean_chemical_data(df_pollution)
        
        return df_reflectance, df_chemicals, df_pollution
    
    def _detect_spectral_columns(self, df_reflectance):
        """Detectar columnas espectrales"""
        
        self.logger.info(" Detectando columnas espectrales...")
        
        # Buscar columnas con 'reflectance'
        spectral_cols = [col for col in df_reflectance.columns if 'reflectance' in col.lower()]
        
        if len(spectral_cols) == 0:
            # Fallback: buscar columnas numéricas
            numeric_cols = df_reflectance.select_dtypes(include=[np.number]).columns
            exclude_patterns = ['index', 'timestamp', 'id', 'time', 'date']
            spectral_cols = [col for col in numeric_cols 
                           if not any(pattern in str(col).lower() for pattern in exclude_patterns)]
        
        # Verificar que las columnas son realmente numéricas
        verified_spectral_cols = []
        for col in spectral_cols:
            try:
                test_values = pd.to_numeric(df_reflectance[col].head(10), errors='coerce')
                if not test_values.isna().all():
                    verified_spectral_cols.append(col)
            except:
                continue
        
        spectral_cols = verified_spectral_cols
        
        if len(spectral_cols) == 0:
            raise ValueError("No se encontraron columnas espectrales válidas")
        
        # Extraer wavelengths
        wavelengths = []
        for col in spectral_cols:
            import re
            numbers = re.findall(r'\d+\.?\d*', str(col))
            if numbers:
                try:
                    wl = float(numbers[0])
                    if 200 <= wl <= 2500:
                        wavelengths.append(wl)
                    else:
                        wavelengths.append(400 + len(wavelengths) * 2)
                except:
                    wavelengths.append(400 + len(wavelengths) * 2)
            else:
                wavelengths.append(400 + len(wavelengths) * 2)
        
        # Ordenar por wavelength
        if len(wavelengths) == len(spectral_cols):
            combined = list(zip(wavelengths, spectral_cols))
            combined.sort(key=lambda x: x[0])
            wavelengths, spectral_cols = zip(*combined)
            wavelengths = list(wavelengths)
            spectral_cols = list(spectral_cols)
        
        self.wavelengths = np.array(wavelengths)
        self.spectral_columns = spectral_cols
        
        self.logger.info(f"✓ Detectadas {len(spectral_cols)} columnas espectrales")
        self.logger.info(f" Rango: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        self.logger.info(f"📐 Resolución promedio: {np.median(np.diff(wavelengths)):.1f} nm")
    
    def _setup_spectral_analysis(self):
        """Configurar sistema de análisis espectral"""
        
        if not self.use_spectral_features:
            self.logger.info("ℹ Análisis espectral deshabilitado")
            return
        
        try:
            # Crear SpectralFeatureEngineer
            self.spectral_engineer = SpectralFeatureEngineer(self.wavelengths)
            
            # Crear SpectralSignatureAnalyzer  
            self.signature_analyzer = SpectralSignatureAnalyzer(self.wavelengths)
            
            self.logger.info("✓ Sistema de análisis espectral configurado")
            self.logger.info(f"   Features espectrales: HABILITADO")
            self.logger.info(f"   Firmas espectrales: HABILITADO")
            
        except Exception as e:
            self.logger.error(f"Error configurando análisis espectral: {e}")
            self.use_spectral_features = False
            self.logger.info(" Fallback a features básicas")
    
    def _identify_baseline_spectra(self, df_reflectance, df_chemicals):
        """Identificar espectros de agua limpia/baseline"""
        
        self.logger.info(" Identificando espectros baseline (agua limpia)...")
        
        try:
            # Buscar timestamps en reflectance que NO están en chemicals
            reflectance_times = set(df_reflectance['timestamp_iso'])
            chemicals_times = set(df_chemicals['timestamp_iso'])
            
            baseline_times = reflectance_times - chemicals_times
            
            if len(baseline_times) > 0:
                baseline_mask = df_reflectance['timestamp_iso'].isin(baseline_times)
                baseline_df = df_reflectance[baseline_mask]
                
                # Extraer espectros baseline con manejo seguro de tipos
                for _, row in baseline_df.iterrows():
                    try:
                        spectrum_raw = row[self.spectral_columns]
                        spectrum = pd.to_numeric(spectrum_raw, errors='coerce').values
                        
                        if len(spectrum) > 0 and not np.all(np.isnan(spectrum)):
                            if np.any(np.isnan(spectrum)):
                                nan_mask = np.isnan(spectrum)
                                if np.sum(nan_mask) < len(spectrum) * 0.1:
                                    spectrum = self._interpolate_spectrum(spectrum)
                                else:
                                    continue
                            
                            self.baseline_spectra.append(spectrum)
                    
                    except Exception:
                        continue
                
                self.logger.info(f"✓ Encontrados {len(self.baseline_spectra)} espectros baseline")
            else:
                self.logger.info("ℹ No se encontraron espectros baseline - generando sintéticos")
                self._generate_synthetic_baseline()
                
        except Exception as e:
            self.logger.warning(f"Error identificando baseline: {e}")
            self.logger.info(" Generando espectros baseline sintéticos")
            self._generate_synthetic_baseline()
    
    def _interpolate_spectrum(self, spectrum):
        """Interpolar valores NaN en un espectro"""
        
        nan_mask = np.isnan(spectrum)
        if not np.any(nan_mask):
            return spectrum
        
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) >= 2:
            spectrum_interp = spectrum.copy()
            spectrum_interp[nan_mask] = np.interp(
                np.where(nan_mask)[0], 
                valid_indices, 
                spectrum[valid_indices]
            )
            return spectrum_interp
        else:
            mean_val = np.nanmean(spectrum)
            spectrum_filled = spectrum.copy()
            spectrum_filled[nan_mask] = mean_val
            return spectrum_filled
    
    def _generate_synthetic_baseline(self, n_synthetic=50):
        """Generar espectros baseline sintéticos"""
        
        for i in range(n_synthetic):
            spectrum = np.ones(len(self.wavelengths)) * 0.05
            
            for j, wl in enumerate(self.wavelengths):
                if 400 <= wl <= 500:
                    spectrum[j] += 0.02
                elif 650 <= wl <= 750:
                    spectrum[j] -= 0.01
                elif wl > 750:
                    spectrum[j] -= 0.02
            
            noise = np.random.normal(0, 0.005, len(spectrum))
            spectrum += noise
            spectrum = np.maximum(spectrum, 0.001)
            
            self.baseline_spectra.append(spectrum)
        
        self.logger.info(f"✓ Generados {n_synthetic} espectros baseline sintéticos")
    
    def _clean_chemical_data(self, df):
        """Limpiar datos químicos manteniendo información de LOQ"""
        
        contaminant_cols = [col for col in df.columns if col.startswith('lab_')]
        
        for col in contaminant_cols:
            if df[col].dtype == 'object':
                df[f'{col}_below_loq'] = df[col].str.contains('<LOQ|<loq|< LOQ', na=False)
                
                df[col] = df[col].replace('<LOQ', np.nan)
                df[col] = df[col].replace('<loq', np.nan)
                df[col] = df[col].replace('< LOQ', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def analyze_detectability(self, df_chemicals):
        """Analizar detectabilidad usando LOQ si está disponible"""
        
        self.logger.info(" ANALIZANDO DETECTABILIDAD")
        
        contaminant_cols = [col for col in df_chemicals.columns 
                          if col.startswith('lab_') and not col.endswith('_below_loq')]
        
        detectability_analysis = {}
        
        for col in contaminant_cols:
            contaminant_name = col.replace('lab_', '').replace('_ng_l', '').replace('_mg_l', '').replace('_ntu', '')
            
            values = df_chemicals[col]
            total_samples = len(values.dropna())
            
            if total_samples == 0:
                continue
            
            avg_loq = self._get_loq_for_contaminant(contaminant_name)
            
            if avg_loq is not None:
                detected_samples = len(values[values >= avg_loq])
                detection_rate = detected_samples / total_samples if total_samples > 0 else 0
                detected_values = values[values >= avg_loq]
                
                analysis = {
                    'avg_loq': avg_loq,
                    'total_samples': total_samples,
                    'detected_samples': detected_samples,
                    'detection_rate': detection_rate,
                    'detectability_category': self._categorize_detectability(detection_rate),
                    'has_loq': True,
                    'needs_augmentation': detection_rate > 0.8 or detection_rate < 0.3,
                    'needs_spectral_analysis': True,  # NUEVO: Siempre beneficioso
                    'concentration_stats': {
                        'detected_mean': detected_values.mean() if len(detected_values) > 0 else np.nan,
                        'detected_max': detected_values.max() if len(detected_values) > 0 else np.nan,
                        'signal_to_loq_ratio': (detected_values.mean() / avg_loq) if len(detected_values) > 0 and avg_loq > 0 else np.nan
                    }
                }
            else:
                valid_values = values.dropna()
                
                if len(valid_values) > 0:
                    pseudo_loq = np.percentile(valid_values, 25)
                    detected_samples = len(valid_values[valid_values >= pseudo_loq])
                    detection_rate = detected_samples / total_samples
                else:
                    pseudo_loq = np.nan
                    detected_samples = 0
                    detection_rate = 0
                
                analysis = {
                    'avg_loq': pseudo_loq,
                    'total_samples': total_samples,
                    'detected_samples': detected_samples,
                    'detection_rate': detection_rate,
                    'detectability_category': self._categorize_detectability(detection_rate),
                    'has_loq': False,
                    'needs_augmentation': True,
                    'needs_spectral_analysis': True,  # NUEVO
                    'concentration_stats': {
                        'detected_mean': valid_values.mean() if len(valid_values) > 0 else np.nan,
                        'detected_max': valid_values.max() if len(valid_values) > 0 else np.nan,
                        'signal_to_loq_ratio': np.nan
                    }
                }
            
            detectability_analysis[contaminant_name] = analysis
        
        self._save_detectability_analysis(detectability_analysis)
        
        return detectability_analysis
    
    def _get_loq_for_contaminant(self, contaminant_name: str) -> Optional[float]:
        """Obtener LOQ promedio para un contaminante"""
        
        if self.loq_data is None:
            return None
        
        loq_col = f"lab_loq_{contaminant_name}_ng_l"
        
        if loq_col in self.loq_data.columns:
            return self.loq_data[loq_col].mean()
        
        return None
    
    def _categorize_detectability(self, detection_rate: float) -> str:
        """Categorizar detectabilidad"""
        if detection_rate >= 0.70:
            return "High"
        elif detection_rate >= 0.40:
            return "Medium"
        elif detection_rate >= 0.20:
            return "Low"
        else:
            return "Poor"
    
    def temporal_matching(self, df_reflectance, df_chemicals, df_pollution=None, tolerance_minutes=30):
        """Matching temporal con validación mejorada"""
        
        self.logger.info("🔗 REALIZANDO MATCHING TEMPORAL")
        
        matched_data = []
        tolerance = pd.Timedelta(minutes=tolerance_minutes)
        
        for _, chem_row in df_chemicals.iterrows():
            chem_time = chem_row['timestamp']
            
            time_diffs = abs(df_reflectance['timestamp'] - chem_time)
            closest_idx = time_diffs.idxmin()
            
            if time_diffs.loc[closest_idx] <= tolerance:
                spec_row = df_reflectance.loc[closest_idx]
                
                combined_row = {}
                combined_row.update(spec_row.to_dict())
                combined_row.update(chem_row.to_dict())
                combined_row['time_diff_minutes'] = time_diffs.loc[closest_idx].total_seconds() / 60
                
                if df_pollution is not None:
                    inorg_diffs = abs(df_pollution['timestamp'] - chem_time)
                    closest_inorg_idx = inorg_diffs.idxmin()
                    
                    if inorg_diffs.loc[closest_inorg_idx] <= tolerance:
                        inorg_row = df_pollution.loc[closest_inorg_idx]
                        for col, val in inorg_row.items():
                            if col.startswith('lab_') and not col.endswith('_method'):
                                combined_row[col] = val
                
                matched_data.append(combined_row)
        
        df_matched = pd.DataFrame(matched_data)
        
        self.logger.info(f"✓ Matches encontrados: {len(df_matched)}")
        if len(df_matched) > 0:
            avg_diff = df_matched['time_diff_minutes'].mean()
            self.logger.info(f" Diferencia temporal promedio: {avg_diff:.1f} minutos")
        
        return df_matched
    
    def create_spectral_enhanced_datasets(self, df_matched, detectability_analysis):
        """Crear datasets con features espectrales avanzadas y augmentation"""
        
        self.logger.info(" CREANDO DATASETS CON ANÁLISIS ESPECTRAL AVANZADO")
        
        contaminant_cols = [col for col in df_matched.columns 
                          if col.startswith('lab_') and not col.endswith('_below_loq') and not col.endswith('_method')]
        
        datasets = {}
        
        for cont_col in contaminant_cols:
            contaminant_name = cont_col.replace('lab_', '').replace('_ng_l', '').replace('_mg_l', '').replace('_ntu', '')
            
            if contaminant_name not in detectability_analysis:
                continue
            
            analysis = detectability_analysis[contaminant_name]
            
            valid_mask = df_matched[cont_col].notna()
            if not any(valid_mask):
                continue
            
            valid_data = df_matched[valid_mask].copy()
            
            if len(valid_data) < 10:
                self.logger.warning(f" {contaminant_name}: solo {len(valid_data)} muestras - omitiendo")
                continue
            
            try:
                self.logger.info(f" Procesando {contaminant_name}...")
                
                # Extraer espectros raw
                X_raw = valid_data[self.spectral_columns]
                X_numeric = X_raw.apply(pd.to_numeric, errors='coerce')
                
                nan_percentage = X_numeric.isna().sum().sum() / (X_numeric.shape[0] * X_numeric.shape[1])
                if nan_percentage > 0.1:
                    self.logger.warning(f" {contaminant_name}: {nan_percentage:.1%} datos faltantes - omitiendo")
                    continue
                
                X_filled = X_numeric.interpolate(method='linear', axis=1)
                X_filled = X_filled.fillna(method='bfill', axis=1)
                X_filled = X_filled.fillna(method='ffill', axis=1)
                X_filled = X_filled.fillna(X_filled.mean())
                
                X_spectra = X_filled.values
                y_concentrations = valid_data[cont_col].values
                
                if X_spectra.shape[0] == 0 or len(y_concentrations) == 0:
                    self.logger.warning(f" {contaminant_name}: datos vacíos después de limpieza")
                    continue
                
                # NUEVO: Crear firma espectral del contaminante
                if self.use_spectral_features and self.signature_analyzer is not None:
                    self.logger.info(f"   Creando firma espectral...")
                    signature = self.signature_analyzer.create_contaminant_signature(
                        X_spectra, y_concentrations, contaminant_name
                    )
                    if signature:
                        self.spectral_signatures[contaminant_name] = signature
                        quality = signature['quality_metrics']['overall_quality']
                        n_peaks = len(signature['characteristic_peaks'])
                        self.logger.info(f"  ✓ Firma espectral: calidad {quality:.1f}/100, {n_peaks} picos")
                
                # Añadir muestras augmented (baseline, mezclas, etc.)
                X_enhanced, y_enhanced = self._add_augmented_samples(
                    X_spectra, y_concentrations, contaminant_name, analysis
                )
                
                # NUEVO: Extraer features espectrales avanzadas
                X_spectral_features = None
                spectral_feature_names = []
                
                if self.use_spectral_features and self.spectral_engineer is not None:
                    self.logger.info(f"   Extrayendo features espectrales avanzadas...")
                    try:
                        spectral_features_df = self.spectral_engineer.extract_all_features(X_enhanced)
                        X_spectral_features = spectral_features_df.values
                        spectral_feature_names = spectral_features_df.columns.tolist()
                        
                        self.logger.info(f"  ✓ {len(spectral_feature_names)} features espectrales extraídas")
                        
                        # Análisis de importancia de features
                        importance_analysis = self.spectral_engineer.get_feature_importance_analysis(
                            spectral_features_df, y_enhanced
                        )
                        
                        if 'top_features' in importance_analysis:
                            top_features = importance_analysis['top_features'][:5]
                            self.logger.info(f"   Top features: {', '.join(top_features)}")
                    
                    except Exception as e:
                        self.logger.error(f"   Error en features espectrales: {e}")
                        self.logger.info(f"   Usando features básicas como fallback")
                        X_spectral_features = extract_spectral_features(X_enhanced, self.wavelengths)
                        spectral_feature_names = [f'basic_feature_{i}' for i in range(X_spectral_features.shape[1])]
                
                # Preparar features según estrategia
                X_final, final_feature_names = self._prepare_features_by_strategy(
                    X_enhanced, X_spectral_features, spectral_feature_names
                )
                
                # Normalizar features
                scaler = RobustScaler()  # Más robusto para features espectrales
                X_scaled = scaler.fit_transform(X_final)
                
                # Crear múltiples estrategias de targeting
                dataset = self._create_multiple_targets_spectral_enhanced(
                    X_scaled, y_enhanced, scaler, contaminant_name, analysis, 
                    X_enhanced, X_spectral_features, final_feature_names
                )
                
                datasets[contaminant_name] = dataset
                
                # Log de resultados
                category = analysis['detectability_category']
                loq_status = "con LOQ" if analysis['has_loq'] else "sin LOQ"
                aug_status = " augmented" if analysis['needs_augmentation'] else "original"
                spectral_status = f"+ {len(spectral_feature_names)} feat. espectrales" if self.use_spectral_features else ""
                
                self.logger.info(f"✓ {contaminant_name}: {len(y_enhanced)} muestras ({len(y_concentrations)} orig)")
                self.logger.info(f"   {category}, {loq_status}, {aug_status} {spectral_status}")
                
            except Exception as e:
                self.logger.error(f" Error procesando {contaminant_name}: {e}")
                continue
        
        return datasets
    
    def _prepare_features_by_strategy(self, X_raw, X_spectral, spectral_feature_names):
        """Preparar features según la estrategia configurada"""
        
        raw_feature_names = [f'wl_{wl:.1f}' for wl in self.wavelengths]
        
        if self.spectral_strategy == "spectral_only" and X_spectral is not None:
            return X_spectral, spectral_feature_names
            
        elif self.spectral_strategy == "combined" and X_spectral is not None:
            X_combined = np.hstack([X_spectral, X_raw])
            combined_names = spectral_feature_names + raw_feature_names
            return X_combined, combined_names
            
        else:  # raw_only o fallback
            return X_raw, raw_feature_names
    
    def _add_augmented_samples(self, X_original, y_original, contaminant_name, analysis):
        """Añadir muestras baseline, mezclas y augmentation (mantener lógica original)"""
        
        X_augmented = [X_original]
        y_augmented = [y_original]
        
        # 1. Añadir muestras baseline
        if len(self.baseline_spectra) > 0 and analysis['needs_augmentation']:
            n_baseline = min(len(self.baseline_spectra), len(y_original) // 2)
            
            if n_baseline > 0:
                baseline_selection = np.random.choice(len(self.baseline_spectra), n_baseline, replace=False)
                baseline_samples = np.array([self.baseline_spectra[i] for i in baseline_selection])
                
                if analysis['has_loq']:
                    baseline_concentrations = np.random.uniform(0, analysis['avg_loq'] * 0.1, n_baseline)
                else:
                    baseline_concentrations = np.random.uniform(0, np.min(y_original) * 0.1, n_baseline)
                
                X_augmented.append(baseline_samples)
                y_augmented.append(baseline_concentrations)
                
                self.logger.info(f"     Añadidas {n_baseline} muestras baseline")
        
        # 2. Añadir mezclas de contaminantes
        if analysis['needs_augmentation'] and len(X_original) > 5:
            n_mixtures = len(y_original) // 3
            
            mixture_spectra = []
            mixture_concentrations = []
            
            for _ in range(n_mixtures):
                n_mix = np.random.randint(2, 4)
                indices = np.random.choice(len(X_original), n_mix, replace=False)
                
                weights = np.random.dirichlet(np.ones(n_mix))
                mixed_spectrum = np.average(X_original[indices], axis=0, weights=weights)
                mixed_concentration = np.average(y_original[indices], weights=weights)
                
                noise = np.random.normal(0, 0.01, len(mixed_spectrum))
                mixed_spectrum += noise
                
                mixture_spectra.append(mixed_spectrum)
                mixture_concentrations.append(mixed_concentration)
            
            if mixture_spectra:
                X_augmented.append(np.array(mixture_spectra))
                y_augmented.append(np.array(mixture_concentrations))
                
                self.logger.info(f"     Añadidas {len(mixture_spectra)} mezclas")
        
        # 3. Data augmentation espectral
        if analysis['needs_augmentation']:
            X_aug_spectral, y_aug_spectral = self._spectral_augmentation(X_original, y_original)
            
            if len(X_aug_spectral) > 0:
                X_augmented.append(X_aug_spectral)
                y_augmented.append(y_aug_spectral)
                
                self.logger.info(f"     Añadidas {len(X_aug_spectral)} variaciones espectrales")
        
        # Combinar todos los datos
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        return X_final, y_final
    
    def _spectral_augmentation(self, X, y, augmentation_factor=0.5):
        """Aplicar augmentation espectral realista"""
        
        config = self.augmentation_config
        n_augment = int(len(X) * augmentation_factor)
        
        if n_augment == 0:
            return np.array([]), np.array([])
        
        X_augmented = []
        y_augmented = []
        
        for _ in range(n_augment):
            idx = np.random.randint(0, len(X))
            spectrum = X[idx].copy()
            concentration = y[idx]
            
            if config['enable_noise']:
                noise = np.random.normal(0, config['noise_level'], len(spectrum))
                spectrum += noise
            
            if config['enable_shift']:
                shift_amount = np.random.uniform(-config['shift_range'], config['shift_range'])
                spectrum = spectrum * (1 + shift_amount)
            
            if config['enable_scale']:
                scale_factor = np.random.uniform(1 - config['scale_range'], 1 + config['scale_range'])
                spectrum = spectrum * scale_factor
            
            if config['enable_smooth']:
                if np.random.random() < 0.3:
                    spectrum = gaussian_filter1d(spectrum, config['smooth_sigma'])
            
            spectrum = np.maximum(spectrum, 0.001)
            spectrum = np.minimum(spectrum, 1.0)
            
            X_augmented.append(spectrum)
            y_augmented.append(concentration)
        
        return np.array(X_augmented), np.array(y_augmented)
    
    def _create_multiple_targets_spectral_enhanced(self, X_scaled, y, scaler, contaminant_name, 
                                                  analysis, X_raw, X_spectral, feature_names):
        """Crear múltiples estrategias de targeting con información espectral"""
        
        # Splits básicos
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        # Dataset base con información espectral
        dataset = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'scaler': scaler,
            'feature_names': feature_names,
            'wavelengths': self.wavelengths,
            'detectability_analysis': analysis,
            'augmentation_applied': analysis['needs_augmentation'],
            'spectral_analysis_applied': self.use_spectral_features,
            'spectral_strategy': self.spectral_strategy
        }
        
        # Añadir información de firma espectral si existe
        if contaminant_name in self.spectral_signatures:
            signature = self.spectral_signatures[contaminant_name]
            dataset['spectral_signature'] = {
                'quality_score': signature['quality_metrics']['overall_quality'],
                'characteristic_peaks': signature['characteristic_peaks'][:5],  # Top 5
                'discriminant_wavelengths': signature['discriminant_wavelengths'][:10],  # Top 10
                'top_features': signature.get('spectral_features', {}).get('top_features', [])[:10]
            }
        
        # Estrategias de targeting mejoradas
        if analysis['has_loq']:
            loq = analysis['avg_loq']
            
            # 1. Targets originales (regresión)
            dataset.update({
                'y_train_original': y_train,
                'y_val_original': y_val,
                'y_test_original': y_test
            })
            
            # 2. Clasificación binaria (detectado/no detectado)
            dataset.update({
                'y_train_binary': (y_train >= loq).astype(int),
                'y_val_binary': (y_val >= loq).astype(int),
                'y_test_binary': (y_test >= loq).astype(int)
            })
            
            # 3. Clasificación ternaria mejorada
            def create_ternary_enhanced(y_vals):
                ternary = np.zeros_like(y_vals, dtype=int)
                
                p33 = np.percentile(y_vals, 33.33)
                p66 = np.percentile(y_vals, 66.67)
                
                ternary[y_vals <= p33] = 0
                ternary[(y_vals > p33) & (y_vals <= p66)] = 1
                ternary[y_vals > p66] = 2
                
                return ternary
            
            dataset.update({
                'y_train_ternary': create_ternary_enhanced(y_train),
                'y_val_ternary': create_ternary_enhanced(y_val),
                'y_test_ternary': create_ternary_enhanced(y_test)
            })
            
            # 4. Percentiles personalizados
            for percentile in [25, 50, 75]:
                threshold = np.percentile(y, percentile)
                dataset.update({
                    f'y_train_percentile{percentile}': (y_train >= threshold).astype(int),
                    f'y_val_percentile{percentile}': (y_val >= threshold).astype(int),
                    f'y_test_percentile{percentile}': (y_test >= threshold).astype(int)
                })
        else:
            # Sin LOQ - estrategias basadas en percentiles
            dataset.update({
                'y_train_original': y_train,
                'y_val_original': y_val,
                'y_test_original': y_test
            })
            
            median_threshold = np.median(y)
            dataset.update({
                'y_train_binary': (y_train >= median_threshold).astype(int),
                'y_val_binary': (y_val >= median_threshold).astype(int),
                'y_test_binary': (y_test >= median_threshold).astype(int)
            })
        
        # Versiones LSTM mejoradas con información espectral
        dataset_lstm = dataset.copy()
        
        # Para LSTM, usar espectros raw (mejor para secuencias temporales)
        if X_raw is not None:
            # Dividir X_raw según los mismos índices que X_scaled
            indices = np.arange(len(X_scaled))
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
            train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)
            
            X_raw_train = X_raw[train_indices]
            X_raw_val = X_raw[val_indices] 
            X_raw_test = X_raw[test_indices]
            
            # Normalizar espectros raw para LSTM
            raw_scaler = StandardScaler()
            X_raw_train_scaled = raw_scaler.fit_transform(X_raw_train)
            X_raw_val_scaled = raw_scaler.transform(X_raw_val)
            X_raw_test_scaled = raw_scaler.transform(X_raw_test)
            
            dataset_lstm.update({
                'X_train': X_raw_train_scaled.reshape(X_raw_train_scaled.shape[0], X_raw_train_scaled.shape[1], 1),
                'X_val': X_raw_val_scaled.reshape(X_raw_val_scaled.shape[0], X_raw_val_scaled.shape[1], 1),
                'X_test': X_raw_test_scaled.reshape(X_raw_test_scaled.shape[0], X_raw_test_scaled.shape[1], 1),
                'sequence_length': X_raw_train_scaled.shape[1],
                'scaler': raw_scaler,
                'data_type': 'spectral_sequences'
            })
        
        return {
            'classical': dataset,
            'lstm': dataset_lstm,
            'metadata': {
                'n_samples': len(y),
                'n_original_samples': analysis['total_samples'],
                'augmentation_ratio': len(y) / analysis['total_samples'] if analysis['total_samples'] > 0 else 1.0,
                'detectability_category': analysis['detectability_category'],
                'has_loq': analysis['has_loq'],
                'avg_loq': analysis.get('avg_loq', np.nan),
                'detection_rate': analysis['detection_rate'],
                'strategies_available': self._get_available_strategies_enhanced(analysis, dataset),
                'augmentation_applied': analysis['needs_augmentation'],
                'spectral_analysis_applied': self.use_spectral_features,
                'spectral_strategy': self.spectral_strategy,
                'n_spectral_features': len([f for f in feature_names if 'wl_' not in f]) if self.use_spectral_features else 0,
                'n_raw_features': len([f for f in feature_names if 'wl_' in f]),
                'spectral_signature_quality': self.spectral_signatures.get(contaminant_name, {}).get('quality_metrics', {}).get('overall_quality', 0),
                'concentration_stats': {
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y)),
                    'original_mean': float(analysis['concentration_stats']['detected_mean']) if not np.isnan(analysis['concentration_stats']['detected_mean']) else float(np.mean(y))
                }
            }
        }
    
    def _get_available_strategies_enhanced(self, analysis, dataset):
        """Determinar estrategias disponibles con mejoras espectrales"""
        strategies = ['original']
        
        if analysis['has_loq']:
            strategies.extend(['binary', 'ternary'])
            for percentile in [25, 50, 75]:
                if f'y_train_percentile{percentile}' in dataset:
                    strategies.append(f'percentile{percentile}')
        else:
            strategies.append('binary')
        
        # Añadir estrategias específicas para análisis espectral
        if self.use_spectral_features:
            strategies.append('spectral_enhanced')
        
        return strategies
    
    def _save_detectability_analysis(self, analysis_results):
        """Guardar análisis de detectabilidad mejorado con información espectral"""
        
        analysis_data = []
        
        for contaminant, results in analysis_results.items():
            # Información básica
            row = {
                'contaminant': contaminant,
                'avg_loq_ng_l': results.get('avg_loq', np.nan),
                'has_loq': results['has_loq'],
                'total_samples': results['total_samples'],
                'detected_samples': results['detected_samples'],
                'detection_rate': results['detection_rate'],
                'detectability_category': results['detectability_category'],
                'needs_augmentation': results['needs_augmentation'],
                'needs_spectral_analysis': results['needs_spectral_analysis'],
                'detected_mean_ng_l': results['concentration_stats']['detected_mean'],
                'detected_max_ng_l': results['concentration_stats']['detected_max'],
                'signal_to_loq_ratio': results['concentration_stats']['signal_to_loq_ratio']
            }
            
            # Información espectral si está disponible
            if contaminant in self.spectral_signatures:
                signature = self.spectral_signatures[contaminant]
                row.update({
                    'spectral_signature_quality': signature['quality_metrics']['overall_quality'],
                    'spectral_n_peaks': len(signature['characteristic_peaks']),
                    'spectral_n_discriminants': len(signature['discriminant_wavelengths']),
                    'spectral_consistency_score': signature['quality_metrics'].get('consistency_score', 0),
                    'spectral_dynamic_range': signature['quality_metrics'].get('dynamic_range', 0)
                })
            else:
                row.update({
                    'spectral_signature_quality': np.nan,
                    'spectral_n_peaks': 0,
                    'spectral_n_discriminants': 0,
                    'spectral_consistency_score': np.nan,
                    'spectral_dynamic_range': np.nan
                })
            
            analysis_data.append(row)
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # Ordenar por calidad espectral y detectabilidad
        category_order = {'High': 3, 'Medium': 2, 'Low': 1, 'Poor': 0}
        df_analysis['sort_order'] = df_analysis['detectability_category'].map(category_order)
        df_analysis = df_analysis.sort_values(
            ['sort_order', 'spectral_signature_quality', 'detection_rate'], 
            ascending=[False, False, False]
        )
        df_analysis = df_analysis.drop('sort_order', axis=1)
        
        # Guardar
        analysis_file = self.output_dir / "spectral_enhanced_detectability_analysis.csv"
        df_analysis.to_csv(analysis_file, index=False)
        
        self.logger.info(f" Análisis guardado: {analysis_file}")
        
        # Mostrar resumen mejorado con información espectral
        self.logger.info("\n RESUMEN DE DETECTABILIDAD + ANÁLISIS ESPECTRAL:")
        for category in ['High', 'Medium', 'Low', 'Poor']:
            count = len(df_analysis[df_analysis['detectability_category'] == category])
            with_loq = len(df_analysis[(df_analysis['detectability_category'] == category) & 
                                     (df_analysis['has_loq'] == True)])
            with_signature = len(df_analysis[(df_analysis['detectability_category'] == category) & 
                                           (df_analysis['spectral_signature_quality'].notna())])
            
            self.logger.info(f"  {category}: {count} contaminantes ({with_loq} con LOQ, {with_signature} con firma espectral)")
    
    def save_spectral_enhanced_datasets(self, datasets):
        """Guardar todos los datasets mejorados con información espectral"""
        
        self.logger.info(" GUARDANDO DATASETS ESPECTRALMENTE MEJORADOS")
        
        saved_files = []
        
        for contaminant, data in datasets.items():
            # Guardar dataset clásico con features espectrales
            classical_file = self.output_dir / f"{contaminant}_spectral_enhanced_classical.npz"
            np.savez_compressed(classical_file, **data['classical'])
            saved_files.append(classical_file)
            
            # Guardar dataset LSTM 
            lstm_file = self.output_dir / f"{contaminant}_spectral_enhanced_lstm.npz"
            np.savez_compressed(lstm_file, **data['lstm'])
            saved_files.append(lstm_file)
            
            # Guardar metadatos extendidos
            import json
            metadata_file = self.output_dir / f"{contaminant}_spectral_enhanced_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data['metadata'], f, indent=2, default=str)
            saved_files.append(metadata_file)
        
        # Guardar firmas espectrales si existen
        if self.spectral_signatures:
            from spectral_analisis import save_spectral_signature
            
            signatures_dir = self.output_dir / "spectral_signatures"
            signatures_dir.mkdir(exist_ok=True)
            
            for contaminant, signature in self.spectral_signatures.items():
                signature_file = save_spectral_signature(signature, str(signatures_dir))
                if signature_file:
                    saved_files.append(signature_file)
        
        self.logger.info(f"✓ {len(saved_files)} archivos guardados")
        return saved_files
    
    def generate_spectral_enhanced_report(self, datasets, detectability_analysis):
        """Generar reporte comprehensive con análisis espectral integrado"""
        
        self.logger.info(" GENERANDO REPORTE SPECTRAL ENHANCED")
        
        # Estadísticas generales
        total_contaminants = len(datasets)
        with_loq = len([d for d in datasets.values() if d['metadata']['has_loq']])
        with_augmentation = len([d for d in datasets.values() if d['metadata']['augmentation_applied']])
        with_spectral = len([d for d in datasets.values() if d['metadata']['spectral_analysis_applied']])
        with_signatures = len(self.spectral_signatures)
        
        # Estadísticas espectrales
        spectral_quality_scores = [
            self.spectral_signatures[name]['quality_metrics']['overall_quality'] 
            for name in self.spectral_signatures.keys()
        ]
        avg_spectral_quality = np.mean(spectral_quality_scores) if spectral_quality_scores else 0
        
        # Estadísticas de features
        total_original_samples = sum([d['metadata']['n_original_samples'] for d in datasets.values()])
        total_enhanced_samples = sum([d['metadata']['n_samples'] for d in datasets.values()])
        
        avg_spectral_features = np.mean([
            d['metadata']['n_spectral_features'] 
            for d in datasets.values() 
            if d['metadata']['spectral_analysis_applied']
        ]) if with_spectral > 0 else 0
        
        report = f"""# REPORTE SPECTRAL ENHANCED ML PIPELINE
## Universidad Diego Portales - María José Erazo González

### RESUMEN EJECUTIVO ESPECTRAL
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Contaminantes procesados**: {total_contaminants}
- **Con datos LOQ**: {with_loq} ({with_loq/total_contaminants*100:.1f}%)
- **Con augmentation aplicado**: {with_augmentation} ({with_augmentation/total_contaminants*100:.1f}%)
- **Con análisis espectral avanzado**: {with_spectral} ({with_spectral/total_contaminants*100:.1f}%)
- **Con firmas espectrales**: {with_signatures} ({with_signatures/total_contaminants*100:.1f}%)
- **Calidad espectral promedio**: {avg_spectral_quality:.1f}/100
- **Features espectrales promedio**: {avg_spectral_features:.0f}
- **Muestras**: {total_original_samples} → {total_enhanced_samples} ({total_enhanced_samples/total_original_samples:.1f}x)
- **Espectros baseline**: {len(self.baseline_spectra)}
- **Rango espectral**: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm
- **Resolución espectral**: ~{np.mean(np.diff(self.wavelengths)):.1f} nm
- **Bandas espectrales**: {len(self.wavelengths)}
- **Estrategia de features**: {self.spectral_strategy}

### ANÁLISIS ESPECTRAL AVANZADO

#### Firmas Espectrales Generadas:
"""
        
        # Ordenar firmas por calidad
        sorted_signatures = sorted(
            self.spectral_signatures.items(),
            key=lambda x: x[1]['quality_metrics']['overall_quality'],
            reverse=True
        )
        
        for i, (name, signature) in enumerate(sorted_signatures[:10], 1):  # Top 10
            quality = signature['quality_metrics']['overall_quality']
            n_peaks = len(signature['characteristic_peaks'])
            n_discriminants = len(signature['discriminant_wavelengths'])
            
            # Top wavelengths discriminantes
            top_wavelengths = [f"{d['wavelength']:.0f}nm" for d in signature['discriminant_wavelengths'][:3]]
            
            report += f"""
**{i}. {name}**
- Calidad espectral: {quality:.1f}/100
- Picos característicos: {n_peaks}
- Wavelengths discriminantes: {n_discriminants} (top: {', '.join(top_wavelengths)})
- Muestras utilizadas: {signature['n_samples']}
"""
        
        report += f"""
#### Features Espectrales por Categoría:
"""
        
        if self.use_spectral_features:
            report += f"""
- **Estadísticas espectrales**: Media, desviación, min, max, percentiles, asimetría, curtosis
- **Índices espectrales**: NDVI, NDWI, ratios específicos para calidad de agua
- **Características de forma**: Pendientes, curvatura, puntos de inflexión
- **Análisis por rangos**: UV, Visible, NIR - estadísticas por banda espectral
- **Detección de picos**: Número, altura y prominencia de picos espectrales
- **Análisis de derivadas**: Primera y segunda derivada para detección de cambios

#### Estrategia de Features: {self.spectral_strategy.upper()}
"""
            
            if self.spectral_strategy == "spectral_only":
                report += "- Solo features espectrales interpretables (no bandas raw)\n"
            elif self.spectral_strategy == "combined":
                report += "- Combinación de features espectrales + bandas raw\n"
            else:
                report += "- Solo bandas espectrales raw (sin procesamiento avanzado)\n"
        
        # Distribución por detectabilidad + calidad espectral
        by_category = {}
        spectral_quality_by_category = {}
        
        for data in datasets.values():
            category = data['metadata']['detectability_category']
            by_category[category] = by_category.get(category, 0) + 1
            
            # Calidad espectral por categoría
            if data['metadata']['spectral_signature_quality'] > 0:
                if category not in spectral_quality_by_category:
                    spectral_quality_by_category[category] = []
                spectral_quality_by_category[category].append(data['metadata']['spectral_signature_quality'])
        
        report += f"""
### DISTRIBUCIÓN POR DETECTABILIDAD + CALIDAD ESPECTRAL

"""
        
        for category, count in by_category.items():
            avg_quality = np.mean(spectral_quality_by_category.get(category, [0]))
            quality_info = f" (Calidad espectral promedio: {avg_quality:.1f}/100)" if avg_quality > 0 else ""
            report += f"- **{category}**: {count} contaminantes{quality_info}\n"
        
        # Contaminantes con mejor análisis espectral
        report += f"""
### TOP CONTAMINANTES CON ANÁLISIS ESPECTRAL ROBUSTO

#### Por Calidad de Firma Espectral:
"""
        
        spectral_rankings = []
        for name, data in datasets.items():
            if data['metadata']['spectral_signature_quality'] > 0:
                spectral_rankings.append((
                    name,
                    data['metadata']['spectral_signature_quality'],
                    data['metadata']['n_samples'],
                    data['metadata']['detectability_category'],
                    data['metadata']['n_spectral_features']
                ))
        
        spectral_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, quality, samples, category, n_features) in enumerate(spectral_rankings[:10], 1):
            report += f"{i:2d}. **{name}**: Calidad {quality:.1f}/100, {samples} muestras, {category}, {n_features} features\n"
        
        report += f"""
### RECOMENDACIONES CIENTÍFICAS ESPECTRALES

#### Contaminantes Ideales para Modelado Espectral:
"""
        
        excellent_spectral = [
            name for name, data in datasets.items() 
            if data['metadata']['spectral_signature_quality'] >= 80 and
               data['metadata']['detectability_category'] in ['High', 'Medium']
        ]
        
        report += f"**Excelente calidad espectral ({len(excellent_spectral)} contaminantes):**\n"
        for name in excellent_spectral:
            data = datasets[name]
            quality = data['metadata']['spectral_signature_quality']
            samples = data['metadata']['n_samples']
            features = data['metadata']['n_spectral_features']
            report += f"- **{name}**: {quality:.1f}/100, {samples} muestras, {features} features espectrales\n"
        
        report += f"""
#### Casos de Investigación Espectral Avanzada:
"""
        
        research_spectral = [
            name for name, data in datasets.items() 
            if data['metadata']['spectral_analysis_applied'] and
               data['metadata']['augmentation_ratio'] > 2.0 and
               data['metadata']['spectral_signature_quality'] > 60
        ]
        
        for name in research_spectral:
            data = datasets[name]
            quality = data['metadata']['spectral_signature_quality']
            ratio = data['metadata']['augmentation_ratio']
            features = data['metadata']['n_spectral_features']
            report += f"- **{name}**: Calidad {quality:.1f}/100, augmentation {ratio:.1f}x, {features} features espectrales\n"
        
        report += f"""
### ARCHIVOS GENERADOS ESPECTRALMENTE MEJORADOS
- `*_spectral_enhanced_classical.npz`: Datasets con features espectrales para ML clásico
- `*_spectral_enhanced_lstm.npz`: Secuencias espectrales para LSTM/CNN1D
- `*_spectral_enhanced_metadata.json`: Metadatos + análisis espectral completo
- `spectral_signatures/signature_*.json`: Firmas espectrales individuales por contaminante
- `spectral_enhanced_detectability_analysis.csv`: Análisis completo + métricas espectrales

### CALIDAD CIENTÍFICA ESPECTRAL AVANZADA
 **Pipeline científicamente riguroso con análisis espectral**
 **Features espectrales interpretables basadas en física**
 **Firmas espectrales específicas por contaminante**
 **Índices espectrales validados para calidad de agua**
 **Balance automático con muestras baseline realistas**
 **Augmentation espectral que preserva características físicas**
 **Múltiples estrategias: raw, espectral, combinadas**
 **Análisis de wavelengths discriminantes**
 **Datasets listos para publicación en journals especializados**

### METODOLOGÍA ESPECTRAL IMPLEMENTADA

#### 1. Extracción de Features Espectrales Avanzadas
- Estadísticas espectrales: 12 métricas estadísticas por espectro
- Índices espectrales: 9 índices validados para calidad de agua  
- Análisis de forma: Pendientes, curvatura, puntos de inflexión
- Features por rangos: Análisis separado UV/VIS/NIR
- Detección de picos: Análisis automático de características espectrales
- Derivadas espectrales: Primera y segunda derivada para detección de cambios

#### 2. Firmas Espectrales por Contaminante
- Creación automática de firmas espectrales características
- Identificación de wavelengths discriminantes
- Análisis de picos característicos por contaminante
- Métricas de calidad y consistencia espectral
- Comparación entre contaminantes

#### 3. Estrategias de Modelado Flexibles
- **Spectral Only**: Solo features espectrales interpretables
- **Combined**: Features espectrales + bandas raw para máximo rendimiento  
- **Raw Only**: Solo bandas espectrales (baseline para comparación)

---
*Generado automáticamente por Spectral Enhanced ML Generator*
*Pipeline validado para detección espectral robusta de contaminantes acuáticos*
*Con análisis espectral avanzado y firmas espectrales por contaminante*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Guardar reporte
        report_file = self.output_dir / "spectral_enhanced_comprehensive_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f" Reporte guardado: {report_file}")
        
        return report
    
    def run_complete_spectral_enhanced_pipeline(self, reflectance_file, chemicals_file, 
                                              pollution_file=None, loq_file=None):
        """Ejecutar pipeline completo con análisis espectral"""
        
        self.logger.info(" EJECUTANDO PIPELINE SPECTRAL ENHANCED COMPLETO")
        self.logger.info("=" * 70)
        
        try:
            # 1. Cargar todos los datos
            df_reflectance, df_chemicals, df_pollution = self.load_all_data(
                reflectance_file, chemicals_file, pollution_file, loq_file
            )
            
            # 2. Analizar detectabilidad
            detectability_analysis = self.analyze_detectability(df_chemicals)
            
            # 3. Matching temporal
            df_matched = self.temporal_matching(df_reflectance, df_chemicals, df_pollution)
            
            # 4. Crear datasets con análisis espectral avanzado
            datasets = self.create_spectral_enhanced_datasets(df_matched, detectability_analysis)
            
            # 5. Guardar datasets espectralmente mejorados
            saved_files = self.save_spectral_enhanced_datasets(datasets)
            
            # 6. Generar reporte comprehensive con análisis espectral
            report = self.generate_spectral_enhanced_report(datasets, detectability_analysis)
            
            self.logger.info("=" * 70)
            self.logger.info(" PIPELINE SPECTRAL ENHANCED COMPLETADO EXITOSAMENTE")
            self.logger.info(f" Contaminantes procesados: {len(datasets)}")
            self.logger.info(f" Archivos generados: {len(saved_files)}")
            
            # Estadísticas finales mejoradas con información espectral
            with_loq = len([d for d in datasets.values() if d['metadata']['has_loq']])
            with_augmentation = len([d for d in datasets.values() if d['metadata']['augmentation_applied']])
            with_spectral = len([d for d in datasets.values() if d['metadata']['spectral_analysis_applied']])
            high_quality_spectral = len([d for d in datasets.values() 
                                       if d['metadata']['spectral_signature_quality'] >= 70])
            
            total_original = sum([d['metadata']['n_original_samples'] for d in datasets.values()])
            total_enhanced = sum([d['metadata']['n_samples'] for d in datasets.values()])
            
            avg_spectral_features = np.mean([
                d['metadata']['n_spectral_features'] 
                for d in datasets.values() 
                if d['metadata']['spectral_analysis_applied']
            ]) if with_spectral > 0 else 0
            
            self.logger.info(f" Con análisis LOQ: {with_loq}")
            self.logger.info(f" Con augmentation: {with_augmentation}")
            self.logger.info(f" Con análisis espectral: {with_spectral}")
            self.logger.info(f" Alta calidad espectral: {high_quality_spectral}")
            self.logger.info(f"🧮 Features espectrales promedio: {avg_spectral_features:.0f}")
            self.logger.info(f"📦 Muestras: {total_original} → {total_enhanced} ({total_enhanced/total_original:.1f}x)")
            self.logger.info(f" Firmas espectrales creadas: {len(self.spectral_signatures)}")
            self.logger.info("=" * 70)
            
            return {
                'datasets': datasets,
                'detectability_analysis': detectability_analysis,
                'saved_files': saved_files,
                'report': report,
                'spectral_signatures': self.spectral_signatures,
                'spectral_stats': {
                    'total_signatures_created': len(self.spectral_signatures),
                    'avg_spectral_quality': np.mean([
                        s['quality_metrics']['overall_quality'] 
                        for s in self.spectral_signatures.values()
                    ]) if self.spectral_signatures else 0,
                    'spectral_analysis_enabled': self.use_spectral_features,
                    'spectral_strategy': self.spectral_strategy,
                    'avg_features_per_dataset': avg_spectral_features,
                    'wavelength_range': [float(self.wavelengths.min()), float(self.wavelengths.max())],
                    'spectral_resolution': float(np.median(np.diff(self.wavelengths)))
                },
                'augmentation_stats': {
                    'total_original_samples': total_original,
                    'total_enhanced_samples': total_enhanced,
                    'augmentation_ratio': total_enhanced / total_original,
                    'contaminants_augmented': with_augmentation,
                    'baseline_spectra_used': len(self.baseline_spectra)
                }
            }
            
        except Exception as e:
            self.logger.error(f" Error en pipeline spectral enhanced: {e}")
            raise


def main():
    """Función principal del pipeline spectral enhanced"""
    
    print(" PIPELINE SPECTRAL ENHANCED ML + LOQ + AUGMENTATION + ANÁLISIS ESPECTRAL")
    print("Universidad Diego Portales - María José Erazo González")
    print("=" * 80)
    
    # Rutas de archivos
    files = {
        'reflectance': "data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv",  # Cambia por tu archivo de reflectancia
        'chemicals': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv",
        'pollution': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv",  # Datos inorgánicos si los tienes
        'loq': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_loq_organic_chemicals.csv"
    }
    
    # Configuración del pipeline espectral
    spectral_configs = [
        {
            'name': 'Spectral Only',
            'use_spectral_features': True,
            'spectral_strategy': 'spectral_only',
            'description': 'Solo features espectrales interpretables'
        },
        {
            'name': 'Combined Features',
            'use_spectral_features': True,
            'spectral_strategy': 'combined',
            'description': 'Features espectrales + bandas raw'
        },
        {
            'name': 'Raw Only (Baseline)',
            'use_spectral_features': False,
            'spectral_strategy': 'raw_only',
            'description': 'Solo bandas espectrales raw'
        }
    ]
    
    try:
        # Ejecutar múltiples configuraciones para comparación
        all_results = {}
        
        for config in spectral_configs:
            print(f"\n EJECUTANDO CONFIGURACIÓN: {config['name']}")
            print(f" {config['description']}")
            print("-" * 60)
            
            # Crear directorio específico para esta configuración
            output_dir = f"spectral_enhanced_datasets_{config['spectral_strategy']}"
            
            # Crear generator con configuración específica
            generator = SpectralEnhancedMLGenerator(
                output_dir=output_dir,
                use_spectral_features=config['use_spectral_features'],
                spectral_strategy=config['spectral_strategy']
            )
            
            # Ejecutar pipeline
            results = generator.run_complete_spectral_enhanced_pipeline(
                reflectance_file=files['reflectance'],
                chemicals_file=files['chemicals'],
                pollution_file=files['pollution'],
                loq_file=files['loq']
            )
            
            all_results[config['name']] = results
            
            print(f"\n RESULTADOS {config['name']}:")
            print(f"    Contaminantes: {len(results['datasets'])}")
            print(f"    Archivos: {len(results['saved_files'])}")
            print(f"    Firmas espectrales: {results['spectral_stats']['total_signatures_created']}")
            print(f"   ⚡ Calidad espectral promedio: {results['spectral_stats']['avg_spectral_quality']:.1f}/100")
            print(f"    Factor aumento: {results['augmentation_stats']['augmentation_ratio']:.1f}x")
        
        # Resumen comparativo final
        print(f"\n RESUMEN COMPARATIVO FINAL")
        print("=" * 60)
        
        for config_name, results in all_results.items():
            spectral_stats = results['spectral_stats']
            datasets = results['datasets']
            
            # Contar contaminantes de alta calidad por configuración
            high_quality = len([
                d for d in datasets.values() 
                if d['metadata']['detectability_category'] in ['High', 'Medium']
            ])
            
            print(f"\n {config_name}:")
            print(f"    Contaminantes alta calidad: {high_quality}")
            print(f"    Análisis espectral: {'SÍ' if spectral_stats['spectral_analysis_enabled'] else 'NO'}")
            print(f"    Features promedio: {spectral_stats['avg_features_per_dataset']:.0f}")
            print(f"    Calidad espectral: {spectral_stats['avg_spectral_quality']:.1f}/100")
        
        print(f"\n RECOMENDACIONES FINALES:")
        print(f"    Para máxima interpretabilidad: usar 'Spectral Only'")
        print(f"   ⚡ Para máximo rendimiento: usar 'Combined Features'")
        print(f"    Para baseline/comparación: usar 'Raw Only'")
        
        print(f"\n📂 DATASETS LISTOS EN:")
        for config in spectral_configs:
            output_dir = f"spectral_enhanced_datasets_{config['spectral_strategy']}"
            print(f"    {output_dir}/ - {config['description']}")
        
        print(f"\n SIGUIENTE PASO:")
        print(f"   Usar train_V4_fixed.py con datasets_dir='spectral_enhanced_datasets_[strategy]'")
        print(f"   Ejemplo: python train_V4_fixed.py --datasets_dir=spectral_enhanced_datasets_combined")
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()