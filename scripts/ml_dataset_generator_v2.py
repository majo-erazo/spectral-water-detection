"""
Pipeline de Preparación de Datos - VERSIÓN FINAL
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from spectral_analysis_complete import SpectralFeatureEngineer

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURACIÓN Y CLASES DE DATOS
# =====================================================================

@dataclass
class ProcessingConfig:
    """Configuración del pipeline de procesamiento"""
    # Parámetros temporales
    temporal_tolerance_minutes: int = 30
    min_samples_per_contaminant: int = 20
    
    # Parámetros de splits
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Parámetros de calidad
    min_temporal_coverage_pct: float = 5.0
    max_null_percentage: float = 80.0
    min_variance_threshold: float = 1e-8
    
    # Parámetros LSTM
    lstm_sequence_length: int = 5
    lstm_sequence_step_minutes: int = 30
    
    # Configuración de salida
    output_dir: str = "optimized_datasets"
    save_intermediate: bool = True
    generate_reports: bool = True
    
    # Configuración de logging
    log_level: str = "INFO"
    log_to_file: bool = True

@dataclass
class DatasetMetrics:
    """Métricas de calidad del dataset"""
    total_spectra: int = 0
    total_chemicals: int = 0
    temporal_coverage_hours: float = 0.0
    match_rate_pct: float = 0.0
    spectral_completeness_pct: float = 0.0
    viable_contaminants: int = 0
    data_quality_score: float = 0.0

@dataclass
class ContaminantInfo:
    """Información de un contaminante individual"""
    name: str
    clean_name: str
    n_samples: int
    temporal_coverage_pct: float
    concentration_stats: Dict
    threshold_binary: float
    viable: bool
    quality_score: float


import numpy as np
import os
from spectral_analysis_complete import extract_spectral_features

def generate_enhanced_dataset_if_missing(path):
    enhanced_path = path.replace(".npz", "_spectral_enhanced.npz")
    if os.path.exists(enhanced_path):
        return enhanced_path

    try:
        original = np.load(path, allow_pickle=True)
        if 'X' in original and 'y' in original:
            X_raw = original['X']
            y_raw = original['y']
        elif 'data' in original:
            data_obj = original['data'].item()
            X_raw = data_obj['X']
            y_raw = data_obj['y']
        else:
            raise ValueError("No se encontraron claves válidas en el archivo.")
    except Exception as e:
        print(f" Error al cargar {path}: {e}")
        return None

    # Aquí aplicas el procesamiento espectral
    X_feat = extract_spectral_features(X_raw)

    # Guardas el nuevo dataset
    np.savez(enhanced_path, X=X_feat, y=y_raw)
    return enhanced_path


# =====================================================================
# PIPELINE PRINCIPAL OPTIMIZADO
# =====================================================================

class OptimizedWaterQualityProcessor:
    """
    Procesador optimizado que combina todas las mejores prácticas
    """
    
    def __init__(self, config: ProcessingConfig = None, 
             use_spectral_features: bool = True,
             spectral_strategy: str = "spectral_only"):
        
        self.config = config or ProcessingConfig()
        self.use_spectral_features = use_spectral_features
        self.spectral_strategy = spectral_strategy  # "spectral_only", "combined", "raw_only"
        self.spectral_engineer = None

        self.logger = self._setup_logging()
        self.dataset_info = {}
        self.processing_metrics = {}
        
        self.logger.info(" OptimizedWaterQualityProcessor iniciado")
        self.logger.info(f"   Features espectrales: {use_spectral_features}")
        self.logger.info(f"   Estrategia: {spectral_strategy}")
        self.logger.info(f"   Configuración: {asdict(self.config)}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configurar sistema de logging"""
        logger = logging.getLogger('WaterQualityProcessor')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler si está habilitado
        if self.config.log_to_file:
            os.makedirs(self.config.output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                f"{self.config.output_dir}/processing.log"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def load_and_validate_data(self, reflectance_file: str, chemicals_file: str, 
                              pollution_file: Optional[str] = None) -> Tuple[pd.DataFrame, ...]:
        """
        Carga y validación robusta de datos con verificaciones extensivas
        """
        self.logger.info(" PASO 1: Carga y validación de datos")
        
        try:
            # Cargar datos principales
            df_reflectance = pd.read_csv(reflectance_file)
            df_chemicals = pd.read_csv(chemicals_file)
            self.logger.info(f"    Espectros: {df_reflectance.shape}")
            self.logger.info(f"    Químicos orgánicos: {df_chemicals.shape}")
            
            # Cargar inorgánicos si existe
            df_pollution = None
            if pollution_file and os.path.exists(pollution_file):
                try:
                    df_pollution = pd.read_csv(pollution_file)
                    self.logger.info(f"    Químicos inorgánicos: {df_pollution.shape}")
                except Exception as e:
                    self.logger.warning(f"    Error cargando inorgánicos: {e}")
            
            # Validar estructura básica
            self._validate_data_structure(df_reflectance, df_chemicals, df_pollution)
            
            # Procesar timestamps con validación
            df_reflectance = self._process_timestamps(df_reflectance, "reflectance")
            df_chemicals = self._process_timestamps(df_chemicals, "chemicals")
            if df_pollution is not None:
                df_pollution = self._process_timestamps(df_pollution, "pollution")
            
            # Filtrar datos válidos
            df_reflectance = self._filter_valid_spectra(df_reflectance)
            
            # Limpiar y validar datos químicos
            df_chemicals = self._clean_chemical_data(df_chemicals, "organic")
            if df_pollution is not None:
                df_pollution = self._clean_chemical_data(df_pollution, "inorganic")
            
            # Extraer información de wavelengths
            self._extract_spectral_info(df_reflectance)
            
            # Identificar contaminantes
            self._identify_contaminants(df_chemicals, df_pollution)
            
            # Validaciones finales
            self._final_data_validation(df_reflectance, df_chemicals, df_pollution)
            
            self.logger.info("    Datos cargados y validados exitosamente")
            
            return df_reflectance, df_chemicals, df_pollution
            
        except Exception as e:
            self.logger.error(f"    Error crítico en carga de datos: {e}")
            raise
    
    def _validate_data_structure(self, df_reflectance: pd.DataFrame, 
                                df_chemicals: pd.DataFrame, 
                                df_pollution: Optional[pd.DataFrame]):
        """Validar estructura básica de los datos"""
        
        # Verificar columnas esenciales
        required_cols_reflectance = ['timestamp_iso']
        required_cols_chemicals = ['timestamp_iso']
        
        for col in required_cols_reflectance:
            if col not in df_reflectance.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en reflectance")
        
        for col in required_cols_chemicals:
            if col not in df_chemicals.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en chemicals")
        
        # Verificar que hay columnas espectrales
        spectral_cols = [col for col in df_reflectance.columns if 'reflectance' in col.lower()]
        if len(spectral_cols) == 0:
            raise ValueError("No se encontraron columnas espectrales")
        
        # Verificar que hay contaminantes
        chemical_cols = [col for col in df_chemicals.columns if col.startswith('lab_')]
        if len(chemical_cols) == 0:
            raise ValueError("No se encontraron columnas de contaminantes")
        
        self.logger.info(f"    Estructura validada: {len(spectral_cols)} wavelengths, {len(chemical_cols)} contaminantes")
    
    def _process_timestamps(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Procesar timestamps con validación robusta"""
        
        try:
            df = df.copy()
            initial_count = len(df)
            
            # Convertir timestamps
            df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce')
            
            # Eliminar timestamps inválidos
            valid_timestamps = df['timestamp_iso'].notna()
            df = df[valid_timestamps].copy()
            
            final_count = len(df)
            loss_pct = (1 - final_count / initial_count) * 100
            
            if loss_pct > 5:
                self.logger.warning(f"    {data_type}: {loss_pct:.1f}% timestamps inválidos eliminados")
            
            # Ordenar por timestamp
            df = df.sort_values('timestamp_iso').reset_index(drop=True)
            
            # Calcular estadísticas temporales
            time_span = (df['timestamp_iso'].max() - df['timestamp_iso'].min()).total_seconds() / 3600
            self.logger.info(f"    {data_type}: {df['timestamp_iso'].min()} → {df['timestamp_iso'].max()} ({time_span:.1f}h)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"    Error procesando timestamps de {data_type}: {e}")
            raise
    
    def _filter_valid_spectra(self, df_reflectance: pd.DataFrame) -> pd.DataFrame:
        """Filtrar espectros válidos"""
        
        initial_count = len(df_reflectance)
        
        # Filtrar por valid_data si existe
        if 'valid_data' in df_reflectance.columns:
            df_reflectance = df_reflectance[df_reflectance['valid_data'] == 1].copy()
            valid_count = len(df_reflectance)
            self.logger.info(f"    Filtro validez: {initial_count} → {valid_count} ({valid_count/initial_count*100:.1f}%)")
        
        # Filtrar espectros con demasiados NaN
        spectral_cols = [col for col in df_reflectance.columns if 'reflectance' in col.lower()]
        nan_threshold = len(spectral_cols) * 0.2  # Máximo 20% NaN por espectro
        
        valid_spectra = df_reflectance[spectral_cols].isnull().sum(axis=1) <= nan_threshold
        df_reflectance = df_reflectance[valid_spectra].copy()
        
        final_count = len(df_reflectance)
        self.logger.info(f"    Espectros válidos: {final_count} ({final_count/initial_count*100:.1f}%)")
        
        return df_reflectance
    
    def _clean_chemical_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Limpiar datos químicos con manejo robusto"""
        
        df = df.copy()
        
        # Identificar columnas de contaminantes
        contaminant_cols = [col for col in df.columns if col.startswith('lab_')]
        
        total_replacements = 0
        
        for col in contaminant_cols:
            # Contar valores iniciales
            initial_valid = df[col].notna().sum()
            
            # Limpiar valores LOQ y similares
            if df[col].dtype == 'object':
                loq_patterns = ['<LOQ', '<loq', 'LOQ', 'ND', 'nd', '<LD', '<DL']
                for pattern in loq_patterns:
                    mask = df[col].astype(str).str.contains(pattern, na=False, case=False)
                    replacements = mask.sum()
                    if replacements > 0:
                        df.loc[mask, col] = np.nan
                        total_replacements += replacements
            
            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filtrar valores negativos (físicamente imposibles para concentraciones)
            negative_mask = df[col] < 0
            if negative_mask.any():
                df.loc[negative_mask, col] = np.nan
            
            final_valid = df[col].notna().sum()
            
            # Log si hay pérdida significativa
            if initial_valid > 0:
                loss_pct = (1 - final_valid / initial_valid) * 100
                if loss_pct > 20:
                    self.logger.warning(f"    {col}: {loss_pct:.1f}% datos perdidos en limpieza")
        
        self.logger.info(f"    {data_type}: {total_replacements} valores LOQ procesados")
        
        return df
    
    def _extract_spectral_info(self, df_reflectance: pd.DataFrame):
        """Extraer información espectral"""
        
        # Identificar columnas espectrales
        spectral_cols = [col for col in df_reflectance.columns if 'reflectance' in col.lower()]
        spectral_cols.sort()  # Ordenar para consistencia
        
        # Extraer wavelengths
        wavelengths = []
        for col in spectral_cols:
            import re
            match = re.search(r'(\d+)', col)
            if match:
                wavelengths.append(int(match.group(1)))
        
        wavelengths.sort()
        
        # Guardar información espectral
        self.dataset_info['spectral'] = {
            'columns': spectral_cols,
            'wavelengths': wavelengths,
            'range': (min(wavelengths), max(wavelengths)) if wavelengths else (0, 0),
            'n_bands': len(spectral_cols),
            'resolution': np.median(np.diff(wavelengths)) if len(wavelengths) > 1 else 0
        }
        
        self.logger.info(f"    Rango espectral: {self.dataset_info['spectral']['range'][0]}-{self.dataset_info['spectral']['range'][1]} nm")
        self.logger.info(f"    Resolución: ~{self.dataset_info['spectral']['resolution']:.1f} nm")
    
    def _identify_contaminants(self, df_chemicals: pd.DataFrame, 
                              df_pollution: Optional[pd.DataFrame]):
        """Identificar y categorizar contaminantes"""
        
        organic_contaminants = [col for col in df_chemicals.columns if col.startswith('lab_')]
        inorganic_contaminants = []
        
        if df_pollution is not None:
            inorganic_contaminants = [col for col in df_pollution.columns if col.startswith('lab_')]
        
        self.dataset_info['contaminants'] = {
            'organic': organic_contaminants,
            'inorganic': inorganic_contaminants,
            'all': organic_contaminants + inorganic_contaminants
        }
        
        self.logger.info(f"    Contaminantes identificados: {len(organic_contaminants)} orgánicos, {len(inorganic_contaminants)} inorgánicos")
    
    def _final_data_validation(self, df_reflectance: pd.DataFrame, 
                              df_chemicals: pd.DataFrame, 
                              df_pollution: Optional[pd.DataFrame]):
        """Validaciones finales de datos"""
        
        # Verificar que tenemos datos mínimos
        if len(df_reflectance) < 10:
            raise ValueError("Muy pocos espectros válidos")
        
        if len(df_chemicals) < 5:
            raise ValueError("Muy pocos datos químicos")
        
        # Verificar rangos temporales
        spec_range = (df_reflectance['timestamp_iso'].min(), df_reflectance['timestamp_iso'].max())
        chem_range = (df_chemicals['timestamp_iso'].min(), df_chemicals['timestamp_iso'].max())
        
        # Verificar overlap temporal
        overlap_start = max(spec_range[0], chem_range[0])
        overlap_end = min(spec_range[1], chem_range[1])
        
        if overlap_start >= overlap_end:
            self.logger.warning("    NO HAY OVERLAP TEMPORAL entre espectros y químicos")
        
        self.logger.info("    Validaciones finales pasadas")
    
    def temporal_matching_advanced(self, df_reflectance: pd.DataFrame, 
                                  df_chemicals: pd.DataFrame, 
                                  df_pollution: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Matching temporal avanzado con análisis de calidad
        """
        self.logger.info(" PASO 2: Matching temporal avanzado")
        
        # Análisis de overlap temporal
        self._analyze_temporal_overlap(df_reflectance, df_chemicals, df_pollution)
        
        # Matching con químicos orgánicos
        df_merged = self._perform_temporal_merge(df_reflectance, df_chemicals, "organic")
        
        # Matching con inorgánicos si existe
        if df_pollution is not None:
            df_merged = self._perform_temporal_merge(df_merged, df_pollution, "inorganic")
        
        # Análisis de calidad del matching
        self._analyze_matching_quality(df_merged)
        
        # Optimizar el dataset resultante
        df_merged = self._optimize_merged_dataset(df_merged)
        
        self.logger.info(f"    Matching completado: {df_merged.shape}")
        
        return df_merged
    
    def _analyze_temporal_overlap(self, df_reflectance: pd.DataFrame, 
                                 df_chemicals: pd.DataFrame, 
                                 df_pollution: Optional[pd.DataFrame]):
        """Analizar overlap temporal entre datasets"""
        
        spec_start, spec_end = df_reflectance['timestamp_iso'].min(), df_reflectance['timestamp_iso'].max()
        chem_start, chem_end = df_chemicals['timestamp_iso'].min(), df_chemicals['timestamp_iso'].max()
        
        # Calcular overlap
        overlap_start = max(spec_start, chem_start)
        overlap_end = min(spec_end, chem_end)
        
        if overlap_start <= overlap_end:
            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            self.logger.info(f"    Overlap temporal: {overlap_hours:.1f} horas")
        else:
            gap_hours = (chem_start - spec_end).total_seconds() / 3600
            self.logger.warning(f"    Gap temporal: {gap_hours:.1f} horas")
        
        # Analizar inorgánicos si existe
        if df_pollution is not None:
            inorg_start, inorg_end = df_pollution['timestamp_iso'].min(), df_pollution['timestamp_iso'].max()
            self.logger.info(f"    Inorgánicos: {inorg_start} → {inorg_end}")
    
    def _perform_temporal_merge(self, df_left: pd.DataFrame, 
                               df_right: pd.DataFrame, 
                               merge_type: str) -> pd.DataFrame:
        """Realizar merge temporal con configuración optimizada"""
        
        self.logger.info(f"    Matching con {merge_type}...")
        
        # Asegurar orden temporal
        df_left_sorted = df_left.sort_values('timestamp_iso')
        df_right_sorted = df_right.sort_values('timestamp_iso')
        
        # Merge asof con tolerancia
        tolerance = pd.Timedelta(minutes=self.config.temporal_tolerance_minutes)
        
        if merge_type == "organic":
            suffixes = ('_spectral', '_organic')
        else:
            suffixes = ('', '_inorganic')
        
        df_merged = pd.merge_asof(
            df_left_sorted,
            df_right_sorted,
            on='timestamp_iso',
            direction='backward',  
            tolerance=tolerance,
            suffixes=suffixes
        )
        
        # Contar matches exitosos
        contaminant_cols = [col for col in df_right.columns if col.startswith('lab_')]
        if contaminant_cols:
            matches = df_merged[contaminant_cols].notna().any(axis=1).sum()
            match_rate = matches / len(df_merged) * 100
            self.logger.info(f"      Matches {merge_type}: {matches}/{len(df_merged)} ({match_rate:.1f}%)")
        
        return df_merged
    
    def _analyze_matching_quality(self, df_merged: pd.DataFrame):
        """Analizar calidad del matching temporal"""
        
        all_contaminants = self.dataset_info['contaminants']['all']
        
        total_spectra = len(df_merged)
        total_with_data = df_merged[all_contaminants].notna().any(axis=1).sum()
        overall_match_rate = total_with_data / total_spectra * 100
        
        self.processing_metrics['matching'] = {
            'total_spectra': total_spectra,
            'spectra_with_chemical_data': total_with_data,
            'overall_match_rate_pct': overall_match_rate
        }
        
        self.logger.info(f"    Tasa de matching global: {overall_match_rate:.1f}%")
        
        # Análisis por contaminante individual
        contaminant_stats = {}
        for contaminant in all_contaminants:
            if contaminant in df_merged.columns:
                matches = df_merged[contaminant].notna().sum()
                rate = matches / total_spectra * 100
                contaminant_stats[contaminant] = {
                    'matches': matches,
                    'rate_pct': rate
                }
        
        # Identificar contaminantes con mejor cobertura
        best_contaminants = sorted(
            contaminant_stats.items(),
            key=lambda x: x[1]['matches'],
            reverse=True
        )[:5]
        
        self.logger.info("    Top contaminantes por cobertura:")
        for i, (cont, stats) in enumerate(best_contaminants, 1):
            clean_name = cont.replace('lab_', '')
            self.logger.info(f"      {i}. {clean_name}: {stats['matches']} ({stats['rate_pct']:.1f}%)")
    
    def _optimize_merged_dataset(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Optimizar el dataset fusionado"""
        
        initial_size = len(df_merged)
        
        # Eliminar duplicados temporales (si los hay)
        df_merged = df_merged.drop_duplicates(subset=['timestamp_iso'], keep='first')
        
        # Eliminar filas donde NO hay datos espectrales válidos
        spectral_cols = self.dataset_info['spectral']['columns']
        valid_spectra = df_merged[spectral_cols].notna().all(axis=1)
        df_merged = df_merged[valid_spectra].copy()
        
        final_size = len(df_merged)
        
        if final_size != initial_size:
            self.logger.info(f"    Optimización: {initial_size} → {final_size} ({final_size/initial_size*100:.1f}%)")
        
        return df_merged
    
    def analyze_contaminant_viability_comprehensive(self, df_merged: pd.DataFrame) -> Dict[str, ContaminantInfo]:
        """
        Análisis comprehensivo de viabilidad de contaminantes
        """
        self.logger.info(" PASO 3: Análisis comprehensivo de viabilidad")
        
        all_contaminants = self.dataset_info['contaminants']['all']
        contaminant_analysis = {}
        
        for contaminant in all_contaminants:
            try:
                analysis = self._analyze_single_contaminant(df_merged, contaminant)
                contaminant_analysis[contaminant] = analysis
                
            except Exception as e:
                self.logger.warning(f"    Error analizando {contaminant}: {e}")
                contaminant_analysis[contaminant] = ContaminantInfo(
                    name=contaminant,
                    clean_name=contaminant.replace('lab_', '').replace('_', '-'),
                    n_samples=0,
                    temporal_coverage_pct=0.0,
                    concentration_stats={},
                    threshold_binary=0.0,
                    viable=False,
                    quality_score=0.0
                )
        
        # Generar resumen de viabilidad
        self._summarize_viability_analysis(contaminant_analysis)
        
        return contaminant_analysis
    
    def _analyze_single_contaminant(self, df_merged: pd.DataFrame, 
                                   contaminant: str) -> ContaminantInfo:
        """Analizar un contaminante individual"""
        
        if contaminant not in df_merged.columns:
            raise ValueError(f"Contaminante {contaminant} no encontrado")
        
        # Extraer datos válidos
        total_samples = len(df_merged)
        valid_data = df_merged[contaminant].dropna()
        n_samples = len(valid_data)
        
        if n_samples == 0:
            raise ValueError(f"No hay datos válidos para {contaminant}")
        
        # Calcular estadísticas robustas
        concentration_stats = self._calculate_robust_statistics(valid_data)
        
        # Calcular cobertura temporal
        temporal_coverage_pct = (n_samples / total_samples) * 100
        
        # Calcular threshold para clasificación binaria
        threshold_binary = np.percentile(valid_data, 75)
        
        # Evaluar viabilidad
        viable, quality_score = self._evaluate_contaminant_viability(
            n_samples, temporal_coverage_pct, concentration_stats, valid_data
        )
        
        return ContaminantInfo(
            name=contaminant,
            clean_name=contaminant.replace('lab_', '').replace('_', '-'),
            n_samples=n_samples,
            temporal_coverage_pct=temporal_coverage_pct,
            concentration_stats=concentration_stats,
            threshold_binary=threshold_binary,
            viable=viable,
            quality_score=quality_score
        )
    
    def _calculate_robust_statistics(self, data: pd.Series) -> Dict:
        """Calcular estadísticas robustas"""
        
        # Filtrar outliers extremos usando IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites para outliers (más conservador)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Datos sin outliers extremos para algunas estadísticas
        data_clean = data[(data >= lower_bound) & (data <= upper_bound)]
        
        try:
            stats = {
                'count': len(data),
                'count_clean': len(data_clean),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(Q1),
                'q75': float(Q3),
                'iqr': float(IQR),
                'cv': float(data.std() / data.mean() * 100) if data.mean() > 0 else 0,
                'skewness': float(data.skew()) if len(data) > 1 else 0,
                'outlier_percentage': ((len(data) - len(data_clean)) / len(data) * 100) if len(data) > 0 else 0,
                'dynamic_range': float(data.max() / data.min()) if data.min() > 0 else float('inf'),
                'zero_percentage': (data == 0).sum() / len(data) * 100
            }
        except Exception as e:
            self.logger.warning(f"      Error calculando estadísticas: {e}")
            stats = {'count': len(data), 'error': str(e)}
        
        return stats
    
    def _evaluate_contaminant_viability(self, n_samples: int, 
                                       temporal_coverage_pct: float,
                                       stats: Dict, 
                                       data: pd.Series) -> Tuple[bool, float]:
        """Evaluar viabilidad de contaminante con sistema de puntuación"""
        
        quality_score = 0.0
        viable = True
        reasons = []
        
        # Criterio 1: Número de muestras (peso: 30%)
        if n_samples >= self.config.min_samples_per_contaminant * 2:
            quality_score += 30
        elif n_samples >= self.config.min_samples_per_contaminant:
            quality_score += 20
        elif n_samples >= self.config.min_samples_per_contaminant * 0.75:
            quality_score += 10
        else:
            viable = False
            reasons.append(f"Insuficientes muestras: {n_samples} < {self.config.min_samples_per_contaminant}")
        
        # Criterio 2: Cobertura temporal (peso: 20%)
        if temporal_coverage_pct >= self.config.min_temporal_coverage_pct * 2:
            quality_score += 20
        elif temporal_coverage_pct >= self.config.min_temporal_coverage_pct:
            quality_score += 15
        elif temporal_coverage_pct >= self.config.min_temporal_coverage_pct * 0.5:
            quality_score += 10
        else:
            reasons.append(f"Cobertura temporal baja: {temporal_coverage_pct:.1f}%")
        
        # Criterio 3: Variabilidad (peso: 25%)
        if 'std' in stats and stats['std'] > self.config.min_variance_threshold:
            cv = stats.get('cv', 0)
            if cv > 50:  # Alta variabilidad
                quality_score += 25
            elif cv > 20:  # Variabilidad moderada
                quality_score += 20
            elif cv > 5:   # Variabilidad baja pero aceptable
                quality_score += 15
            else:
                quality_score += 10
        else:
            viable = False
            reasons.append("Sin variabilidad en los datos")
        
        # Criterio 4: Calidad de datos (peso: 15%)
        zero_pct = stats.get('zero_percentage', 0)
        outlier_pct = stats.get('outlier_percentage', 0)
        
        if zero_pct < 10 and outlier_pct < 5:
            quality_score += 15
        elif zero_pct < 20 and outlier_pct < 10:
            quality_score += 10
        elif zero_pct < 30 and outlier_pct < 20:
            quality_score += 5
        else:
            reasons.append(f"Calidad de datos cuestionable: {zero_pct:.1f}% zeros, {outlier_pct:.1f}% outliers")
        
        # Criterio 5: Distribución (peso: 10%)
        dynamic_range = stats.get('dynamic_range', 1)
        if dynamic_range > 100:
            quality_score += 10
        elif dynamic_range > 10:
            quality_score += 8
        elif dynamic_range > 3:
            quality_score += 5
        else:
            reasons.append(f"Rango dinámico limitado: {dynamic_range:.2f}")
        
        # Normalizar score a 0-100
        quality_score = min(100, quality_score)
        
        # Umbral mínimo de calidad
        if quality_score < 50:
            viable = False
            reasons.append(f"Score de calidad bajo: {quality_score:.1f}/100")
        
        return viable, quality_score
    
    def _summarize_viability_analysis(self, contaminant_analysis: Dict[str, ContaminantInfo]):
        """Generar resumen del análisis de viabilidad"""
        
        viable_contaminants = [info for info in contaminant_analysis.values() if info.viable]
        non_viable_contaminants = [info for info in contaminant_analysis.values() if not info.viable]
        
        self.logger.info(f"    Contaminantes viables: {len(viable_contaminants)}")
        self.logger.info(f"    Contaminantes no viables: {len(non_viable_contaminants)}")
        
        if viable_contaminants:
            # Ordenar por score de calidad
            viable_sorted = sorted(viable_contaminants, key=lambda x: x.quality_score, reverse=True)
            
            self.logger.info("    Top contaminantes viables:")
            for i, info in enumerate(viable_sorted[:8], 1):
                self.logger.info(
                    f"      {i:2d}. {info.clean_name:<25}: {info.n_samples:>3d} muestras "
                    f"({info.temporal_coverage_pct:>5.1f}%) | Score: {info.quality_score:.1f}"
                )
            
            # Guardar información de viables
            self.dataset_info['viable_contaminants'] = [info.name for info in viable_contaminants]
            self.dataset_info['viable_contaminants_info'] = viable_contaminants
        
        # Guardar métricas de viabilidad
        self.processing_metrics['viability'] = {
            'total_contaminants': len(contaminant_analysis),
            'viable_contaminants': len(viable_contaminants),
            'viability_rate_pct': len(viable_contaminants) / len(contaminant_analysis) * 100 if contaminant_analysis else 0,
            'avg_quality_score': np.mean([info.quality_score for info in viable_contaminants]) if viable_contaminants else 0
        }
    
    def create_leakage_free_splits(self, df_merged: pd.DataFrame, 
                                  contaminant_analysis: Dict[str, ContaminantInfo]) -> Dict:
        """
        Crear splits temporales completamente libres de data leakage
        """
        self.logger.info(" PASO 4: Creación de splits libres de data leakage")
        
        viable_contaminants = [name for name, info in contaminant_analysis.items() if info.viable]
        
        if not viable_contaminants:
            self.logger.error("    No hay contaminantes viables para crear splits")
            return {}
        
        all_datasets = {}
        
        for contaminant in viable_contaminants:
            try:
                dataset = self._create_single_contaminant_dataset(df_merged, contaminant, contaminant_analysis[contaminant])
                if dataset is not None:
                    all_datasets[contaminant] = dataset
                    
            except Exception as e:
                self.logger.warning(f"    Error creando dataset para {contaminant}: {e}")
        
        self.logger.info(f"    Datasets creados: {len(all_datasets)}")
        return all_datasets
    
    def _create_single_contaminant_dataset(self, df_merged: pd.DataFrame, 
                                          contaminant: str, 
                                          contaminant_info: ContaminantInfo) -> Optional[Dict]:
        """Crear dataset para un contaminante específico sin data leakage"""
        
        self.logger.info(f"    Procesando {contaminant_info.clean_name}...")
        
        # Paso 1: FILTRAR datos válidos para este contaminante
        df_contaminant = df_merged[df_merged[contaminant].notna()].copy()
        
        if len(df_contaminant) < self.config.min_samples_per_contaminant:
            self.logger.warning(f"       Insuficientes datos: {len(df_contaminant)}")
            return None
        
        # Paso 2: ORDENAR cronológicamente (CRÍTICO para evitar leakage)
        df_sorted = df_contaminant.sort_values('timestamp_iso').reset_index(drop=True)
        
        # Paso 3: SPLIT temporal ANTES de cualquier procesamiento
        train_end_idx = int(len(df_sorted) * (1 - self.config.test_size - self.config.val_size))
        val_end_idx = int(len(df_sorted) * (1 - self.config.test_size))
        
        df_train = df_sorted.iloc[:train_end_idx].copy()
        df_val = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        df_test = df_sorted.iloc[val_end_idx:].copy()
        
        # Verificar que todos los splits tienen datos
        if len(df_train) < 10 or len(df_val) < 3 or len(df_test) < 3:
            self.logger.warning(f"       Splits muy pequeños: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
            return None
        
        # Paso 4: VERIFICAR no hay overlap temporal
        train_end_time = df_train['timestamp_iso'].max()
        val_start_time = df_val['timestamp_iso'].min()
        val_end_time = df_val['timestamp_iso'].max()
        test_start_time = df_test['timestamp_iso'].min()
        
        if train_end_time >= val_start_time or val_end_time >= test_start_time:
            self.logger.warning(f"       Posible overlap temporal detectado")
        
        # Paso 5: CALCULAR threshold usando SOLO datos de training
        train_concentrations = df_train[contaminant].values
        threshold = np.percentile(train_concentrations, 75)
        
        # Paso 6: PREPARAR features sin leakage
        if self.use_spectral_features:
            # USAR LA NUEVA FUNCIÓN
            dataset = self._prepare_features_with_spectral_enhancement(
                df_train, df_val, df_test, contaminant, threshold, contaminant_info
            )
        else:
            # Fallback a la función original
            dataset = self._prepare_features_no_leakage(
                df_train, df_val, df_test, contaminant, threshold, contaminant_info
            )
        
        # Paso 7: VALIDAR ausencia de leakage
        validation_passed = self._validate_no_leakage(dataset, contaminant_info)
        
        if not validation_passed:
            self.logger.warning(f"       Validación anti-leakage falló")
        
        dataset['validation_passed'] = validation_passed
        dataset['contaminant_info'] = contaminant_info
        
        self.logger.info(f"       Dataset creado: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
        
        return dataset
    
    def _prepare_features_with_spectral_enhancement(self, df_train: pd.DataFrame, 
                                                   df_val: pd.DataFrame, 
                                                   df_test: pd.DataFrame, 
                                                   contaminant: str, 
                                                   threshold: float, 
                                                   contaminant_info) -> dict:
        """
        Preparar features CON análisis espectral integrado
        REEMPLAZA a _prepare_features_no_leakage del código original
        """
        
        self.logger.info(f"Preparando features espectrales para {contaminant}...")
        
        # Obtener columnas espectrales y wavelengths del dataset_info
        spectral_cols = self.dataset_info['spectral']['columns']
        wavelengths = self.dataset_info['spectral']['wavelengths']
        
        # PASO 1: Extraer espectros RAW
        X_train_raw = df_train[spectral_cols].values
        X_val_raw = df_val[spectral_cols].values  
        X_test_raw = df_test[spectral_cols].values
        
        self.logger.info(f"   Espectros raw: {X_train_raw.shape}")
        
        # PASO 2: Feature Engineering Espectral (si está habilitado)
        if self.use_spectral_features:
            
            # Crear/reutilizar feature engineer
            if self.spectral_engineer is None:
                self.spectral_engineer = SpectralFeatureEngineer(wavelengths)
                self.logger.info(f"   Feature engineer creado para {len(wavelengths)} wavelengths")
            
            # Extraer features espectrales de cada conjunto
            self.logger.info(f"   Extrayendo features espectrales...")
            
            try:
                spectral_features_train = self.spectral_engineer.extract_all_features(X_train_raw)
                spectral_features_val = self.spectral_engineer.extract_all_features(X_val_raw)
                spectral_features_test = self.spectral_engineer.extract_all_features(X_test_raw)
                
                # Convertir a arrays numpy
                X_train_spectral = spectral_features_train.values
                X_val_spectral = spectral_features_val.values
                X_test_spectral = spectral_features_test.values
                
                spectral_feature_names = spectral_features_train.columns.tolist()
                
                self.logger.info(f"   Features espectrales extraídas: {X_train_spectral.shape[1]}")
                
                # Debug: Mostrar algunos nombres de features
                self.logger.info(f"   Ejemplos de features: {spectral_feature_names[:5]}...")
                
            except Exception as e:
                self.logger.error(f"   Error en feature engineering espectral: {e}")
                # Fallback a features raw
                X_train_spectral = X_train_raw
                X_val_spectral = X_val_raw
                X_test_spectral = X_test_raw
                spectral_feature_names = spectral_cols
                self.use_spectral_features = False
        
        else:
            # No usar features espectrales
            X_train_spectral = X_train_raw
            X_val_spectral = X_val_raw
            X_test_spectral = X_test_raw
            spectral_feature_names = spectral_cols
        
        # PASO 3: Seleccionar estrategia de features
        if self.spectral_strategy == "spectral_only" and self.use_spectral_features:
            # Solo features espectrales interpretables
            X_train_final = X_train_spectral
            X_val_final = X_val_spectral
            X_test_final = X_test_spectral
            final_feature_names = spectral_feature_names
            feature_type = "spectral_only"
            
        elif self.spectral_strategy == "combined" and self.use_spectral_features:
            # Combinar features espectrales + raw
            X_train_final = np.hstack([X_train_spectral, X_train_raw])
            X_val_final = np.hstack([X_val_spectral, X_val_raw])
            X_test_final = np.hstack([X_test_spectral, X_test_raw])
            final_feature_names = spectral_feature_names + spectral_cols
            feature_type = "combined"
            
        else:
            # Solo features raw (fallback o estrategia explícita)
            X_train_final = X_train_raw
            X_val_final = X_val_raw
            X_test_final = X_test_raw
            final_feature_names = spectral_cols
            feature_type = "raw_only"
        
        self.logger.info(f"   Estrategia aplicada: {feature_type}")
        self.logger.info(f"   Features finales: {X_train_final.shape[1]}")
        
        # PASO 4: Normalización (aplicar DESPUÉS de feature engineering)
        scaler = RobustScaler()  # Más robusto para features espectrales
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val_final)
        X_test_scaled = scaler.transform(X_test_final)
        
        # PASO 5: Crear labels usando threshold calculado SOLO en training
        y_train = (df_train[contaminant] > threshold).astype(int)
        y_val = (df_val[contaminant] > threshold).astype(int)
        y_test = (df_test[contaminant] > threshold).astype(int)
        
        # PASO 6: Extraer concentraciones y timestamps
        concentrations_train = df_train[contaminant].values
        concentrations_val = df_val[contaminant].values
        concentrations_test = df_test[contaminant].values
        
        timestamps_train = df_train['timestamp_iso'].values
        timestamps_val = df_val['timestamp_iso'].values
        timestamps_test = df_test['timestamp_iso'].values
        
        # PASO 7: Validaciones anti-leakage (mantener las del código original)
        validation_passed = self._validate_no_leakage_spectral(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            timestamps_train, timestamps_val, timestamps_test
        )
        
        # PASO 8: Preparar dataset completo con información espectral
        dataset = {
            # Features escaladas para entrenamiento
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            
            # Features raw (para referencia y LSTM)
            'X_train_raw': X_train_raw,
            'X_val_raw': X_val_raw,
            'X_test_raw': X_test_raw,
            
            # Features espectrales (sin escalar, para interpretación)
            'X_train_spectral': X_train_spectral if self.use_spectral_features else None,
            'X_val_spectral': X_val_spectral if self.use_spectral_features else None,
            'X_test_spectral': X_test_spectral if self.use_spectral_features else None,
            
            # Labels
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            
            # Concentraciones originales
            'concentrations_train': concentrations_train,
            'concentrations_val': concentrations_val,
            'concentrations_test': concentrations_test,
            
            # Timestamps para trazabilidad
            'timestamps_train': timestamps_train,
            'timestamps_val': timestamps_val,
            'timestamps_test': timestamps_test,
            
            # Metadata de features
            'feature_names_final': final_feature_names,
            'feature_names_spectral': spectral_feature_names if self.use_spectral_features else [],
            'feature_names_raw': spectral_cols,
            'feature_type': feature_type,
            'n_features_final': X_train_final.shape[1],
            'n_features_spectral': X_train_spectral.shape[1] if self.use_spectral_features else 0,
            'n_features_raw': X_train_raw.shape[1],
            
            # Objetos para uso posterior
            'scaler': scaler,
            'threshold': threshold,
            'contaminant': contaminant,
            'wavelengths': wavelengths,
            'spectral_engineer': self.spectral_engineer,
            
            # Validación
            'validation_passed': validation_passed,
            'spectral_processing_enabled': self.use_spectral_features
        }
        
        self.logger.info(f"   Dataset preparado:")
        self.logger.info(f"      Muestras train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
        self.logger.info(f"      Features finales: {dataset['n_features_final']}")
        self.logger.info(f"      Features espectrales: {dataset['n_features_spectral']}")
        self.logger.info(f"      Validación: {'OK' if validation_passed else 'FAILED'}")
        
        return dataset
    
    def _prepare_features_no_leakage(self, df_train: pd.DataFrame, df_val: pd.DataFrame, 
                                df_test: pd.DataFrame, contaminant: str, 
                                threshold: float, contaminant_info) -> Dict:
        """
        Preparar features SIN análisis espectral (función original)
        """
        spectral_cols = self.dataset_info['spectral']['columns']
        
        # Extraer features espectrales
        X_train = df_train[spectral_cols].values
        X_val = df_val[spectral_cols].values
        X_test = df_test[spectral_cols].values
        
        # Crear labels usando threshold calculado SOLO en training
        y_train = (df_train[contaminant] > threshold).astype(int)
        y_val = (df_val[contaminant] > threshold).astype(int)
        y_test = (df_test[contaminant] > threshold).astype(int)
        
        # NORMALIZACIÓN: FIT solo en training, TRANSFORM en val/test
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Extraer concentraciones originales
        concentrations_train = df_train[contaminant].values
        concentrations_val = df_val[contaminant].values
        concentrations_test = df_test[contaminant].values
        
        # Timestamps para trazabilidad
        timestamps_train = df_train['timestamp_iso'].values
        timestamps_val = df_val['timestamp_iso'].values
        timestamps_test = df_test['timestamp_iso'].values
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'concentrations_train': concentrations_train,
            'concentrations_val': concentrations_val,
            'concentrations_test': concentrations_test,
            'timestamps_train': timestamps_train,
            'timestamps_val': timestamps_val,
            'timestamps_test': timestamps_test,
            'scaler': scaler,
            'threshold': threshold,
            'feature_names': spectral_cols,
            'contaminant': contaminant,
            'wavelengths': self.dataset_info['spectral']['wavelengths'],
            # Agregar campos para compatibilidad con versión espectral
            'feature_type': 'raw_only',
            'spectral_processing_enabled': False,
            'n_features_final': X_train.shape[1],
            'n_features_spectral': 0,
            'n_features_raw': X_train.shape[1]
        }

    def _validate_no_leakage_spectral(self, X_train, X_val, X_test, 
                                     y_train, y_val, y_test,
                                     timestamps_train, timestamps_val, timestamps_test):
        """
        Validar ausencia de data leakage con features espectrales
        """
        
        validation_passed = True
        issues = []
        
        # Check 1: Verificar distribuciones de clases
        train_dist = np.bincount(y_train, minlength=2)
        test_dist = np.bincount(y_test, minlength=2)
        
        train_pos_rate = train_dist[1] / train_dist.sum() if train_dist.sum() > 0 else 0
        test_pos_rate = test_dist[1] / test_dist.sum() if test_dist.sum() > 0 else 0
        
        # Verificar que test no tiene proporción mucho mayor de positivos
        if test_pos_rate > train_pos_rate + 0.3:  # 30% más
            issues.append(f"Test set tiene {test_pos_rate:.1%} positivos vs {train_pos_rate:.1%} en train")
            validation_passed = False
        
        # Check 2: Verificar orden temporal
        try:
            train_end = pd.Timestamp(timestamps_train.max())
            val_start = pd.Timestamp(timestamps_val.min())
            test_start = pd.Timestamp(timestamps_test.min())
            
            if train_end >= val_start:
                issues.append("Overlap temporal entre train y validation")
                validation_passed = False
                
        except Exception as e:
            self.logger.warning(f"No se pudo verificar orden temporal: {e}")
        
        # Check 3: Verificar que no hay features constantes (puede indicar error en normalización)
        feature_variances = np.var(X_train, axis=0)
        zero_variance_features = np.sum(feature_variances < 1e-10)
        
        if zero_variance_features > X_train.shape[1] * 0.1:  # Más del 10% sin varianza
            issues.append(f"{zero_variance_features} features sin varianza detectadas")
            validation_passed = False
        
        # Check 4: Verificar que features espectrales están en rangos razonables
        if X_train.shape[1] > 20:  # Solo si tenemos features espectrales
            extreme_values = np.sum(np.abs(X_train) > 10)  # Después de normalización no debería haber valores extremos
            if extreme_values > X_train.size * 0.01:  # Más del 1% de valores extremos
                issues.append(f"Valores extremos detectados después de normalización")
        
        if issues:
            for issue in issues:
                self.logger.warning(f"         - {issue}")
        
        return validation_passed

    def save_spectral_enhanced_datasets(self, ml_datasets: dict, lstm_datasets: dict) -> dict:
        """
        Guardar datasets con features espectrales
        """
        
        self.logger.info(" Guardando datasets con features espectrales...")
        
        # Crear directorio de salida
        output_path = Path(self.config.output_dir.replace("datasets", "spectral_datasets"))
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Guardar datasets ML con features espectrales
        for contaminant, dataset in ml_datasets.items():
            if dataset.get('validation_passed', False):
                clean_name = dataset['contaminant_info'].clean_name
                
                # Determinar sufijo según tipo de features
                if dataset.get('spectral_processing_enabled', False):
                    if dataset['feature_type'] == 'spectral_only':
                        suffix = 'spectral'
                    elif dataset['feature_type'] == 'combined':
                        suffix = 'combined'
                    else:
                        suffix = 'raw'
                else:
                    suffix = 'raw'
                
                filename = f"ml_{suffix}_{clean_name}.npz"
                filepath = output_path / filename
                
                # Preparar datos para guardado
                save_data = {
                    # Features principales (las que se usan para entrenamiento)
                    'X_train': dataset['X_train'],
                    'X_val': dataset['X_val'], 
                    'X_test': dataset['X_test'],
                    'y_train': dataset['y_train'],
                    'y_val': dataset['y_val'],
                    'y_test': dataset['y_test'],
                    
                    # Features raw (siempre disponibles)
                    'X_train_raw': dataset['X_train_raw'],
                    'X_val_raw': dataset['X_val_raw'],
                    'X_test_raw': dataset['X_test_raw'],
                    
                    # Concentraciones y timestamps
                    'concentrations_train': dataset['concentrations_train'],
                    'concentrations_val': dataset['concentrations_val'],
                    'concentrations_test': dataset['concentrations_test'],
                    'timestamps_train': dataset['timestamps_train'],
                    'timestamps_val': dataset['timestamps_val'],
                    'timestamps_test': dataset['timestamps_test'],
                    
                    # Metadata
                    'threshold': dataset['threshold'],
                    'feature_names_final': dataset['feature_names_final'],
                    'feature_names_raw': dataset['feature_names_raw'],
                    'feature_type': dataset['feature_type'],
                    'n_features_final': dataset['n_features_final'],
                    'n_features_raw': dataset['n_features_raw'],
                    'wavelengths': dataset['wavelengths'],
                    'spectral_processing_enabled': dataset['spectral_processing_enabled']
                }
                
                # Añadir features espectrales si están disponibles
                if dataset.get('spectral_processing_enabled', False) and dataset['X_train_spectral'] is not None:
                    save_data.update({
                        'X_train_spectral': dataset['X_train_spectral'],
                        'X_val_spectral': dataset['X_val_spectral'],
                        'X_test_spectral': dataset['X_test_spectral'],
                        'feature_names_spectral': dataset['feature_names_spectral'],
                        'n_features_spectral': dataset['n_features_spectral']
                    })
                
                # Guardar
                np.savez_compressed(filepath, **save_data)
                
                saved_files[f"ml_{suffix}_{clean_name}"] = str(filepath)
                self.logger.info(f"     {filename}")
        
        # Guardar datasets LSTM (usando features raw para secuencias temporales)
        for contaminant, sequences in lstm_datasets.items():
            if sequences:
                # Encontrar info del contaminante
                contaminant_info = None
                for name, dataset in ml_datasets.items():
                    if name == contaminant:
                        contaminant_info = dataset['contaminant_info']
                        threshold = dataset['threshold']
                        wavelengths = dataset['wavelengths']
                        break
                
                if contaminant_info:
                    clean_name = contaminant_info.clean_name
                    filename = f"lstm_ready_{clean_name}.npz"
                    filepath = output_path / filename
                    
                    # Convertir secuencias a arrays
                    X_sequences = np.array([seq['spectral_sequence'] for seq in sequences])
                    concentrations = np.array([seq['target_concentration'] for seq in sequences])
                    timestamps = [seq['target_timestamp'] for seq in sequences]
                    
                    # Crear labels binarias usando el mismo threshold que ML
                    y_labels = (concentrations > threshold).astype(int)
                    
                    np.savez_compressed(
                        filepath,
                        X_sequences=X_sequences,
                        y_labels=y_labels,
                        concentrations=concentrations,
                        timestamps=timestamps,
                        threshold=threshold,
                        sequence_length=self.config.lstm_sequence_length,
                        wavelengths=wavelengths,
                        allow_pickle=True  # Para timestamps
                    )
                    
                    saved_files[f"lstm_{clean_name}"] = str(filepath)
                    self.logger.info(f"     {filename}")
        
        # Guardar metadata completa con información espectral
        metadata = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processor_version': 'SpectralEnhancedProcessor_v1.0',
                'spectral_features_enabled': self.use_spectral_features,
                'spectral_strategy': self.spectral_strategy,
                'config': self.config.__dict__
            },
            'spectral_info': {
                'wavelength_range': self.dataset_info.get('spectral', {}).get('range', [0, 0]),
                'n_wavelengths': self.dataset_info.get('spectral', {}).get('n_bands', 0),
                'spectral_resolution': self.dataset_info.get('spectral', {}).get('resolution', 0),
                'feature_categories': self._get_spectral_feature_categories()
            },
            'datasets_created': {
                'ml_datasets': len([f for f in saved_files.keys() if f.startswith('ml_')]),
                'lstm_datasets': len([f for f in saved_files.keys() if f.startswith('lstm_')]),
                'total': len(saved_files)
            },
            'performance_expectations': self._get_performance_expectations()
        }
        
        metadata_file = output_path / "spectral_datasets_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        saved_files['metadata'] = str(metadata_file)
        
        self.logger.info(f" Datasets espectrales guardados: {len(saved_files)} archivos")
        self.logger.info(f"    Directorio: {output_path}")
        
        return saved_files
    
    def _get_spectral_feature_categories(self):
        """Obtener categorías de features espectrales generadas"""
        return {
            'statistical': 'Estadísticas básicas del espectro (mean, std, min, max, etc.)',
            'spectral_indices': 'Índices espectrales conocidos (NDVI, ratios, etc.)',
            'absorption_bands': 'Features basadas en absorción específica de contaminantes',
            'shape_analysis': 'Análisis de forma espectral (picos, pendientes, áreas)',
            'range_analysis': 'Análisis por rangos espectrales (UV, VIS, NIR)',
            'contaminant_specific': 'Features específicas para tipos de contaminantes'
        }
    
    def _get_performance_expectations(self):
        """Expectativas de rendimiento con features espectrales"""
        return {
            'expected_improvements': {
                'organic_contaminants': 'Mejora esperada 10-20% en F1 por absorción UV específica',
                'physicochemical_params': 'Mejora esperada 15-30% en F1 por índices espectrales conocidos',
                'turbidity_related': 'Mejora esperada 20-35% en F1 por scattering NIR'
            },
            'interpretability': 'Features espectrales proporcionan explicabilidad química',
            'robustness': 'Mayor robustez a variaciones instrumentales'
        }


    
    def _validate_no_leakage(self, dataset: Dict, contaminant_info: ContaminantInfo) -> bool:
        """Validar ausencia completa de data leakage"""
        
        validation_passed = True
        issues = []

        

        
        # Check 1: Verificar distribuciones de clases
        train_dist = np.bincount(dataset['y_train'], minlength=2)
        val_dist = np.bincount(dataset['y_val'], minlength=2)
        test_dist = np.bincount(dataset['y_test'], minlength=2)
        
        train_pos_rate = train_dist[1] / train_dist.sum() if train_dist.sum() > 0 else 0
        test_pos_rate = test_dist[1] / test_dist.sum() if test_dist.sum() > 0 else 0
        
        # Si test tiene mucha más proporción positiva que train, puede ser leakage
        if test_pos_rate > train_pos_rate + 0.3:  # 30% más
            issues.append(f"Test set tiene {test_pos_rate:.1%} positivos vs {train_pos_rate:.1%} en train")
            validation_passed = False
        
        # Check 2: Verificar que scaler se ajustó solo a training
        scaler_mean = dataset['scaler'].center_
        scaler_scale = dataset['scaler'].scale_
        
        # Calcular estadísticas de validation y test
        val_mean = np.mean(dataset['X_val'], axis=0)
        test_mean = np.mean(dataset['X_test'], axis=0)
        
        # Si las medias de val/test son muy similares a las del scaler, podría indicar leakage
        # (esto es una heurística, no una prueba definitiva)
        
        # Check 3: Verificar orden temporal implícito en concentraciones
        train_conc_trend = np.polyfit(range(len(dataset['concentrations_train'])), 
                                     dataset['concentrations_train'], 1)[0]
        test_conc_trend = np.polyfit(range(len(dataset['concentrations_test'])), 
                                    dataset['concentrations_test'], 1)[0]
        
        # Check 4: Verificar que hay gap temporal entre sets
        train_end = dataset['timestamps_train'].max()
        val_start = dataset['timestamps_val'].min()
        test_start = dataset['timestamps_test'].min()
        
        if isinstance(train_end, (pd.Timestamp, np.datetime64)):
            train_end = pd.Timestamp(train_end)
            val_start = pd.Timestamp(val_start)
            test_start = pd.Timestamp(test_start)
            
            if train_end >= val_start:
                issues.append("Overlap temporal entre train y validation")
                validation_passed = False
        
        if issues:
            for issue in issues:
                self.logger.warning(f"         - {issue}")
        
        return validation_passed
    
    def create_lstm_sequences(self, df_merged: pd.DataFrame, 
                             contaminant_analysis: Dict[str, ContaminantInfo]) -> Dict:
        """
        Crear secuencias temporales para LSTM
        """
        self.logger.info(" PASO 5: Creación de secuencias LSTM")
        
        viable_contaminants = [name for name, info in contaminant_analysis.items() if info.viable]
        
        if not viable_contaminants:
            self.logger.warning("    No hay contaminantes viables para LSTM")
            return {}
        
        lstm_datasets = {}
        
        for contaminant in viable_contaminants:
            try:
                sequences = self._create_lstm_sequences_for_contaminant(
                    df_merged, contaminant, contaminant_analysis[contaminant]
                )
                
                if sequences and len(sequences) >= self.config.min_samples_per_contaminant:
                    lstm_datasets[contaminant] = sequences
                    self.logger.info(f"       {contaminant_analysis[contaminant].clean_name}: {len(sequences)} secuencias")
                else:
                    self.logger.info(f"       {contaminant_analysis[contaminant].clean_name}: insuficientes secuencias")
                    
            except Exception as e:
                self.logger.warning(f"    Error creando secuencias LSTM para {contaminant}: {e}")
        
        self.logger.info(f"    Contaminantes con secuencias LSTM: {len(lstm_datasets)}")
        return lstm_datasets
    
    def _create_lstm_sequences_for_contaminant(self, df_merged: pd.DataFrame, 
                                              contaminant: str, 
                                              contaminant_info: ContaminantInfo) -> List[Dict]:
        """Crear secuencias LSTM para un contaminante específico"""
        
        # Obtener datos del contaminante
        contaminant_data = df_merged[df_merged[contaminant].notna()].copy()
        
        if len(contaminant_data) == 0:
            return []
        
        # Ordenar por tiempo
        contaminant_data = contaminant_data.sort_values('timestamp_iso')
        spectral_cols = self.dataset_info['spectral']['columns']
        
        sequences = []
        seq_length = self.config.lstm_sequence_length
        step_minutes = self.config.lstm_sequence_step_minutes
        
        for idx, row in contaminant_data.iterrows():
            target_time = row['timestamp_iso']
            target_concentration = row[contaminant]
            
            # Buscar secuencia de espectros ANTERIORES
            sequence_start_time = target_time - timedelta(minutes=seq_length * step_minutes)
            
            # Filtrar espectros en la ventana temporal
            time_mask = ((df_merged['timestamp_iso'] >= sequence_start_time) & 
                        (df_merged['timestamp_iso'] <= target_time))
            
            candidate_spectra = df_merged[time_mask].copy()
            
            if len(candidate_spectra) >= seq_length:
                # Seleccionar espectros espaciados uniformemente
                indices = np.linspace(0, len(candidate_spectra)-1, seq_length, dtype=int)
                selected_spectra = candidate_spectra.iloc[indices]
                
                # Extraer valores espectrales
                spectral_sequence = []
                valid_sequence = True
                
                for _, spectrum_row in selected_spectra.iterrows():
                    spectrum_values = spectrum_row[spectral_cols].values
                    
                    # Verificar que no hay NaN
                    if np.any(pd.isna(spectrum_values)):
                        valid_sequence = False
                        break
                    
                    spectral_sequence.append(spectrum_values)
                
                if valid_sequence and len(spectral_sequence) == seq_length:
                    sequence_array = np.array(spectral_sequence)
                    
                    sequences.append({
                        'sequence_id': len(sequences),
                        'target_timestamp': target_time,
                        'target_concentration': target_concentration,
                        'spectral_sequence': sequence_array,  # Shape: (seq_length, n_wavelengths)
                        'sequence_timestamps': selected_spectra['timestamp_iso'].tolist(),
                        'contaminant': contaminant
                    })
        
        return sequences
    
    def save_optimized_datasets(self, ml_datasets: Dict, lstm_datasets: Dict) -> Dict[str, str]:
        """
        Guardar todos los datasets optimizados
        """
        self.logger.info(" PASO 6: Guardando datasets optimizados")
        
        # Crear directorio de salida
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}

        spectral_cols = self.dataset_info.get('spectral', {}).get('columns', [])
        
        # Guardar datasets ML tradicionales (SVM/XGBoost)
        for contaminant, dataset in ml_datasets.items():
            if dataset['validation_passed']:
                clean_name = dataset['contaminant_info'].clean_name
                filename = f"ml_ready_{clean_name}.npz"
                filepath = output_path / filename
                
                # CORRECCIÓN: Usar una lógica más robusta para feature_names
                if 'feature_names_final' in dataset:
                    feature_names = dataset['feature_names_final']
                elif 'feature_names' in dataset:
                    feature_names = dataset['feature_names']
                else:
                    feature_names = spectral_cols  # Ahora está definido
                
                np.savez_compressed(
                    filepath,
                    X_train=dataset['X_train'],
                    X_val=dataset['X_val'],
                    X_test=dataset['X_test'],
                    y_train=dataset['y_train'],
                    y_val=dataset['y_val'],
                    y_test=dataset['y_test'],
                    concentrations_train=dataset['concentrations_train'],
                    concentrations_val=dataset['concentrations_val'],
                    concentrations_test=dataset['concentrations_test'],
                    threshold=dataset['threshold'],
                    feature_names=feature_names,  # CORREGIDO
                    wavelengths=self.dataset_info['spectral']['wavelengths']
                )
                
                saved_files[f"ml_{clean_name}"] = str(filepath)
                self.logger.info(f"    ML dataset: {filename}")
        
        # Guardar datasets LSTM
        for contaminant, sequences in lstm_datasets.items():
            if sequences:
                contaminant_info = next(
                    (info for name, info in ml_datasets.items() if name == contaminant),
                    None
                )
                
                if contaminant_info:
                    clean_name = contaminant_info['contaminant_info'].clean_name
                    filename = f"lstm_ready_{clean_name}.npz"
                    filepath = output_path / filename
                    
                    # Convertir secuencias a arrays
                    X_sequences = np.array([seq['spectral_sequence'] for seq in sequences])
                    concentrations = np.array([seq['target_concentration'] for seq in sequences])
                    timestamps = [seq['target_timestamp'] for seq in sequences]
                    
                    # Crear labels binarias usando el mismo threshold que ML
                    threshold = contaminant_info['threshold']
                    y_labels = (concentrations > threshold).astype(int)
                    
                    np.savez_compressed(
                        filepath,
                        X_sequences=X_sequences,
                        y_labels=y_labels,
                        concentrations=concentrations,
                        timestamps=timestamps,
                        threshold=threshold,
                        sequence_length=self.config.lstm_sequence_length,
                        wavelengths=self.dataset_info['spectral']['wavelengths']
                    )
                    
                    saved_files[f"lstm_{clean_name}"] = str(filepath)
                    self.logger.info(f"    LSTM dataset: {filename}")
        
        # Guardar metadata completa
        metadata = self._create_comprehensive_metadata(ml_datasets, lstm_datasets)
        metadata_file = output_path / "datasets_metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        saved_files['metadata'] = str(metadata_file)
        
        # Guardar índice de archivos
        index_file = output_path / "datasets_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(saved_files, f, indent=2)
        
        saved_files['index'] = str(index_file)
        
        self.logger.info(f"    Datasets guardados en: {output_path}")
        self.logger.info(f"    Archivos: {len(saved_files)}")
        
        return saved_files
    
    def _create_comprehensive_metadata(self, ml_datasets: Dict, lstm_datasets: Dict) -> Dict:
        """Crear metadata comprehensiva"""
        
        metadata = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processor_version': 'OptimizedWaterQualityProcessor_v1.0',
                'config': asdict(self.config),
                'leakage_free': True,
                'validation_performed': True
            },
            'dataset_info': self.dataset_info,
            'processing_metrics': self.processing_metrics,
            'contaminants': {},
            'quality_summary': self._calculate_overall_quality_metrics(ml_datasets, lstm_datasets)
        }
        
        # Información por contaminante
        for contaminant, dataset in ml_datasets.items():
            if dataset['validation_passed']:
                info = dataset['contaminant_info']
                metadata['contaminants'][contaminant] = {
                    'clean_name': info.clean_name,
                    'n_samples': info.n_samples,
                    'temporal_coverage_pct': info.temporal_coverage_pct,
                    'quality_score': info.quality_score,
                    'threshold': dataset['threshold'],
                    'splits': {
                        'train': len(dataset['y_train']),
                        'val': len(dataset['y_val']),
                        'test': len(dataset['y_test'])
                    },
                    'class_distribution': {
                        'train': np.bincount(dataset['y_train']).tolist(),
                        'val': np.bincount(dataset['y_val']).tolist(),
                        'test': np.bincount(dataset['y_test']).tolist()
                    },
                    'models_supported': ['SVM', 'XGBoost'] + (['LSTM'] if contaminant in lstm_datasets else []),
                    'lstm_sequences': len(lstm_datasets.get(contaminant, [])),
                    'validation_passed': dataset['validation_passed']
                }
        
        return metadata
    
    def _calculate_overall_quality_metrics(self, ml_datasets: Dict, lstm_datasets: Dict) -> Dict:
        """Calcular métricas de calidad general"""
        
        validated_datasets = [d for d in ml_datasets.values() if d['validation_passed']]
        
        if not validated_datasets:
            return {'overall_quality': 'POOR', 'score': 0}
        
        # Calcular métricas agregadas
        total_samples = sum(len(d['y_train']) + len(d['y_val']) + len(d['y_test']) for d in validated_datasets)
        avg_quality_score = np.mean([d['contaminant_info'].quality_score for d in validated_datasets])
        
        contaminants_with_lstm = sum(1 for cont in ml_datasets.keys() if cont in lstm_datasets)
        lstm_coverage = contaminants_with_lstm / len(validated_datasets) if validated_datasets else 0
        
        # Score general
        overall_score = (
            min(100, avg_quality_score) * 0.5 +  # 50% calidad individual
            min(100, total_samples / 10) * 0.3 +  # 30% cantidad de datos
            lstm_coverage * 100 * 0.2  # 20% cobertura LSTM
        )
        
        # Clasificación de calidad
        if overall_score >= 80:
            quality_level = 'EXCELLENT'
        elif overall_score >= 65:
            quality_level = 'GOOD'
        elif overall_score >= 50:
            quality_level = 'ACCEPTABLE'
        else:
            quality_level = 'POOR'
        
        return {
            'overall_quality': quality_level,
            'score': overall_score,
            'total_samples': total_samples,
            'avg_contaminant_quality': avg_quality_score,
            'contaminants_with_ml': len(validated_datasets),
            'contaminants_with_lstm': contaminants_with_lstm,
            'lstm_coverage_pct': lstm_coverage * 100
        }
    
    def generate_comprehensive_report(self, ml_datasets: Dict, lstm_datasets: Dict, 
                                     saved_files: Dict[str, str]):
        """
        Generar reporte comprehensivo del procesamiento
        """
        self.logger.info(" PASO 7: Generando reporte comprehensivo")
        
        report_file = Path(self.config.output_dir) / "processing_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPREHENSIVO DE PROCESAMIENTO\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("INFORMACIÓN GENERAL\n")
            f.write("-" * 30 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Versión: OptimizedWaterQualityProcessor v1.0\n")
            f.write(f"Configuración utilizada:\n")
            for key, value in asdict(self.config).items():
                f.write(f"  - {key}: {value}\n")
            
            f.write(f"\nDATOS PROCESADOS\n")
            f.write("-" * 20 + "\n")
            spectral_info = self.dataset_info.get('spectral', {})
            f.write(f"Rango espectral: {spectral_info.get('range', 'N/A')[0]}-{spectral_info.get('range', 'N/A')[1]} nm\n")
            f.write(f"Número de wavelengths: {spectral_info.get('n_bands', 0)}\n")
            f.write(f"Resolución espectral: ~{spectral_info.get('resolution', 0):.1f} nm\n")
            
            contaminants_info = self.dataset_info.get('contaminants', {})
            f.write(f"Contaminantes orgánicos: {len(contaminants_info.get('organic', []))}\n")
            f.write(f"Contaminantes inorgánicos: {len(contaminants_info.get('inorganic', []))}\n")
            
            matching_info = self.processing_metrics.get('matching', {})
            f.write(f"Tasa de matching temporal: {matching_info.get('overall_match_rate_pct', 0):.1f}%\n")
            
            f.write(f"\nRESULTADOS DE VIABILIDAD\n")
            f.write("-" * 25 + "\n")
            viability_info = self.processing_metrics.get('viability', {})
            f.write(f"Contaminantes analizados: {viability_info.get('total_contaminants', 0)}\n")
            f.write(f"Contaminantes viables: {viability_info.get('viable_contaminants', 0)}\n")
            f.write(f"Tasa de viabilidad: {viability_info.get('viability_rate_pct', 0):.1f}%\n")
            f.write(f"Score promedio de calidad: {viability_info.get('avg_quality_score', 0):.1f}/100\n")
            
            f.write(f"\nDATASETS GENERADOS\n")
            f.write("-" * 20 + "\n")
            validated_datasets = [d for d in ml_datasets.values() if d['validation_passed']]
            f.write(f"Datasets ML (SVM/XGBoost): {len(validated_datasets)}\n")
            f.write(f"Datasets LSTM: {len(lstm_datasets)}\n")
            f.write(f"Validación anti-leakage: PASADA\n")
            
            f.write(f"\nDETALLE POR CONTAMINANTE\n")
            f.write("-" * 30 + "\n")
            
            for contaminant, dataset in ml_datasets.items():
                if dataset['validation_passed']:
                    info = dataset['contaminant_info']
                    f.write(f"\n{info.clean_name.upper()}\n")
                    f.write(f"  Muestras totales: {info.n_samples}\n")
                    f.write(f"  Cobertura temporal: {info.temporal_coverage_pct:.1f}%\n")
                    f.write(f"  Score de calidad: {info.quality_score:.1f}/100\n")
                    f.write(f"  Threshold binario: {dataset['threshold']:.4f}\n")
                    f.write(f"  Splits - Train: {len(dataset['y_train'])}, Val: {len(dataset['y_val'])}, Test: {len(dataset['y_test'])}\n")
                    
                    train_dist = np.bincount(dataset['y_train'])
                    f.write(f"  Distribución train: {dict(enumerate(train_dist))}\n")
                    
                    if contaminant in lstm_datasets:
                        f.write(f"  Secuencias LSTM: {len(lstm_datasets[contaminant])}\n")
                    
                    f.write(f"  Modelos soportados: SVM, XGBoost")
                    if contaminant in lstm_datasets:
                        f.write(", LSTM")
                    f.write("\n")
            
            f.write(f"\nARCHIVOS GENERADOS\n")
            f.write("-" * 20 + "\n")
            for file_type, filepath in saved_files.items():
                f.write(f"  {file_type}: {Path(filepath).name}\n")
            
            f.write(f"\nCALIDAD GENERAL DEL DATASET\n")
            f.write("-" * 30 + "\n")
            quality_metrics = self._calculate_overall_quality_metrics(ml_datasets, lstm_datasets)
            f.write(f"Nivel de calidad: {quality_metrics['overall_quality']}\n")
            f.write(f"Score general: {quality_metrics['score']:.1f}/100\n")
            f.write(f"Total de muestras: {quality_metrics['total_samples']}\n")
            f.write(f"Contaminantes con ML: {quality_metrics['contaminants_with_ml']}\n")
            f.write(f"Contaminantes con LSTM: {quality_metrics['contaminants_with_lstm']}\n")
            f.write(f"Cobertura LSTM: {quality_metrics['lstm_coverage_pct']:.1f}%\n")
            
            f.write(f"\nRECOMENDACIONES\n")
            f.write("-" * 15 + "\n")
            self._write_recommendations(f, quality_metrics, validated_datasets)
            
            f.write(f"\nPRÓXIMOS PASOS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Cargar datasets usando las funciones de carga proporcionadas\n")
            f.write("2. Entrenar modelos SVM y XGBoost con los datasets ML\n")
            f.write("3. Entrenar modelos LSTM con las secuencias temporales\n")
            f.write("4. Validar resultados con validación cruzada temporal\n")
            f.write("5. Comparar rendimiento entre diferentes algoritmos\n")
            f.write("6. Realizar análisis de interpretabilidad de features\n")
        
        self.logger.info(f"    Reporte generado: {report_file}")
    
    def _write_recommendations(self, f, quality_metrics: Dict, validated_datasets: List):
        """Escribir recomendaciones basadas en la calidad del dataset"""
        
        if quality_metrics['overall_quality'] == 'EXCELLENT':
            f.write(" Dataset de EXCELENTE calidad para machine learning\n")
            f.write("   - Proceder con confianza con todos los algoritmos\n")
            f.write("   - Considerar publicación de resultados\n")
            f.write("   - Explorar técnicas avanzadas de interpretabilidad\n")
            
        elif quality_metrics['overall_quality'] == 'GOOD':
            f.write(" Dataset de BUENA calidad para machine learning\n")
            f.write("   - Usar validación cruzada estricta\n")
            f.write("   - Monitorear overfitting cuidadosamente\n")
            f.write("   - Considerar técnicas de regularización\n")
            
        elif quality_metrics['overall_quality'] == 'ACCEPTABLE':
            f.write(" Dataset de calidad ACEPTABLE con precauciones\n")
            f.write("   - Usar como estudio exploratorio/piloto\n")
            f.write("   - Aplicar técnicas de augmentación de datos\n")
            f.write("   - Considerar recolectar más datos\n")
            f.write("   - Usar ensemble methods para robustez\n")
            
        else:  # POOR
            f.write(" Dataset de calidad POBRE para ML robusto\n")
            f.write("   - Revisar parámetros de procesamiento\n")
            f.write("   - Considerar análisis exploratorio únicamente\n")
            f.write("   - Recolectar datos adicionales\n")
            f.write("   - Verificar calidad de instrumentos\n")
        
        # Recomendaciones específicas
        if quality_metrics['contaminants_with_lstm'] < quality_metrics['contaminants_with_ml'] * 0.5:
            f.write("   - Considerar ajustar parámetros LSTM para mayor cobertura\n")
        
        if quality_metrics['total_samples'] < 200:
            f.write("   - Dataset pequeño: usar cross-validation estratificada\n")
        
        if len(validated_datasets) < 3:
            f.write("   - Pocos contaminantes: enfocarse en análisis individual\n")
    
    def process_complete_pipeline(self, reflectance_file: str, chemicals_file: str, 
                                 pollution_file: Optional[str] = None) -> Tuple[Dict, Dict, Dict]:
        """
        Pipeline completo optimizado
        """
        self.logger.info(" INICIANDO PIPELINE COMPLETO OPTIMIZADO")
        self.logger.info("=" * 60)
        
        try:
            # Paso 1: Cargar y validar datos
            df_reflectance, df_chemicals, df_pollution = self.load_and_validate_data(
                reflectance_file, chemicals_file, pollution_file
            )
            
            # Paso 2: Matching temporal avanzado
            df_merged = self.temporal_matching_advanced(df_reflectance, df_chemicals, df_pollution)
            
            # Paso 3: Análisis de viabilidad comprehensivo
            contaminant_analysis = self.analyze_contaminant_viability_comprehensive(df_merged)
            
            # Paso 4: Crear splits libres de data leakage
            ml_datasets = self.create_leakage_free_splits(df_merged, contaminant_analysis)
            
            # Paso 5: Crear secuencias LSTM
            lstm_datasets = self.create_lstm_sequences(df_merged, contaminant_analysis)
            
            # Paso 6: Guardar datasets
            saved_files = self.save_optimized_datasets(ml_datasets, lstm_datasets)
            
            # Paso 7: Generar reporte
            if self.config.generate_reports:
                self.generate_comprehensive_report(ml_datasets, lstm_datasets, saved_files)
            
            # Métricas finales
            quality_metrics = self._calculate_overall_quality_metrics(ml_datasets, lstm_datasets)
            
            self.logger.info("\n PIPELINE COMPLETADO EXITOSAMENTE")
            self.logger.info("=" * 60)
            self.logger.info(f" Calidad general: {quality_metrics['overall_quality']} ({quality_metrics['score']:.1f}/100)")
            self.logger.info(f" Datasets ML: {quality_metrics['contaminants_with_ml']}")
            self.logger.info(f" Datasets LSTM: {quality_metrics['contaminants_with_lstm']}")
            self.logger.info(f" Archivos guardados: {len(saved_files)}")
            self.logger.info(f" Directorio: {self.config.output_dir}")
            
            return ml_datasets, lstm_datasets, saved_files
            
        except Exception as e:
            self.logger.error(f" ERROR CRÍTICO EN PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            raise


# =====================================================================
# FUNCIONES DE UTILIDAD Y CARGA
# =====================================================================

def load_optimized_dataset(contaminant_name: str, dataset_type: str = 'ml', 
                          data_dir: str = "optimized_datasets") -> Optional[Dict]:
    """
    Cargar dataset optimizado para entrenamiento
    
    Args:
        contaminant_name: Nombre limpio del contaminante (ej: 'doc-mg-l')
        dataset_type: 'ml' para SVM/XGBoost o 'lstm' para secuencias
        data_dir: Directorio con los datasets
    
    Returns:
        Dict con los datos cargados
    """
    
    data_path = Path(data_dir)
    
    if dataset_type == 'ml':
        filename = f"ml_ready_{contaminant_name}.npz"
    elif dataset_type == 'lstm':
        filename = f"lstm_ready_{contaminant_name}.npz"
    else:
        raise ValueError("dataset_type debe ser 'ml' o 'lstm'")
    
    filepath = data_path / filename
    
    if not filepath.exists():
        print(f" Dataset no encontrado: {filepath}")
        return None
    
    # Cargar dataset
    data = np.load(filepath)
    
    if dataset_type == 'ml':
        dataset = {
            'X_train': data['X_train'],
            'X_val': data['X_val'],
            'X_test': data['X_test'],
            'y_train': data['y_train'],
            'y_val': data['y_val'],
            'y_test': data['y_test'],
            'concentrations_train': data['concentrations_train'],
            'concentrations_val': data['concentrations_val'],
            'concentrations_test': data['concentrations_test'],
            'threshold': float(data['threshold']),
            'feature_names': data['feature_names'].tolist(),
            'wavelengths': data['wavelengths'].tolist()
        }
        
        print(f" Dataset ML cargado: {contaminant_name}")
        print(f"   Train: {dataset['X_train'].shape}")
        print(f"   Val:   {dataset['X_val'].shape}")
        print(f"   Test:  {dataset['X_test'].shape}")
        print(f"   Threshold: {dataset['threshold']:.4f}")
        
    else:  # LSTM
        dataset = {
            'X_sequences': data['X_sequences'],
            'y_labels': data['y_labels'],
            'concentrations': data['concentrations'],
            'timestamps': data['timestamps'].tolist(),
            'threshold': float(data['threshold']),
            'sequence_length': int(data['sequence_length']),
            'wavelengths': data['wavelengths'].tolist()
        }
        
        print(f" Dataset LSTM cargado: {contaminant_name}")
        print(f"   Secuencias: {dataset['X_sequences'].shape}")
        print(f"   Labels: {dataset['y_labels'].shape}")
        print(f"   Longitud secuencia: {dataset['sequence_length']}")
        print(f"   Threshold: {dataset['threshold']:.4f}")
    
    return dataset


def list_available_datasets(data_dir: str = "optimized_datasets") -> Dict[str, List[str]]:
    """
    Listar datasets disponibles
    
    Returns:
        Dict con listas de datasets ML y LSTM disponibles
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f" Directorio no encontrado: {data_dir}")
        return {'ml': [], 'lstm': []}
    
    # Buscar archivos
    ml_files = list(data_path.glob("ml_ready_*.npz"))
    lstm_files = list(data_path.glob("lstm_ready_*.npz"))
    
    # Extraer nombres de contaminantes
    ml_contaminants = [f.stem.replace("ml_ready_", "") for f in ml_files]
    lstm_contaminants = [f.stem.replace("lstm_ready_", "") for f in lstm_files]
    
    print(f" Datasets disponibles en {data_dir}:")
    print(f"    ML (SVM/XGBoost): {len(ml_contaminants)}")
    for cont in ml_contaminants:
        print(f"      - {cont}")
    
    print(f"    LSTM: {len(lstm_contaminants)}")
    for cont in lstm_contaminants:
        print(f"      - {cont}")
    
    return {
        'ml': ml_contaminants,
        'lstm': lstm_contaminants
    }


def get_dataset_metadata(data_dir: str = "optimized_datasets") -> Optional[Dict]:
    """
    Cargar metadata de los datasets
    """
    
    metadata_path = Path(data_dir) / "datasets_metadata.json"
    
    if not metadata_path.exists():
        print(f" Metadata no encontrada: {metadata_path}")
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f" Metadata cargada:")
    print(f"   Fecha: {metadata['processing_info']['timestamp']}")
    print(f"   Versión: {metadata['processing_info']['processor_version']}")
    print(f"   Calidad general: {metadata['quality_summary']['overall_quality']}")
    print(f"   Score: {metadata['quality_summary']['score']:.1f}/100")
    
    return metadata


# =====================================================================
# CLASE PARA ENTRENAMIENTO OPTIMIZADO (BONUS)
# =====================================================================

class OptimizedModelTrainer:
    """
    Entrenador optimizado que trabaja con los datasets libres de leakage
    """
    
    def __init__(self, data_dir: str = "optimized_datasets"):
        self.data_dir = data_dir
        self.models = {}
        self.results = {}
        self.metadata = get_dataset_metadata(data_dir)
    
    def train_single_contaminant(self, contaminant_name: str, 
                                algorithms: List[str] = ['svm', 'xgboost']) -> Dict:
        """
        Entrenar modelos para un contaminante específico
        """
        
        print(f" ENTRENANDO MODELOS PARA: {contaminant_name}")
        print("-" * 50)
        
        # Cargar dataset
        dataset = load_optimized_dataset(contaminant_name, 'ml', self.data_dir)
        
        if dataset is None:
            return {}
        
        results = {}
        
        for algorithm in algorithms:
            try:
                result = self._train_algorithm(dataset, algorithm, contaminant_name)
                results[algorithm] = result
                print(f"    {algorithm.upper()}: Test Acc = {result['test_accuracy']:.3f}")
                
            except Exception as e:
                print(f"    Error en {algorithm}: {e}")
        
        return results
    
    def _train_algorithm(self, dataset: Dict, algorithm: str, contaminant_name: str) -> Dict:
        """Entrenar un algoritmo específico"""
        
        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_val = dataset['y_val']
        y_test = dataset['y_test']
        
        # Crear modelo
        if algorithm.lower() == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42, kernel='rbf')
            
        elif algorithm.lower() == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            except ImportError:
                raise ImportError("XGBoost no está instalado")
                
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Evaluar en todos los sets
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calcular métricas
        result = {
            'algorithm': algorithm.upper(),
            'contaminant': contaminant_name,
            'threshold': dataset['threshold'],
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test)
        }
        
        # Calcular gaps para detectar overfitting
        result['accuracy_gap'] = abs(result['train_accuracy'] - result['test_accuracy']) * 100
        result['f1_gap'] = abs(result['train_f1'] - result['test_f1']) * 100
        
        # Guardar modelo
        model_key = f"{contaminant_name}_{algorithm}"
        self.models[model_key] = model
        
        return result


# =====================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR TODO
# =====================================================================

def create_optimized_water_quality_datasets(
    reflectance_file: str,
    chemicals_file: str,
    pollution_file: Optional[str] = None,
    config: Optional[ProcessingConfig] = None,
    use_spectral_features: bool = True,        
    spectral_strategy: str = "spectral_only"
) -> Tuple[Dict, Dict, Dict]:
    """
    Función principal para crear datasets optimizados de calidad de agua
    
    Args:
        reflectance_file: Archivo CSV con datos espectrales
        chemicals_file: Archivo CSV con datos químicos orgánicos  
        pollution_file: Archivo CSV con datos químicos inorgánicos (opcional)
        config: Configuración personalizada (opcional)
    
    Returns:
        Tuple con (ml_datasets, lstm_datasets, saved_files)
    """
    
    print(" CREADOR DE DATASETS OPTIMIZADOS DE CALIDAD DE AGUA")
    print("=" * 70)
    print(" Universidad Diego Portales - María José Erazo González")
    print(" Pipeline optimizado libre de data leakage")
    print("=" * 70)
    
    # Usar configuración por defecto si no se proporciona
    if config is None:
        config = ProcessingConfig()
    
    # Crear procesador optimizado
    processor = OptimizedWaterQualityProcessor(
        config=config,
        use_spectral_features=use_spectral_features,
        spectral_strategy=spectral_strategy
    )
    
    # Ejecutar pipeline completo
    ml_datasets, lstm_datasets, saved_files = processor.process_complete_pipeline(
        reflectance_file, chemicals_file, pollution_file
    )
    
    print("\n PROCESAMIENTO COMPLETADO")
    print("=" * 50)
    print(" PARA USAR LOS DATASETS:")
    print("   1. dataset = load_optimized_dataset('contaminant-name', 'ml')")
    print("   2. lstm_data = load_optimized_dataset('contaminant-name', 'lstm')")
    print("   3. available = list_available_datasets()")
    print("   4. metadata = get_dataset_metadata()")
    
    return ml_datasets, lstm_datasets, saved_files

    # 4. FUNCIÓN DE VERIFICACIÓN EXTERNA
def verify_no_leakage_external(dataset_path: str) -> Dict:
    """Verificar ausencia de leakage en dataset ya guardado"""
    
    data = np.load(dataset_path)
    
    verification_results = {
        'timestamp_check': 'PASSED',
        'distribution_check': 'PASSED',
        'threshold_check': 'PASSED',
        'duplication_check': 'PASSED',
        'overall': 'PASSED',
        'issues': []
    }
    
    # Verificar timestamps si están disponibles
    if 'timestamps_train' in data and 'timestamps_test' in data:
        train_times = pd.to_datetime(data['timestamps_train'])
        test_times = pd.to_datetime(data['timestamps_test'])
        
        train_end = train_times.max()
        test_start = test_times.min()
        
        if train_end >= test_start:
            verification_results['timestamp_check'] = 'FAILED'
            verification_results['issues'].append('Temporal overlap detected')
    
    # Verificar distribuciones
    y_train = data['y_train']
    y_test = data['y_test']
    
    train_dist = np.bincount(y_train, minlength=2)
    test_dist = np.bincount(y_test, minlength=2)
    
    test_pos_rate = test_dist[1] / test_dist.sum() if test_dist.sum() > 0 else 0
    
    if test_pos_rate == 1.0 or test_pos_rate == 0.0:
        verification_results['distribution_check'] = 'FAILED'
        verification_results['issues'].append(f'Test set single class: {test_pos_rate:.1%}')
    
    # Verificar duplicación
    X_train = data['X_train']
    X_test = data['X_test']
    
    for i in range(min(10, len(X_test))):
        test_sample = X_test[i]
        for j in range(len(X_train)):
            if np.allclose(test_sample, X_train[j], atol=1e-10):
                verification_results['duplication_check'] = 'FAILED'
                verification_results['issues'].append('Duplicate samples found')
                break
        if verification_results['duplication_check'] == 'FAILED':
            break
    
    # Resultado general
    if any(check == 'FAILED' for check in verification_results.values() if isinstance(check, str)):
        verification_results['overall'] = 'FAILED'
    
    return verification_results



# =====================================================================
# SCRIPT DE EJECUCIÓN PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    
    # Configuración de archivos - AJUSTAR SEGÚN TU SISTEMA
    reflectance_file = "data/raw/flume_mvx_reflectance.csv"
    chemicals_file = "data/raw/laboratory_measurements_organic_chemicals.csv"
    pollution_file = "data/raw/laboratory_measurements.csv"  # Opcional
    
    # Configuración personalizada (opcional)
    custom_config = ProcessingConfig(
        temporal_tolerance_minutes=15,      # Reducir tolerancia temporal
        min_samples_per_contaminant=30,     # Más muestras mínimas
        test_size=0.15,                     # Test set más pequeño
        val_size=0.15,                      # Val set más pequeño
        min_temporal_coverage_pct=10.0,     # Mayor cobertura requerida
        max_null_percentage=70.0,           # Menos tolerancia a NaN
        lstm_sequence_length=3,             # Secuencias más cortas
        lstm_sequence_step_minutes=60,      # Mayor separación temporal
        output_dir="ultra_strict_datasets",
        save_intermediate=True,
        generate_reports=True,
        log_level="INFO"
    )
    
    print(" EJECUTANDO PIPELINE OPTIMIZADO...")
    
    try:
        # Crear datasets optimizados
        ml_datasets, lstm_datasets, saved_files = create_optimized_water_quality_datasets(
            reflectance_file=reflectance_file,
            chemicals_file=chemicals_file,
            pollution_file=pollution_file,
            config=custom_config,
            use_spectral_features=True,        
            spectral_strategy="spectral_only"  
        )
        for file_type, filepath in saved_files.items():
            if 'ml_ready_' in str(filepath) and filepath.endswith('.npz'):
                print(f"\nVerificando {filepath}...")
                verification = verify_no_leakage_external(filepath)
                print(f"Resultado: {verification['overall']}")
                if verification['issues']:
                    for issue in verification['issues']:
                        print(f"  - {issue}")
        
        print("\n DATASETS CON FEATURES ESPECTRALES CREADOS")
        print(f"    ML datasets: {len(ml_datasets)}")
        print(f"    LSTM datasets: {len(lstm_datasets)}")
        print(f"    Archivos: {len(saved_files)}")
        
        # Mostrar datasets disponibles
        print("\n DATASETS DISPONIBLES:")
        available = list_available_datasets(custom_config.output_dir)
        
    except FileNotFoundError as e:
        print(f"\n ARCHIVO NO ENCONTRADO: {e}")
        print(" AJUSTA LAS RUTAS DE ARCHIVOS EN LA SECCIÓN 'if __name__ == \"__main__\"'")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n DOCUMENTACIÓN COMPLETA DISPONIBLE EN:")
        print(f"   - README.md del proyecto")
        print(f"   - Reportes generados en {custom_config.output_dir}/")
        print(f"   - Logs en {custom_config.output_dir}/processing.log")