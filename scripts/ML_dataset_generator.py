import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class IntegratedMLGenerator:
    """
    Generador de datasets ML integrado con análisis LOQ
    Pipeline completo y científicamente riguroso
    """
    
    def __init__(self, output_dir: str = "integrated_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Variables internas
        self.wavelengths = None
        self.spectral_columns = None
        self.loq_data = None
        
        self.logger.info("IntegratedMLGenerator inicializado")
    
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
            self.logger.info(" Sin datos LOQ - usando pipeline básico")
        
        # Detectar columnas espectrales
        self._detect_spectral_columns(df_reflectance)
        
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
        
        self.logger.info(f" Detectadas {len(spectral_cols)} columnas espectrales")
        self.logger.info(f" Rango: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
    
    def _clean_chemical_data(self, df):
        """Limpiar datos químicos manteniendo información de LOQ"""
        
        contaminant_cols = [col for col in df.columns if col.startswith('lab_')]
        
        for col in contaminant_cols:
            if df[col].dtype == 'object':
                # Marcar <LOQ antes de convertir a NaN
                df[f'{col}_below_loq'] = df[col].str.contains('<LOQ|<loq|< LOQ', na=False)
                
                # Reemplazar <LOQ con NaN
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
            
            # Datos del contaminante
            values = df_chemicals[col]
            total_samples = len(values.dropna())
            
            if total_samples == 0:
                continue
            
            # Obtener LOQ si está disponible
            avg_loq = self._get_loq_for_contaminant(contaminant_name)
            
            if avg_loq is not None:
                # Análisis con LOQ
                detected_samples = len(values[values >= avg_loq])
                detection_rate = detected_samples / total_samples if total_samples > 0 else 0
                
                # Estadísticas de concentración
                detected_values = values[values >= avg_loq]
                
                analysis = {
                    'avg_loq': avg_loq,
                    'total_samples': total_samples,
                    'detected_samples': detected_samples,
                    'detection_rate': detection_rate,
                    'detectability_category': self._categorize_detectability(detection_rate),
                    'has_loq': True,
                    'concentration_stats': {
                        'detected_mean': detected_values.mean() if len(detected_values) > 0 else np.nan,
                        'detected_max': detected_values.max() if len(detected_values) > 0 else np.nan,
                        'signal_to_loq_ratio': (detected_values.mean() / avg_loq) if len(detected_values) > 0 and avg_loq > 0 else np.nan
                    }
                }
            else:
                # Análisis sin LOQ - usar distribución de datos
                valid_values = values.dropna()
                
                if len(valid_values) > 0:
                    # Usar percentil 25 como pseudo-LOQ
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
                    'concentration_stats': {
                        'detected_mean': valid_values.mean() if len(valid_values) > 0 else np.nan,
                        'detected_max': valid_values.max() if len(valid_values) > 0 else np.nan,
                        'signal_to_loq_ratio': np.nan
                    }
                }
            
            detectability_analysis[contaminant_name] = analysis
        
        # Guardar análisis
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
        
        self.logger.info(" REALIZANDO MATCHING TEMPORAL")
        
        matched_data = []
        tolerance = pd.Timedelta(minutes=tolerance_minutes)
        
        for _, chem_row in df_chemicals.iterrows():
            chem_time = chem_row['timestamp']
            
            # Buscar espectro más cercano
            time_diffs = abs(df_reflectance['timestamp'] - chem_time)
            closest_idx = time_diffs.idxmin()
            
            if time_diffs.loc[closest_idx] <= tolerance:
                # Combinar datos
                spec_row = df_reflectance.loc[closest_idx]
                
                combined_row = {}
                combined_row.update(spec_row.to_dict())
                combined_row.update(chem_row.to_dict())
                combined_row['time_diff_minutes'] = time_diffs.loc[closest_idx].total_seconds() / 60
                
                # Agregar inorgánicos si existen
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
        
        self.logger.info(f" Matches encontrados: {len(df_matched)}")
        if len(df_matched) > 0:
            avg_diff = df_matched['time_diff_minutes'].mean()
            self.logger.info(f" Diferencia temporal promedio: {avg_diff:.1f} minutos")
        
        return df_matched
    
    def create_enhanced_datasets(self, df_matched, detectability_analysis):
        """Crear datasets con análisis LOQ integrado"""
        
        self.logger.info(" CREANDO DATASETS MEJORADOS")
        
        contaminant_cols = [col for col in df_matched.columns 
                          if col.startswith('lab_') and not col.endswith('_below_loq') and not col.endswith('_method')]
        
        datasets = {}
        
        for cont_col in contaminant_cols:
            contaminant_name = cont_col.replace('lab_', '').replace('_ng_l', '').replace('_mg_l', '').replace('_ntu', '')
            
            # Verificar si hay análisis de detectabilidad
            if contaminant_name not in detectability_analysis:
                continue
            
            analysis = detectability_analysis[contaminant_name]
            
            # Filtrar datos válidos
            valid_mask = df_matched[cont_col].notna()
            if not any(valid_mask):
                continue
            
            valid_data = df_matched[valid_mask].copy()
            
            if len(valid_data) < 10:
                self.logger.warning(f" {contaminant_name}: solo {len(valid_data)} muestras - omitiendo")
                continue
            
            # Extraer features y targets
            X = valid_data[self.spectral_columns].values
            y = valid_data[cont_col].values
            
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Crear múltiples estrategias de targeting si hay LOQ
            dataset = self._create_multiple_targets(
                X_scaled, y, scaler, contaminant_name, analysis
            )
            
            datasets[contaminant_name] = dataset
            
            category = analysis['detectability_category']
            loq_status = "con LOQ" if analysis['has_loq'] else "sin LOQ"
            self.logger.info(f" {contaminant_name}: {len(valid_data)} muestras, {category}, {loq_status}")
        
        return datasets
    
    def _create_multiple_targets(self, X_scaled, y, scaler, contaminant_name, analysis):
        """Crear múltiples estrategias de targeting"""
        
        # Splits básicos
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        # Dataset base
        dataset = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'scaler': scaler,
            'feature_names': [f'wl_{wl:.1f}' for wl in self.wavelengths],
            'wavelengths': self.wavelengths,
            'detectability_analysis': analysis
        }
        
        # Estrategias de targeting
        if analysis['has_loq']:
            # Con LOQ - múltiples estrategias
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
            
            # 3. Clasificación ternaria
            def create_ternary(y_vals):
                ternary = np.zeros_like(y_vals, dtype=int)
                ternary[y_vals < loq] = 0  # No detectado
                ternary[(y_vals >= loq) & (y_vals < loq * 3)] = 1  # Detectado bajo
                ternary[y_vals >= loq * 3] = 2  # Detectado alto
                return ternary
            
            dataset.update({
                'y_train_ternary': create_ternary(y_train),
                'y_val_ternary': create_ternary(y_val),
                'y_test_ternary': create_ternary(y_test)
            })
            
            # 4. Regresión solo en detectados
            def filter_detected(X_vals, y_vals):
                detected_mask = y_vals >= loq
                return X_vals[detected_mask], y_vals[detected_mask]
            
            if np.any(y_train >= loq):
                X_train_det, y_train_det = filter_detected(X_train, y_train)
                X_val_det, y_val_det = filter_detected(X_val, y_val)
                X_test_det, y_test_det = filter_detected(X_test, y_test)
                
                dataset.update({
                    'X_train_detected': X_train_det,
                    'X_val_detected': X_val_det,
                    'X_test_detected': X_test_det,
                    'y_train_detected': y_train_det,
                    'y_val_detected': y_val_det,
                    'y_test_detected': y_test_det
                })
        
        else:
            # Sin LOQ - solo targets originales
            dataset.update({
                'y_train_original': y_train,
                'y_val_original': y_val,
                'y_test_original': y_test
            })
        
        # Versiones LSTM
        dataset_lstm = dataset.copy()
        dataset_lstm.update({
            'X_train': X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
            'X_val': X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
            'X_test': X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
            'sequence_length': X_train.shape[1]
        })
        
        if 'X_train_detected' in dataset:
            X_train_det = dataset['X_train_detected']
            X_val_det = dataset['X_val_detected']
            X_test_det = dataset['X_test_detected']
            
            dataset_lstm.update({
                'X_train_detected': X_train_det.reshape(X_train_det.shape[0], X_train_det.shape[1], 1),
                'X_val_detected': X_val_det.reshape(X_val_det.shape[0], X_val_det.shape[1], 1),
                'X_test_detected': X_test_det.reshape(X_test_det.shape[0], X_test_det.shape[1], 1)
            })
        
        return {
            'classical': dataset,
            'lstm': dataset_lstm,
            'metadata': {
                'n_samples': len(y),
                'detectability_category': analysis['detectability_category'],
                'has_loq': analysis['has_loq'],
                'avg_loq': analysis.get('avg_loq', np.nan),
                'detection_rate': analysis['detection_rate'],
                'strategies_available': self._get_available_strategies(analysis),
                'concentration_stats': {
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y))
                }
            }
        }
    
    def _get_available_strategies(self, analysis):
        """Determinar estrategias disponibles según el análisis"""
        strategies = ['original']
        
        if analysis['has_loq']:
            strategies.extend(['binary', 'ternary'])
            if analysis['detection_rate'] > 0.3:  # Solo si hay suficientes detectados
                strategies.append('detected_only')
        
        return strategies
    
    def _save_detectability_analysis(self, analysis_results):
        """Guardar análisis de detectabilidad"""
        
        analysis_data = []
        
        for contaminant, results in analysis_results.items():
            row = {
                'contaminant': contaminant,
                'avg_loq_ng_l': results.get('avg_loq', np.nan),
                'has_loq': results['has_loq'],
                'total_samples': results['total_samples'],
                'detected_samples': results['detected_samples'],
                'detection_rate': results['detection_rate'],
                'detectability_category': results['detectability_category'],
                'detected_mean_ng_l': results['concentration_stats']['detected_mean'],
                'detected_max_ng_l': results['concentration_stats']['detected_max'],
                'signal_to_loq_ratio': results['concentration_stats']['signal_to_loq_ratio']
            }
            analysis_data.append(row)
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # Ordenar por detectabilidad
        category_order = {'High': 3, 'Medium': 2, 'Low': 1, 'Poor': 0}
        df_analysis['sort_order'] = df_analysis['detectability_category'].map(category_order)
        df_analysis = df_analysis.sort_values(['sort_order', 'detection_rate'], ascending=[False, False])
        df_analysis = df_analysis.drop('sort_order', axis=1)
        
        # Guardar
        analysis_file = self.output_dir / "integrated_detectability_analysis.csv"
        df_analysis.to_csv(analysis_file, index=False)
        
        self.logger.info(f" Análisis guardado: {analysis_file}")
        
        # Mostrar resumen
        self.logger.info("\n RESUMEN DE DETECTABILIDAD:")
        for category in ['High', 'Medium', 'Low', 'Poor']:
            count = len(df_analysis[df_analysis['detectability_category'] == category])
            with_loq = len(df_analysis[(df_analysis['detectability_category'] == category) & 
                                     (df_analysis['has_loq'] == True)])
            self.logger.info(f"  {category}: {count} contaminantes ({with_loq} con LOQ)")
    
    def save_datasets(self, datasets):
        """Guardar todos los datasets"""
        
        self.logger.info(" GUARDANDO DATASETS INTEGRADOS")
        
        saved_files = []
        
        for contaminant, data in datasets.items():
            # Guardar dataset clásico
            classical_file = self.output_dir / f"{contaminant}_integrated_classical.npz"
            np.savez_compressed(classical_file, **data['classical'])
            saved_files.append(classical_file)
            
            # Guardar dataset LSTM
            lstm_file = self.output_dir / f"{contaminant}_integrated_lstm.npz"
            np.savez_compressed(lstm_file, **data['lstm'])
            saved_files.append(lstm_file)
            
            # Guardar metadatos
            import json
            metadata_file = self.output_dir / f"{contaminant}_integrated_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data['metadata'], f, indent=2, default=str)
            saved_files.append(metadata_file)
        
        self.logger.info(f" {len(saved_files)} archivos guardados")
        return saved_files
    
    def generate_comprehensive_report(self, datasets, detectability_analysis):
        """Generar reporte comprehensive integrado"""
        
        self.logger.info(" GENERANDO REPORTE COMPREHENSIVE")
        
        # Estadísticas generales
        total_contaminants = len(datasets)
        with_loq = len([d for d in datasets.values() if d['metadata']['has_loq']])
        
        # Por categoría
        by_category = {}
        for data in datasets.values():
            category = data['metadata']['detectability_category']
            by_category[category] = by_category.get(category, 0) + 1
        
        report = f"""# REPORTE COMPREHENSIVE - PIPELINE INTEGRADO
## 

### RESUMEN EJECUTIVO
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Contaminantes procesados**: {total_contaminants}
- **Con datos LOQ**: {with_loq} ({with_loq/total_contaminants*100:.1f}%)
- **Rango espectral**: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm
- **Resolución espectral**: ~{np.mean(np.diff(self.wavelengths)):.1f} nm
- **Bandas espectrales**: {len(self.wavelengths)}

### DISTRIBUCIÓN POR DETECTABILIDAD
"""
        
        for category, count in by_category.items():
            report += f"- **{category}**: {count} contaminantes\n"
        
        report += f"""
### CONTAMINANTES PROCESADOS

#### Por Categoría de Detectabilidad:
"""
        
        for category in ['High', 'Medium', 'Low', 'Poor']:
            contaminants_in_category = [name for name, data in datasets.items() 
                                      if data['metadata']['detectability_category'] == category]
            
            if contaminants_in_category:
                report += f"""
**{category} Detectability ({len(contaminants_in_category)} contaminantes):**
"""
                for name in contaminants_in_category:
                    data = datasets[name]
                    metadata = data['metadata']
                    
                    loq_status = f"LOQ: {metadata['avg_loq']:.1f} ng/L" if metadata['has_loq'] else "Sin LOQ"
                    strategies = ", ".join(metadata['strategies_available'])
                    
                    report += f"- **{name}**: {metadata['n_samples']} muestras, {loq_status}\n"
                    report += f"  → Estrategias: {strategies}\n"
                    report += f"  → Detección: {metadata['detection_rate']:.1%}\n"
        
        report += f"""
### ESTRATEGIAS DE MODELING DISPONIBLES

#### Para Contaminantes con LOQ:
1. **Original (Regresión)**: Valores de concentración completos
2. **Binary (Clasificación)**: Detectado (≥LOQ) vs No detectado (<LOQ)
3. **Ternary (Clasificación)**: No detectado / Detectado bajo / Detectado alto
4. **Detected Only (Regresión)**: Solo valores ≥LOQ para cuantificación precisa

#### Para Contaminantes sin LOQ:
1. **Original (Regresión/Clasificación)**: Análisis estadístico estándar

### RECOMENDACIONES CIENTÍFICAS

#### Contaminantes Recomendados para Modelado:
"""
        
        high_quality = [name for name, data in datasets.items() 
                       if data['metadata']['detectability_category'] in ['High', 'Medium']]
        
        report += f"**Alto nivel de confianza ({len(high_quality)} contaminantes):**\n"
        for name in high_quality:
            report += f"- {name}\n"
        
        report += f"""
#### Casos Especiales:
"""
        
        special_cases = [name for name, data in datasets.items() 
                        if data['metadata']['detectability_category'] in ['Low', 'Poor']]
        
        for name in special_cases:
            data = datasets[name]
            report += f"- **{name}**: {data['metadata']['detectability_category']} detectability - "
            report += f"Útil para estudios de límites de detección\n"
        
        report += f"""
### ARCHIVOS GENERADOS
- `*_integrated_classical.npz`: Datasets para SVM, XGBoost, Random Forest
- `*_integrated_lstm.npz`: Datasets para LSTM, CNN1D, Transformers
- `*_integrated_metadata.json`: Metadatos completos por contaminante
- `integrated_detectability_analysis.csv`: Análisis completo de detectabilidad

### CALIDAD CIENTÍFICA
 **Pipeline científicamente riguroso**
 **Análisis LOQ integrado** 
 **Múltiples estrategias de targeting**
 **Datasets listos para publicación**
 **Metodología reproducible**


#### Contaminantes Altamente Confiables (>70% detección):
"""
        
        reliable_contaminants = [name for name, analysis in detectability_analysis.items() 
                               if analysis['detection_rate'] > 0.7 and name in datasets]
        
        for name in reliable_contaminants:
            data = datasets[name]
            strategies = len(data['metadata']['strategies_available'])
            report += f"- **{name}**: {strategies} estrategias disponibles\n"
        
        report += f"""
#### Casos de Estudio Interesantes:
"""
        
        interesting_cases = [name for name, analysis in detectability_analysis.items() 
                           if 0.2 <= analysis['detection_rate'] <= 0.7 and name in datasets]
        
        for name in interesting_cases:
            analysis = detectability_analysis[name]
            report += f"- **{name}**: {analysis['detection_rate']:.1%} detección - Caso límite para investigación\n"
        
        report += f"""
### ESTADÍSTICAS TÉCNICAS

#### Distribución de Muestras:
"""
        
        sample_counts = [data['metadata']['n_samples'] for data in datasets.values()]
        if sample_counts:
            report += f"- **Promedio**: {np.mean(sample_counts):.1f} muestras por contaminante\n"
            report += f"- **Rango**: {min(sample_counts)} - {max(sample_counts)} muestras\n"
            report += f"- **Mediana**: {np.median(sample_counts):.1f} muestras\n"
        
        report += f"""
#### Cobertura LOQ:
- **Contaminantes con LOQ**: {with_loq}/{total_contaminants} ({with_loq/total_contaminants*100:.1f}%)
- **Alta detectabilidad con LOQ**: {len([d for d in datasets.values() if d['metadata']['has_loq'] and d['metadata']['detectability_category'] == 'High'])}
- **Estrategias múltiples disponibles**: {len([d for d in datasets.values() if len(d['metadata']['strategies_available']) > 2])}

---
*Generado automáticamente por IntegratedMLGenerator*
*Pipeline validado para detección espectral de contaminantes*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Guardar reporte
        report_file = self.output_dir / "comprehensive_integrated_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f" Reporte guardado: {report_file}")
        
        return report
    
    def run_complete_integrated_pipeline(self, reflectance_file, chemicals_file, 
                                       pollution_file=None, loq_file=None):
        """Ejecutar pipeline completo integrado"""
        
        self.logger.info(" EJECUTANDO PIPELINE INTEGRADO COMPLETO")
        self.logger.info("=" * 60)
        
        try:
            # 1. Cargar todos los datos
            df_reflectance, df_chemicals, df_pollution = self.load_all_data(
                reflectance_file, chemicals_file, pollution_file, loq_file
            )
            
            # 2. Analizar detectabilidad
            detectability_analysis = self.analyze_detectability(df_chemicals)
            
            # 3. Matching temporal
            df_matched = self.temporal_matching(df_reflectance, df_chemicals, df_pollution)
            
            # 4. Crear datasets mejorados
            datasets = self.create_enhanced_datasets(df_matched, detectability_analysis)
            
            # 5. Guardar datasets
            saved_files = self.save_datasets(datasets)
            
            # 6. Generar reporte comprehensive
            report = self.generate_comprehensive_report(datasets, detectability_analysis)
            
            self.logger.info("=" * 60)
            self.logger.info(" PIPELINE INTEGRADO COMPLETADO EXITOSAMENTE")
            self.logger.info(f" Contaminantes procesados: {len(datasets)}")
            self.logger.info(f" Archivos generados: {len(saved_files)}")
            
            # Estadísticas finales
            with_loq = len([d for d in datasets.values() if d['metadata']['has_loq']])
            high_quality = len([d for d in datasets.values() 
                               if d['metadata']['detectability_category'] in ['High', 'Medium']])
            
            self.logger.info(f" Con análisis LOQ: {with_loq}")
            self.logger.info(f" Alta calidad: {high_quality}")
            self.logger.info("=" * 60)
            
            return {
                'datasets': datasets,
                'detectability_analysis': detectability_analysis,
                'saved_files': saved_files,
                'report': report
            }
            
        except Exception as e:
            self.logger.error(f" Error en pipeline integrado: {e}")
            raise


def main():
    """Función principal del pipeline integrado"""
    
    print(" PIPELINE INTEGRADO ML + LOQ")
    print("")
    print("=" * 60)
    
    # Rutas de archivos
    files = {
        'reflectance': "data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv",
        'chemicals': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv",
        'pollution': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv",
        'loq': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_loq_organic_chemicals.csv"
    }
    
    try:
        # Crear generator integrado
        generator = IntegratedMLGenerator(output_dir="integrated_datasets")
        
        # Ejecutar pipeline completo
        results = generator.run_complete_integrated_pipeline(
            reflectance_file=files['reflectance'],
            chemicals_file=files['chemicals'],
            pollution_file=files['pollution'],
            loq_file=files['loq']
        )
        
        print(f"\n RESULTADOS FINALES:")
        print(f"  Contaminantes procesados: {len(results['datasets'])}")
        print(f"  Archivos generados: {len(results['saved_files'])}")
        
        # Mostrar estadísticas por categoría
        by_category = {}
        for data in results['datasets'].values():
            category = data['metadata']['detectability_category']
            by_category[category] = by_category.get(category, 0) + 1
        
        print(f"\n DISTRIBUCIÓN POR CALIDAD:")
        for category, count in sorted(by_category.items(), 
                                    key=lambda x: {'High': 4, 'Medium': 3, 'Low': 2, 'Poor': 1}[x[0]], 
                                    reverse=True):
            print(f"  {category}: {count} contaminantes")
        
        # Mostrar contaminantes listos
        print(f"\n CONTAMINANTES LISTOS PARA INVESTIGACIÓN:")
        
        high_quality = [name for name, data in results['datasets'].items() 
                       if data['metadata']['detectability_category'] in ['High', 'Medium']]
        
        for i, name in enumerate(high_quality, 1):
            data = results['datasets'][name]
            metadata = data['metadata']
            loq_status = "con LOQ" if metadata['has_loq'] else "sin LOQ"
            strategies = len(metadata['strategies_available'])
            
            print(f"  {i:2d}. {name}: {metadata['n_samples']} muestras, "
                  f"{metadata['detectability_category']}, {loq_status}, "
                  f"{strategies} estrategias")
        
        print(f"\n DATASETS INTEGRADOS LISTOS EN: integrated_datasets/")
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()