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


class FixedMLDatasetGenerator:
    """
    Generador de datasets ML corregido para tu estructura específica
    """
    
    def __init__(self, output_dir: str = "processed_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Variables internas
        self.wavelengths = None
        self.spectral_columns = None
        
        self.logger.info("FixedMLDatasetGenerator inicializado")
    
    def load_and_validate_data(self, reflectance_file: str, chemicals_file: str, pollution_file: str = None):
        """Cargar y validar datos con detección flexible"""
        
        self.logger.info(" CARGANDO Y VALIDANDO DATOS")
        self.logger.info("=" * 50)
        
        # Cargar archivos
        df_reflectance = pd.read_csv(reflectance_file)
        df_chemicals = pd.read_csv(chemicals_file)
        df_pollution = pd.read_csv(pollution_file) if pollution_file else None
        
        self.logger.info(f" Reflectancia cargada: {df_reflectance.shape}")
        self.logger.info(f" Químicos orgánicos: {df_chemicals.shape}")
        if df_pollution is not None:
            self.logger.info(f" Químicos inorgánicos: {df_pollution.shape}")
        
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
        """Detectar columnas espectrales de manera robusta"""
        
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
                        wavelengths.append(400 + len(wavelengths) * 2)  # Fallback
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
        """Limpiar datos químicos"""
        
        # Encontrar columnas de contaminantes
        contaminant_cols = [col for col in df.columns if col.startswith('lab_')]
        
        for col in contaminant_cols:
            if df[col].dtype == 'object':
                # Reemplazar <LOQ con NaN
                df[col] = df[col].replace('<LOQ', np.nan)
                df[col] = df[col].replace('<loq', np.nan)
                df[col] = df[col].replace('< LOQ', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def temporal_matching(self, df_reflectance, df_chemicals, df_pollution=None, tolerance_minutes=30):
        """Matching temporal simple pero efectivo"""
        
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
                        # Solo agregar columnas lab_ que no sean de método
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
    
    def create_contaminant_datasets(self, df_matched):
        """Crear datasets por contaminante"""
        
        self.logger.info(" CREANDO DATASETS POR CONTAMINANTE")
        
        # Encontrar contaminantes
        contaminant_cols = [col for col in df_matched.columns 
                          if col.startswith('lab_') and not col.endswith('_method')]
        
        datasets = {}
        
        for cont_col in contaminant_cols:
            contaminant_name = cont_col.replace('lab_', '').replace('_ng_l', '').replace('_mg_l', '').replace('_ntu', '')
            
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
            
            # Splits para ML
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 total
            )
            
            # Dataset para algoritmos clásicos
            classical_dataset = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler,
                'feature_names': [f'wl_{wl:.1f}' for wl in self.wavelengths],
                'wavelengths': self.wavelengths
            }
            
            # Dataset para LSTM (reshape)
            lstm_dataset = {
                'X_train': X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
                'X_val': X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
                'X_test': X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'sequence_length': X_train.shape[1],
                'wavelengths': self.wavelengths
            }
            
            datasets[contaminant_name] = {
                'classical': classical_dataset,
                'lstm': lstm_dataset,
                'metadata': {
                    'n_samples': len(valid_data),
                    'concentration_stats': {
                        'mean': float(np.mean(y)),
                        'std': float(np.std(y)),
                        'min': float(np.min(y)),
                        'max': float(np.max(y))
                    }
                }
            }
            
            self.logger.info(f" {contaminant_name}: {len(valid_data)} muestras")
        
        return datasets
    
    def save_datasets(self, datasets):
        """Guardar todos los datasets"""
        
        self.logger.info(" GUARDANDO DATASETS")
        
        saved_files = []
        
        for contaminant, data in datasets.items():
            # Guardar dataset clásico
            classical_file = self.output_dir / f"{contaminant}_classical_ml.npz"
            np.savez_compressed(classical_file, **data['classical'])
            saved_files.append(classical_file)
            
            # Guardar dataset LSTM
            lstm_file = self.output_dir / f"{contaminant}_lstm.npz"
            np.savez_compressed(lstm_file, **data['lstm'])
            saved_files.append(lstm_file)
            
            # Guardar metadatos
            import json
            metadata_file = self.output_dir / f"{contaminant}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data['metadata'], f, indent=2)
            saved_files.append(metadata_file)
        
        self.logger.info(f" {len(saved_files)} archivos guardados")
        return saved_files
    
    def generate_summary_report(self, datasets):
        """Generar reporte de resumen"""
        
        self.logger.info(" GENERANDO REPORTE")
        
        report = f"""# REPORTE DE DATASETS ML - DETECCIÓN DE CONTAMINANTES

### RESUMEN GENERAL
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Contaminantes procesados**: {len(datasets)}
- **Rango espectral**: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm
- **Resolución espectral**: ~{np.mean(np.diff(self.wavelengths)):.1f} nm
- **Bandas espectrales**: {len(self.wavelengths)}

### CONTAMINANTES PROCESADOS

"""
        
        for contaminant, data in datasets.items():
            metadata = data['metadata']
            stats = metadata['concentration_stats']
            
            report += f"""#### {contaminant.upper()}
- **Muestras**: {metadata['n_samples']}
- **Concentración media**: {stats['mean']:.3f} ± {stats['std']:.3f}
- **Rango**: {stats['min']:.3f} - {stats['max']:.3f}
- **Train/Val/Test**: {len(data['classical']['X_train'])}/{len(data['classical']['X_val'])}/{len(data['classical']['X_test'])}

"""
        
        report += f"""
### ARCHIVOS GENERADOS
- `*_classical_ml.npz`: Datasets para SVM, XGBoost
- `*_lstm.npz`: Datasets para LSTM
- `*_metadata.json`: Metadatos por contaminante
---
*Generado automáticamente: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_file = self.output_dir / "datasets_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f" Reporte guardado: {report_file}")
        
        return report
    
    def run_complete_pipeline(self, reflectance_file, chemicals_file, pollution_file=None):
        """Ejecutar pipeline completo"""
        
        self.logger.info(" EJECUTANDO PIPELINE COMPLETO")
        self.logger.info("=" * 60)
        
        try:
            # 1. Cargar y validar datos
            df_reflectance, df_chemicals, df_pollution = self.load_and_validate_data(
                reflectance_file, chemicals_file, pollution_file
            )
            
            # 2. Matching temporal
            df_matched = self.temporal_matching(df_reflectance, df_chemicals, df_pollution)
            
            # 3. Crear datasets por contaminante
            datasets = self.create_contaminant_datasets(df_matched)
            
            # 4. Guardar datasets
            saved_files = self.save_datasets(datasets)
            
            # 5. Generar reporte
            report = self.generate_summary_report(datasets)
            
            self.logger.info("=" * 60)
            self.logger.info(" PIPELINE COMPLETADO EXITOSAMENTE")
            self.logger.info(f" Contaminantes procesados: {len(datasets)}")
            self.logger.info(f" Archivos generados: {len(saved_files)}")
            self.logger.info("=" * 60)
            
            return {
                'datasets': datasets,
                'saved_files': saved_files,
                'report': report
            }
            
        except Exception as e:
            self.logger.error(f" Error en pipeline: {e}")
            raise


def main():
    """Función principal"""
    
    print(" PIPELINE FIJO PARA TU DATASET")
    print("Universidad Diego Portales - María José Erazo González")
    print("=" * 60)
    
    # Rutas corregidas
    reflectance_file = "data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv"
    chemicals_file = "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv"
    pollution_file = "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv"
    
    try:
        # Crear generator
        generator = FixedMLDatasetGenerator(output_dir="processed_datasets")
        
        # Ejecutar pipeline
        results = generator.run_complete_pipeline(
            reflectance_file=reflectance_file,
            chemicals_file=chemicals_file,
            pollution_file=pollution_file
        )
        
        print(f"\n RESULTADOS FINALES:")
        print(f"  Contaminantes procesados: {len(results['datasets'])}")
        print(f"  Archivos generados: {len(results['saved_files'])}")
        
        # Mostrar contaminantes
        print(f"\n CONTAMINANTES LISTOS PARA ML:")
        for i, (name, data) in enumerate(results['datasets'].items(), 1):
            n_samples = data['metadata']['n_samples']
            print(f"  {i:2d}. {name}: {n_samples} muestras")
        
        print(f"\n DATASETS LISTOS EN: processed_datasets/")
        print(f" PRÓXIMO PASO: python train.py")
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()