import pandas as pd
import numpy as np

def debug_data_structure():
    """Script para analizar la estructura de tus archivos de datos"""
    
    print(" ANALIZANDO ESTRUCTURA DE DATOS")
    print("=" * 50)
    
    # Rutas de archivos - ajusta según tu estructura
    files_to_check = {
        'reflectance': "data/raw/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv",
        'chemicals': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv",
        'pollution': "data/raw/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv"
    }
    
    for file_type, file_path in files_to_check.items():
        try:
            print(f"\n ARCHIVO: {file_type}")
            print(f"Ruta: {file_path}")
            
            # Cargar primeras filas
            df = pd.read_csv(file_path, nrows=5)
            
            print(f"Forma: {df.shape}")
            print(f"Columnas totales: {len(df.columns)}")
            
            print(f"\nPrimeras 10 columnas:")
            for i, col in enumerate(df.columns[:10]):
                print(f"  {i+1:2d}. {col} ({df[col].dtype})")
            
            if len(df.columns) > 10:
                print(f"  ... y {len(df.columns) - 10} columnas más")
            
            # Análisis específico para reflectance
            if file_type == 'reflectance':
                print(f"\n ANÁLISIS DETALLADO DE REFLECTANCE:")
                
                # Buscar columnas que contengan 'reflectance'
                reflectance_cols = [col for col in df.columns if 'reflectance' in col.lower()]
                print(f"Columnas con 'reflectance': {len(reflectance_cols)}")
                if reflectance_cols:
                    print(f"  Ejemplos: {reflectance_cols[:5]}")
                
                # Buscar columnas numéricas
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Columnas numéricas: {len(numeric_cols)}")
                if len(numeric_cols) > 0:
                    print(f"  Ejemplos: {list(numeric_cols[:10])}")
                
                # Buscar columnas que parezcan wavelengths
                potential_wavelengths = []
                for col in df.columns:
                    try:
                        # Intentar extraer número del nombre
                        import re
                        numbers = re.findall(r'\d+\.?\d*', str(col))
                        for num_str in numbers:
                            wl = float(num_str)
                            if 200 <= wl <= 2500:  # Rango válido
                                potential_wavelengths.append((col, wl))
                                break
                    except:
                        continue
                
                print(f"Posibles wavelengths detectadas: {len(potential_wavelengths)}")
                if potential_wavelengths:
                    print(f"  Primeras 5: {potential_wavelengths[:5]}")
                
                # Mostrar sample de datos
                print(f"\nSample de datos (primeras 3 filas, primeras 5 columnas):")
                print(df.iloc[:3, :5].to_string())
            
            # Análisis para químicos
            if 'chemical' in file_type or 'pollution' in file_type:
                print(f"\n ANÁLISIS DE CONTAMINANTES:")
                
                lab_cols = [col for col in df.columns if col.startswith('lab_')]
                print(f"Columnas 'lab_': {len(lab_cols)}")
                if lab_cols:
                    print(f"  Ejemplos: {lab_cols[:5]}")
                
                # Buscar columna timestamp
                timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
                print(f"Columnas timestamp: {timestamp_cols}")
        
        except FileNotFoundError:
            print(f" Archivo no encontrado: {file_path}")
        except Exception as e:
            print(f" Error leyendo {file_path}: {e}")
    
    print(f"\n ANÁLISIS COMPLETADO")
    print(f"\nSugerencias para arreglar el script:")
    print(f"1. Actualiza las rutas de archivos si son diferentes")
    print(f"2. Revisa los nombres de columnas espectrales detectadas")
    print(f"3. Si no hay columnas 'reflectance', el script las detectará automáticamente")

if __name__ == "__main__":
    debug_data_structure()