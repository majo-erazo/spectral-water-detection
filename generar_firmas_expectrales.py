import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import json


class SpectralSignatureGenerator:
    """
    Generador de firmas espectrales usando archivos CSV de laboratorio
    """
    
    def __init__(self, reflectance_file, lab_inorganic_file, lab_organic_file, loq_file=None):
        """
        Inicializar el generador con las rutas de archivos
        
        Args:
            reflectance_file: Ruta al archivo CSV de datos espectrales
            lab_inorganic_file: Ruta al archivo CSV de contaminantes inorgánicos
            lab_organic_file: Ruta al archivo CSV de contaminantes orgánicos
            loq_file: Ruta al archivo CSV de límites LOQ (opcional)
        """
        self.reflectance_file = reflectance_file
        self.lab_inorganic_file = lab_inorganic_file
        self.lab_organic_file = lab_organic_file
        self.loq_file = loq_file
        
        # Datos cargados
        self.spectral_data = None
        self.inorganic_data = None
        self.organic_data = None
        self.loq_data = None
        
        # Configuración por defecto
        self.time_window_minutes = 30
        self.min_samples = 5
        self.save_dir = "firmas_espectrales_csv"
        
        # Crear directorio de salida
        os.makedirs(self.save_dir, exist_ok=True)
    
    def load_data(self):
        """Cargar todos los archivos CSV"""
        print("🔄 Cargando archivos CSV...")
        
        # Cargar datos espectrales
        print("  📊 Cargando datos espectrales...")
        self.spectral_data = pd.read_csv(self.reflectance_file)
        if 'timestamp_iso' in self.spectral_data.columns:
            self.spectral_data['timestamp_iso'] = pd.to_datetime(self.spectral_data['timestamp_iso'])
        
        # Filtrar solo datos válidos
        if 'valid_data' in self.spectral_data.columns:
            self.spectral_data = self.spectral_data[self.spectral_data['valid_data'] == 1]
        
        print(f"    ✅ {len(self.spectral_data)} muestras espectrales válidas")
        
        # Cargar datos inorgánicos
        print("  🧪 Cargando contaminantes inorgánicos...")
        self.inorganic_data = pd.read_csv(self.lab_inorganic_file)
        if 'timestamp_iso' in self.inorganic_data.columns:
            self.inorganic_data['timestamp_iso'] = pd.to_datetime(self.inorganic_data['timestamp_iso'])
        
        print(f"    ✅ {len(self.inorganic_data)} muestras inorgánicas")
        
        # Cargar datos orgánicos
        print("  🌿 Cargando contaminantes orgánicos...")
        self.organic_data = pd.read_csv(self.lab_organic_file, na_values="<LOQ")
        if 'timestamp_iso' in self.organic_data.columns:
            self.organic_data['timestamp_iso'] = pd.to_datetime(self.organic_data['timestamp_iso'])
        
        print(f"    ✅ {len(self.organic_data)} muestras orgánicas")
        
        # Cargar LOQ si está disponible
        if self.loq_file and os.path.exists(self.loq_file):
            print("  📏 Cargando límites LOQ...")
            self.loq_data = pd.read_csv(self.loq_file)
            print(f"    ✅ Límites LOQ cargados")
    
    def identify_viable_contaminants(self, dataset_type='both'):
        """
        Identificar contaminantes viables para análisis espectral
        
        Args:
            dataset_type: 'inorganic', 'organic', o 'both'
        
        Returns:
            Lista de contaminantes viables con estadísticas
        """
        viable_contaminants = []
        
        datasets_to_check = []
        if dataset_type in ['inorganic', 'both']:
            datasets_to_check.append(('inorganic', self.inorganic_data))
        if dataset_type in ['organic', 'both']:
            datasets_to_check.append(('organic', self.organic_data))
        
        for data_type, data in datasets_to_check:
            contaminant_cols = [col for col in data.columns if col.startswith('lab_') and col != 'lab_loq']
            
            for col in contaminant_cols:
                # Estadísticas básicas
                total_samples = len(data)
                non_null_samples = data[col].notna().sum()
                
                # Para orgánicos, contar valores <LOQ si existen
                loq_count = 0
                if data[col].dtype == 'object':
                    loq_count = (data[col].astype(str).str.contains('<LOQ', na=False)).sum()
                
                # Muestras realmente cuantificables
                quantifiable_samples = non_null_samples - loq_count
                quantifiable_percentage = (quantifiable_samples / total_samples) * 100
                
                # Criterios de viabilidad
                is_viable = (
                    quantifiable_samples >= self.min_samples and 
                    quantifiable_percentage >= 30  # Al menos 30% cuantificable
                )
                
                # Calcular rango de valores si es numérico
                value_range = None
                if quantifiable_samples > 0:
                    numeric_values = pd.to_numeric(data[col], errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        value_range = (numeric_values.min(), numeric_values.max())
                
                contaminant_info = {
                    'name': col,
                    'display_name': col.replace('lab_', '').replace('_', ' ').title(),
                    'type': data_type,
                    'total_samples': total_samples,
                    'quantifiable_samples': quantifiable_samples,
                    'quantifiable_percentage': quantifiable_percentage,
                    'loq_count': loq_count,
                    'value_range': value_range,
                    'is_viable': is_viable,
                    'priority': self._calculate_priority(col, quantifiable_samples, quantifiable_percentage)
                }
                
                viable_contaminants.append(contaminant_info)
        
        # Ordenar por prioridad (mayor prioridad primero)
        viable_contaminants.sort(key=lambda x: x['priority'], reverse=True)
        
        return viable_contaminants
    
    def _calculate_priority(self, contaminant_name, samples, percentage):
        """Calcular prioridad basada en viabilidad y importancia"""
        base_score = samples * (percentage / 100)
        
        # Bonificaciones por tipo de contaminante
        high_priority_keywords = ['turbidity', 'tss', 'doc', 'toc', 'nh4', 'caffeine', 'acesulfame']
        medium_priority_keywords = ['po4', 'so4', 'diclofenac', 'candesartan']
        
        name_lower = contaminant_name.lower()
        
        if any(keyword in name_lower for keyword in high_priority_keywords):
            return base_score * 1.5
        elif any(keyword in name_lower for keyword in medium_priority_keywords):
            return base_score * 1.2
        else:
            return base_score
    
    def find_closest_spectrum(self, timestamp, max_minutes=None):
        """Encontrar el espectro más cercano a un timestamp dado"""
        if max_minutes is None:
            max_minutes = self.time_window_minutes
            
        window_start = timestamp - timedelta(minutes=max_minutes)
        window_end = timestamp + timedelta(minutes=max_minutes)
        
        window_spectra = self.spectral_data[
            (self.spectral_data['timestamp_iso'] >= window_start) & 
            (self.spectral_data['timestamp_iso'] <= window_end)
        ].copy()
        
        if len(window_spectra) == 0:
            return None
        
        window_spectra['time_diff'] = abs(window_spectra['timestamp_iso'] - timestamp)
        return window_spectra.loc[window_spectra['time_diff'].idxmin()]
    
    def generate_signature_for_contaminant(self, contaminant_info):
        """
        Generar firma espectral para un contaminante específico
        
        Args:
            contaminant_info: Diccionario con información del contaminante
        
        Returns:
            Diccionario con resultados del análisis o None si falla
        """
        contaminant = contaminant_info['name']
        data_type = contaminant_info['type']
        
        print(f"\n🔬 Analizando: {contaminant_info['display_name']} ({data_type})")
        
        # Seleccionar dataset apropiado
        if data_type == 'inorganic':
            lab_data = self.inorganic_data
        else:
            lab_data = self.organic_data
        
        # Verificar que el contaminante existe
        if contaminant not in lab_data.columns:
            print(f"   ❌ Contaminante {contaminant} no encontrado")
            return None
        
        # Limpiar datos: convertir a numérico y eliminar <LOQ
        lab_values = lab_data[contaminant].copy()
        
        # Si es object (string), intentar convertir eliminando <LOQ
        if lab_values.dtype == 'object':
            # Filtrar solo valores que no son <LOQ
            numeric_mask = ~lab_values.astype(str).str.contains('<LOQ', na=False)
            lab_values = lab_values[numeric_mask]
            lab_data_filtered = lab_data[numeric_mask].copy()
        else:
            lab_data_filtered = lab_data.copy()
        
        # Convertir a numérico
        lab_data_filtered[contaminant] = pd.to_numeric(lab_data_filtered[contaminant], errors='coerce')
        
        # Eliminar valores nulos
        valid_samples = lab_data_filtered.dropna(subset=[contaminant])
        
        if len(valid_samples) < self.min_samples:
            print(f"   ⚠️ Insuficientes muestras válidas: {len(valid_samples)}")
            return None
        
        print(f"   📊 Muestras válidas: {len(valid_samples)}")
        
        # Definir umbrales (percentiles 25 y 75)
        high_threshold = valid_samples[contaminant].quantile(0.75)
        low_threshold = valid_samples[contaminant].quantile(0.25)
        
        print(f"   📈 Umbral alto: {high_threshold:.4f}")
        print(f"   📉 Umbral bajo: {low_threshold:.4f}")
        
        # Identificar muestras de alta y baja concentración
        high_samples = valid_samples[valid_samples[contaminant] > high_threshold]
        low_samples = valid_samples[valid_samples[contaminant] < low_threshold]
        
        print(f"   🔴 Muestras alta concentración: {len(high_samples)}")
        print(f"   🟢 Muestras baja concentración: {len(low_samples)}")
        
        # Encontrar espectros correspondientes
        high_spectra = []
        low_spectra = []
        
        print("   🔍 Buscando espectros correspondientes...")
        
        for _, row in high_samples.iterrows():
            spectrum = self.find_closest_spectrum(row['timestamp_iso'])
            if spectrum is not None:
                reflectance_cols = [col for col in spectrum.index 
                                  if 'reflectance' in col.lower() and not 'valid' in col]
                if reflectance_cols:
                    spectrum_values = {col: spectrum[col] for col in reflectance_cols}
                    spectrum_values['contaminant_value'] = row[contaminant]
                    spectrum_values['timestamp'] = row['timestamp_iso']
                    high_spectra.append(spectrum_values)
        
        for _, row in low_samples.iterrows():
            spectrum = self.find_closest_spectrum(row['timestamp_iso'])
            if spectrum is not None:
                reflectance_cols = [col for col in spectrum.index 
                                  if 'reflectance' in col.lower() and not 'valid' in col]
                if reflectance_cols:
                    spectrum_values = {col: spectrum[col] for col in reflectance_cols}
                    spectrum_values['contaminant_value'] = row[contaminant]
                    spectrum_values['timestamp'] = row['timestamp_iso']
                    low_spectra.append(spectrum_values)
        
        print(f"   📡 Espectros alta concentración: {len(high_spectra)}")
        print(f"   📡 Espectros baja concentración: {len(low_spectra)}")
        
        if len(high_spectra) < self.min_samples or len(low_spectra) < self.min_samples:
            print(f"   ❌ Insuficientes espectros emparejados")
            return None
        
        # Convertir a DataFrames
        high_spectra_df = pd.DataFrame(high_spectra)
        low_spectra_df = pd.DataFrame(low_spectra)
        
        # Extraer longitudes de onda
        reflectance_cols = [col for col in high_spectra_df.columns 
                           if 'reflectance' in col.lower() and col != 'contaminant_value' and col != 'timestamp']
        
        wavelengths = []
        for col in reflectance_cols:
            # Extraer número de longitud de onda del nombre de columna
            import re
            match = re.search(r'(\d+)', col)
            if match:
                wavelengths.append(int(match.group(1)))
        
        if not wavelengths:
            print(f"   ❌ No se pudieron extraer longitudes de onda")
            return None
        
        # Ordenar longitudes de onda y columnas correspondientes
        wl_col_pairs = list(zip(wavelengths, reflectance_cols))
        wl_col_pairs.sort(key=lambda x: x[0])
        wavelengths_sorted, cols_sorted = zip(*wl_col_pairs)
        
        # Calcular espectros promedio
        high_mean_spectrum = high_spectra_df[list(cols_sorted)].mean().values
        low_mean_spectrum = low_spectra_df[list(cols_sorted)].mean().values
        
        # Calcular desviaciones estándar
        high_std_spectrum = high_spectra_df[list(cols_sorted)].std().values
        low_std_spectrum = low_spectra_df[list(cols_sorted)].std().values
        
        # Calcular firma espectral (diferencia)
        spectral_signature = high_mean_spectrum - low_mean_spectrum
        
        # Identificar longitudes de onda importantes
        threshold = np.std(spectral_signature) * 1.5
        significant_pos = spectral_signature > threshold
        significant_neg = spectral_signature < -threshold
        
        important_wavelengths = []
        for i, wl in enumerate(wavelengths_sorted):
            if significant_pos[i] or significant_neg[i]:
                important_wavelengths.append((wl, spectral_signature[i]))
        
        important_wavelengths.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"   🎯 Longitudes de onda importantes: {len(important_wavelengths)}")
        
        # Preparar resultados
        result = {
            'contaminant': contaminant,
            'display_name': contaminant_info['display_name'],
            'type': data_type,
            'wavelengths': list(wavelengths_sorted),
            'high_mean_spectrum': high_mean_spectrum,
            'low_mean_spectrum': low_mean_spectrum,
            'high_std_spectrum': high_std_spectrum,
            'low_std_spectrum': low_std_spectrum,
            'spectral_signature': spectral_signature,
            'important_wavelengths': important_wavelengths[:20],  # Top 20
            'high_threshold': high_threshold,
            'low_threshold': low_threshold,
            'high_sample_count': len(high_samples),
            'low_sample_count': len(low_samples),
            'high_spectra_count': len(high_spectra),
            'low_spectra_count': len(low_spectra),
            'quantifiable_samples': contaminant_info['quantifiable_samples'],
            'value_range': contaminant_info['value_range']
        }
        
        print(f"   ✅ Firma espectral generada exitosamente")
        
        return result
    
    def plot_signature(self, result, save_plot=True):
        """Generar gráficos para una firma espectral"""
        
        contaminant_name = result['display_name']
        wavelengths = result['wavelengths']
        high_mean = result['high_mean_spectrum']
        low_mean = result['low_mean_spectrum']
        signature = result['spectral_signature']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Gráfico 1: Distribución de concentraciones (simulado)
        axes[0].bar(['Baja', 'Media', 'Alta'], 
                   [result['low_spectra_count'], 
                    result['quantifiable_samples'] - result['low_spectra_count'] - result['high_spectra_count'],
                    result['high_spectra_count']], 
                   color=['green', 'orange', 'red'], alpha=0.7)
        axes[0].axhline(y=result['low_threshold'], color='g', linestyle='--', 
                       label=f'Umbral bajo: {result["low_threshold"]:.2f}')
        axes[0].axhline(y=result['high_threshold'], color='r', linestyle='--', 
                       label=f'Umbral alto: {result["high_threshold"]:.2f}')
        axes[0].set_ylabel('Número de muestras')
        axes[0].set_title(f'Distribución de muestras - {contaminant_name}')
        axes[0].legend()
        
        # Gráfico 2: Espectros promedio
        axes[1].plot(wavelengths, high_mean, 'r-', linewidth=2, 
                    label=f'Alta concentración (n={result["high_spectra_count"]})')
        axes[1].plot(wavelengths, low_mean, 'g-', linewidth=2, 
                    label=f'Baja concentración (n={result["low_spectra_count"]})')
        
        # Añadir bandas de desviación estándar
        axes[1].fill_between(wavelengths, 
                            high_mean - result['high_std_spectrum'],
                            high_mean + result['high_std_spectrum'],
                            color='red', alpha=0.2)
        axes[1].fill_between(wavelengths, 
                            low_mean - result['low_std_spectrum'],
                            low_mean + result['low_std_spectrum'],
                            color='green', alpha=0.2)
        
        axes[1].set_xlabel('Longitud de onda (nm)')
        axes[1].set_ylabel('Reflectancia')
        axes[1].set_title('Comparación de espectros promedio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gráfico 3: Firma espectral
        axes[2].plot(wavelengths, signature, 'b-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Destacar regiones significativas
        threshold = np.std(signature) * 1.5
        significant_pos = signature > threshold
        significant_neg = signature < -threshold
        
        axes[2].fill_between(wavelengths, 0, signature, where=significant_pos, 
                           color='green', alpha=0.4, label='Aumento significativo')
        axes[2].fill_between(wavelengths, 0, signature, where=significant_neg, 
                           color='red', alpha=0.4, label='Disminución significativa')
        
        axes[2].set_xlabel('Longitud de onda (nm)')
        axes[2].set_ylabel('Diferencia de reflectancia')
        axes[2].set_title(f'Firma espectral - {contaminant_name}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            safe_name = contaminant_name.replace('/', '-').replace(' ', '_')
            plot_path = os.path.join(self.save_dir, f"{safe_name}_firma_espectral.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Gráfico guardado: {plot_path}")
        
        plt.close()
    
    def save_results(self, result):
        """Guardar resultados en archivos"""
        safe_name = result['display_name'].replace('/', '-').replace(' ', '_')
        
        # Guardar datos numéricos en CSV
        wavelength_df = pd.DataFrame({
            'wavelength': result['wavelengths'],
            'high_mean': result['high_mean_spectrum'],
            'low_mean': result['low_mean_spectrum'],
            'high_std': result['high_std_spectrum'],
            'low_std': result['low_std_spectrum'],
            'signature': result['spectral_signature']
        })
        
        csv_path = os.path.join(self.save_dir, f"{safe_name}_datos_espectrales.csv")
        wavelength_df.to_csv(csv_path, index=False)
        
        # Guardar resumen en texto
        txt_path = os.path.join(self.save_dir, f"{safe_name}_resumen.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"ANÁLISIS DE FIRMA ESPECTRAL\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Contaminante: {result['display_name']}\n")
            f.write(f"Tipo: {result['type']}\n")
            f.write(f"Muestras cuantificables: {result['quantifiable_samples']}\n")
            f.write(f"Umbral alto: {result['high_threshold']:.4f}\n")
            f.write(f"Umbral bajo: {result['low_threshold']:.4f}\n")
            f.write(f"Espectros alta concentración: {result['high_spectra_count']}\n")
            f.write(f"Espectros baja concentración: {result['low_spectra_count']}\n")
            
            if result['value_range']:
                f.write(f"Rango de valores: {result['value_range'][0]:.4f} - {result['value_range'][1]:.4f}\n")
            
            f.write(f"\nLONGITUDES DE ONDA MÁS IMPORTANTES:\n")
            f.write(f"{'-'*40}\n")
            for i, (wl, diff) in enumerate(result['important_wavelengths'][:10], 1):
                direction = "↑ aumento" if diff > 0 else "↓ disminución"
                f.write(f"{i:2d}. {wl:3d} nm: {direction} de {abs(diff):.6f}\n")
        
        # Guardar datos completos en JSON
        try:
            json_data = self._convert_numpy_to_native(result.copy())
            
            json_path = os.path.join(self.save_dir, f"{safe_name}_datos_completos.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            json_success = True
        except Exception as e:
            print(f"   ⚠️ No se pudo guardar JSON: {str(e)}")
            json_success = False
        
        print(f"   💾 Archivos guardados:")
        print(f"      📊 CSV: {csv_path}")
        print(f"      📝 TXT: {txt_path}")
        if json_success:
            print(f"      🗂️ JSON: {json_path}")
    
    def _convert_numpy_to_native(self, obj):
        """Convertir tipos numpy/pandas a tipos nativos de Python para JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_native(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_native(item) for item in obj)
        elif hasattr(obj, 'tolist'):  # NumPy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # NumPy scalars
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_all_signatures(self, dataset_type='both', max_contaminants=None):
        """
        Generar firmas espectrales para todos los contaminantes viables
        
        Args:
            dataset_type: 'inorganic', 'organic', o 'both'
            max_contaminants: Máximo número de contaminantes a procesar (None = todos)
        
        Returns:
            Lista de resultados exitosos
        """
        print("🚀 INICIANDO GENERACIÓN DE FIRMAS ESPECTRALES")
        print("="*60)
        
        # Cargar datos
        self.load_data()
        
        # Identificar contaminantes viables
        print(f"\n🔍 Identificando contaminantes viables...")
        viable_contaminants = self.identify_viable_contaminants(dataset_type)
        
        # Filtrar solo los viables
        viable_only = [c for c in viable_contaminants if c['is_viable']]
        
        if max_contaminants:
            viable_only = viable_only[:max_contaminants]
        
        print(f"\n📋 CONTAMINANTES SELECCIONADOS PARA ANÁLISIS:")
        print(f"   Total disponibles: {len(viable_contaminants)}")
        print(f"   Viables: {len(viable_only)}")
        print(f"   A procesar: {len(viable_only)}")
        
        # Mostrar lista
        for i, cont in enumerate(viable_only, 1):
            print(f"   {i:2d}. {cont['display_name']} ({cont['type']}) - "
                  f"{cont['quantifiable_samples']} muestras ({cont['quantifiable_percentage']:.1f}%)")
        
        # Procesar cada contaminante
        successful_results = []
        failed_contaminants = []
        
        print(f"\n🔬 PROCESANDO CONTAMINANTES...")
        print("="*60)
        
        for i, contaminant_info in enumerate(viable_only, 1):
            print(f"\n[{i}/{len(viable_only)}] {contaminant_info['display_name']}")
            
            try:
                result = self.generate_signature_for_contaminant(contaminant_info)
                
                if result:
                    # Generar gráficos
                    self.plot_signature(result)
                    
                    # Guardar resultados
                    self.save_results(result)
                    
                    successful_results.append(result)
                    print(f"   ✅ COMPLETADO")
                else:
                    failed_contaminants.append(contaminant_info['display_name'])
                    print(f"   ❌ FALLIDO")
                    
            except Exception as e:
                failed_contaminants.append(contaminant_info['display_name'])
                print(f"   ❌ ERROR: {str(e)}")
        
        # Generar reporte final
        self.generate_final_report(successful_results, failed_contaminants, viable_contaminants)
        
        return successful_results
    
    def generate_final_report(self, successful_results, failed_contaminants, all_contaminants):
        """Generar reporte final con resumen de todos los análisis"""
        
        report_path = os.path.join(self.save_dir, "REPORTE_FINAL.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE FINAL - GENERACIÓN DE FIRMAS ESPECTRALES\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen ejecutivo
            f.write("RESUMEN EJECUTIVO\n")
            f.write("-"*20 + "\n")
            f.write(f"Contaminantes totales evaluados: {len(all_contaminants)}\n")
            f.write(f"Contaminantes viables: {len([c for c in all_contaminants if c['is_viable']])}\n")
            f.write(f"Firmas espectrales generadas exitosamente: {len(successful_results)}\n")
            f.write(f"Contaminantes fallidos: {len(failed_contaminants)}\n")
            f.write(f"Tasa de éxito: {len(successful_results)/len([c for c in all_contaminants if c['is_viable']])*100:.1f}%\n\n")
            
            # Firmas exitosas
            f.write("FIRMAS ESPECTRALES EXITOSAS\n")
            f.write("-"*30 + "\n")
            for i, result in enumerate(successful_results, 1):
                f.write(f"{i:2d}. {result['display_name']} ({result['type']})\n")
                f.write(f"    Muestras: {result['quantifiable_samples']}\n")
                f.write(f"    Espectros: {result['high_spectra_count']} alta, {result['low_spectra_count']} baja\n")
                f.write(f"    Longitudes de onda importantes: {len(result['important_wavelengths'])}\n")
                if result['important_wavelengths']:
                    top_wl = result['important_wavelengths'][:3]
                    f.write(f"    Top 3 λ: {', '.join([f'{wl}nm' for wl, _ in top_wl])}\n")
                f.write("\n")
            
            # Contaminantes fallidos
            if failed_contaminants:
                f.write("CONTAMINANTES FALLIDOS\n")
                f.write("-"*25 + "\n")
                for i, name in enumerate(failed_contaminants, 1):
                    f.write(f"{i:2d}. {name}\n")
                f.write("\n")
            
            # Recomendaciones
            f.write("RECOMENDACIONES\n")
            f.write("-"*15 + "\n")
            
            inorganic_success = len([r for r in successful_results if r['type'] == 'inorganic'])
            organic_success = len([r for r in successful_results if r['type'] == 'organic'])
            
            f.write(f"✅ Contaminantes inorgánicos: {inorganic_success} exitosos\n")
            f.write(f"✅ Contaminantes orgánicos: {organic_success} exitosos\n\n")
            
            if inorganic_success > 0:
                f.write("• Las firmas espectrales de contaminantes inorgánicos muestran excelente potencial\n")
            if organic_success > 0:
                f.write("• Las firmas espectrales de contaminantes orgánicos son viables para los seleccionados\n")
            if len(failed_contaminants) > 0:
                f.write("• Considerar obtener más muestras para contaminantes fallidos\n")
                f.write("• Evaluar mejores métodos de correlación temporal para espectros\n")
        
        print(f"\n📄 REPORTE FINAL generado: {report_path}")

# Función principal para uso fácil
def generar_firmas_espectrales(reflectance_file, lab_inorganic_file, lab_organic_file, 
                              loq_file=None, dataset_type='both', max_contaminants=None):
    """
    Función principal para generar firmas espectrales
    
    Args:
        reflectance_file: Ruta al archivo CSV de datos espectrales
        lab_inorganic_file: Ruta al archivo CSV de contaminantes inorgánicos
        lab_organic_file: Ruta al archivo CSV de contaminantes orgánicos
        loq_file: Ruta al archivo CSV de límites LOQ (opcional)
        dataset_type: 'inorganic', 'organic', o 'both'
        max_contaminants: Máximo número de contaminantes a procesar
    
    Returns:
        Lista de firmas espectrales generadas exitosamente
    """
    
    generator = SpectralSignatureGenerator(
        reflectance_file=reflectance_file,
        lab_inorganic_file=lab_inorganic_file,
        lab_organic_file=lab_organic_file,
        loq_file=loq_file
    )
    
    return generator.generate_all_signatures(dataset_type=dataset_type, 
                                           max_contaminants=max_contaminants)

# Ejecutar si es script principal
if __name__ == "__main__":
    # CONFIGURACIÓN - Ajustar estas rutas según tu estructura
    reflectance_file = "D:/2_data/2_spectra_extracted_from_hyperspectral_acquisitions/flume_mvx_reflectance.csv"
    lab_inorganic_file = "D:/2_data/5_laboratory_reference_measurements/laboratory_measurements.csv"
    lab_organic_file = "D:/2_data/5_laboratory_reference_measurements/laboratory_measurements_organic_chemicals.csv"
    loq_file = "D:/2_data/5_laboratory_reference_measurements/laboratory_measurements_loq_organic_chemicals.csv"  # Opcional
    
    print("🌊 GENERADOR DE FIRMAS ESPECTRALES DESDE CSV")
    print("="*60)
    print("Este programa analizará tus archivos CSV para generar firmas espectrales")
    print("de contaminantes en agua usando datos de laboratorio.\n")
    
    # Verificar archivos
    files_to_check = [
        ("Espectros", reflectance_file),
        ("Inorgánicos", lab_inorganic_file),
        ("Orgánicos", lab_organic_file)
    ]
    
    all_files_exist = True
    for name, filepath in files_to_check:
        if os.path.exists(filepath):
            print(f"✅ {name}: {filepath}")
        else:
            print(f"❌ {name}: {filepath} - NO ENCONTRADO")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Faltan archivos necesarios. Ajusta las rutas en la configuración.")
        exit()
    
    # Opciones de procesamiento
    print(f"\n🔧 OPCIONES DE PROCESAMIENTO:")
    print(f"1. Solo contaminantes inorgánicos (más confiables)")
    print(f"2. Solo contaminantes orgánicos")
    print(f"3. Ambos tipos de contaminantes")
    print(f"4. Los 5 mejores contaminantes de cada tipo")
    
    choice = input(f"\nElige una opción (1-4) [por defecto: 3]: ").strip()
    
    if choice == "1":
        dataset_type = "inorganic"
        max_contaminants = None
    elif choice == "2":
        dataset_type = "organic"
        max_contaminants = None
    elif choice == "4":
        dataset_type = "both"
        max_contaminants = 10  # 5 de cada tipo aproximadamente
    else:  # Por defecto
        dataset_type = "both"
        max_contaminants = None
    
    print(f"\n🚀 Iniciando procesamiento...")
    
    # Ejecutar análisis
    results = generar_firmas_espectrales(
        reflectance_file=reflectance_file,
        lab_inorganic_file=lab_inorganic_file,
        lab_organic_file=lab_organic_file,
        loq_file=loq_file,
        dataset_type=dataset_type,
        max_contaminants=max_contaminants
    )
    
    print(f"\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"   Firmas espectrales generadas: {len(results)}")
    print(f"   Archivos guardados en: ./firmas_espectrales_csv/")
    print(f"   Revisa el REPORTE_FINAL.txt para detalles completos")