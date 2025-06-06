# diagnostico_features_datos.py
# Sistema para diagnosticar problemas en features y datos
# María José Erazo González

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from numpy import trapz

class DiagnosticoFeaturesYDatos:
    """
    Sistema de diagnóstico para identificar problemas en features y datos
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.mapeo_carpetas = {
            'Nh4_Mg_L': 'Nh4', 
            'Caffeine_Ng_L': 'Caffeine',
            'Turbidity_Ntu': 'Turbidity'
        }
    
    def diagnosticar_contaminante_completo(self, contaminante):
        """
        Diagnóstico completo de un contaminante específico
        """
        print(f"\n{'='*70}")
        print(f"🔬 DIAGNÓSTICO COMPLETO: {contaminante}")
        print(f"{'='*70}")
        
        try:
            # 1. Cargar y examinar datos crudos
            datos_crudos = self.cargar_datos_crudos(contaminante)
            self.diagnosticar_datos_crudos(datos_crudos, contaminante)
            
            # 2. Extraer y examinar features
            features = self.extraer_features_basicos(datos_crudos)
            self.diagnosticar_features(features, contaminante)
            
            # 3. Crear y examinar dataset
            dataset = self.crear_dataset_simple(features)
            self.diagnosticar_dataset(dataset, contaminante)
            
            # 4. Probar separabilidad
            self.probar_separabilidad_clases(dataset, contaminante)
            
            return {
                'datos_crudos': datos_crudos,
                'features': features,
                'dataset': dataset
            }
            
        except Exception as e:
            print(f"❌ Error en diagnóstico: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def cargar_datos_crudos(self, contaminante):
        """Carga datos crudos y los examina"""
        print(f"\n📂 1. CARGANDO DATOS CRUDOS...")
        
        carpeta = self.mapeo_carpetas[contaminante]
        ruta_carpeta = os.path.join(self.directorio_base, carpeta)
        
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) 
                              if f.endswith('_datos_espectrales.csv')]
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        datos = pd.read_csv(ruta_archivo)
        datos = datos.dropna().sort_values('wavelength').reset_index(drop=True)
        
        print(f"   📁 Archivo: {archivo_espectral}")
        print(f"   📏 Shape: {datos.shape}")
        print(f"   📋 Columnas: {list(datos.columns)}")
        
        return datos
    
    def diagnosticar_datos_crudos(self, datos, contaminante):
        """Examina la calidad de los datos espectrales crudos"""
        print(f"\n🔍 2. DIAGNÓSTICO DATOS ESPECTRALES...")
        
        wavelengths = datos['wavelength'].values
        high_mean = datos['high_mean'].values  
        low_mean = datos['low_mean'].values
        
        # Estadísticas básicas
        print(f"   🌈 Wavelengths: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm ({len(wavelengths)} puntos)")
        print(f"   📊 High mean: {high_mean.min():.6f} - {high_mean.max():.6f}")
        print(f"   📊 Low mean:  {low_mean.min():.6f} - {low_mean.max():.6f}")
        
        # Verificar diferencias entre high y low
        diferencia_absoluta = np.abs(high_mean - low_mean)
        diferencia_relativa = diferencia_absoluta / (np.abs(high_mean) + 1e-8)
        
        print(f"   📈 Diferencia absoluta: {diferencia_absoluta.mean():.6f} ± {diferencia_absoluta.std():.6f}")
        print(f"   📈 Diferencia relativa: {diferencia_relativa.mean():.6f} ± {diferencia_relativa.std():.6f}")
        
        # Verificar si hay diferencias significativas
        if diferencia_relativa.mean() < 0.01:
            print(f"   🚨 PROBLEMA: Diferencias muy pequeñas entre high/low (<1%)")
        elif diferencia_relativa.mean() < 0.05:
            print(f"   ⚠️ ADVERTENCIA: Diferencias pequeñas entre high/low (<5%)")
        else:
            print(f"   ✅ Diferencias detectables entre high/low (>{diferencia_relativa.mean()*100:.1f}%)")
        
        # Verificar valores problemáticos
        if np.any(np.isnan(high_mean)) or np.any(np.isnan(low_mean)):
            print(f"   🚨 PROBLEMA: Valores NaN detectados")
        
        if np.all(high_mean == low_mean):
            print(f"   🚨 PROBLEMA CRÍTICO: High y Low son idénticos")
        
        # Buscar regiones más discriminativas
        indices_max_diff = np.argsort(diferencia_relativa)[-5:]
        print(f"   🎯 Regiones más discriminativas:")
        for i in indices_max_diff:
            wl = wavelengths[i]
            diff = diferencia_relativa[i]
            print(f"      - {wl:.1f} nm: {diff*100:.2f}% diferencia")
    
    def extraer_features_basicos(self, datos):
        """Extrae features básicos para diagnóstico"""
        print(f"\n🔧 3. EXTRAYENDO FEATURES BÁSICOS...")
        
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        features = {}
        
        # Features estadísticos básicos
        for concentracion, response in [('high', high_response), ('low', low_response)]:
            features[f'{concentracion}_mean'] = np.mean(response)
            features[f'{concentracion}_std'] = np.std(response)
            features[f'{concentracion}_max'] = np.max(response)
            features[f'{concentracion}_min'] = np.min(response)
            features[f'{concentracion}_range'] = np.ptp(response)
            
            # Área bajo la curva
            features[f'{concentracion}_auc'] = trapz(response, wavelengths)
            
            # Pendiente global
            if len(wavelengths) > 1:
                slope, _ = np.polyfit(wavelengths, response, 1)
                features[f'{concentracion}_slope'] = slope
        
        # Features comparativos
        features['ratio_mean'] = features['high_mean'] / (features['low_mean'] + 1e-8)
        features['diff_mean'] = features['high_mean'] - features['low_mean']
        features['ratio_auc'] = features['high_auc'] / (features['low_auc'] + 1e-8)
        features['ratio_max'] = features['high_max'] / (features['low_max'] + 1e-8)
        
        print(f"   ✅ Features extraídos: {len(features)}")
        
        return features
    
    def diagnosticar_features(self, features, contaminante):
        """Diagnóstica la calidad de los features extraídos"""
        print(f"\n🎯 4. DIAGNÓSTICO FEATURES...")
        
        # Verificar valores problemáticos
        valores_problematicos = 0
        
        for nombre, valor in features.items():
            if np.isnan(valor) or np.isinf(valor):
                print(f"   🚨 PROBLEMA: {nombre} = {valor}")
                valores_problematicos += 1
            elif abs(valor) < 1e-10:
                print(f"   ⚠️ ADVERTENCIA: {nombre} muy pequeño = {valor:.2e}")
        
        if valores_problematicos == 0:
            print(f"   ✅ No se detectaron valores problemáticos")
        
        # Mostrar features clave
        print(f"\n   📊 FEATURES CLAVE:")
        features_clave = ['high_mean', 'low_mean', 'ratio_mean', 'diff_mean', 'high_auc', 'low_auc']
        
        for nombre in features_clave:
            if nombre in features:
                valor = features[nombre]
                print(f"      {nombre:12}: {valor:.6f}")
        
        # Verificar ratios y diferencias
        ratio_mean = features.get('ratio_mean', 1.0)
        diff_mean = features.get('diff_mean', 0.0)
        
        print(f"\n   🔍 ANÁLISIS DISCRIMINATIVO:")
        print(f"      Ratio mean (H/L): {ratio_mean:.4f}")
        print(f"      Diff mean (H-L):  {diff_mean:.6f}")
        
        if abs(ratio_mean - 1.0) < 0.01:
            print(f"      🚨 PROBLEMA: Ratio muy cercano a 1 (sin diferencia)")
        elif abs(ratio_mean - 1.0) < 0.1:
            print(f"      ⚠️ ADVERTENCIA: Ratio cercano a 1 (poca diferencia)")
        else:
            print(f"      ✅ Ratio discriminativo")
        
        if abs(diff_mean) < 1e-6:
            print(f"      🚨 PROBLEMA: Diferencia media muy pequeña")
        else:
            print(f"      ✅ Diferencia media detectable")
    
    def crear_dataset_simple(self, features):
        """Crea dataset simple para diagnóstico"""
        print(f"\n📊 5. CREANDO DATASET DIAGNÓSTICO...")
        
        # Muestra alta concentración (original)
        muestra_alta = list(features.values())
        
        # Muestra baja concentración (invertida simple)
        muestra_baja = []
        for nombre, valor in features.items():
            if nombre.startswith('high_'):
                # Buscar equivalente low_
                low_nombre = nombre.replace('high_', 'low_')
                if low_nombre in features:
                    muestra_baja.append(features[low_nombre])
                else:
                    muestra_baja.append(valor * 0.5)  # Reducir
            elif nombre.startswith('low_'):
                # Buscar equivalente high_
                high_nombre = nombre.replace('low_', 'high_')
                if high_nombre in features:
                    muestra_baja.append(features[high_nombre])
                else:
                    muestra_baja.append(valor * 1.5)  # Aumentar
            elif 'ratio' in nombre:
                muestra_baja.append(1 / (valor + 1e-8))
            elif 'diff' in nombre:
                muestra_baja.append(-valor)
            else:
                muestra_baja.append(valor)
        
        # Crear DataFrame
        feature_names = list(features.keys())
        datos_matriz = [muestra_alta, muestra_baja]
        labels = [1, 0]  # 1=alta, 0=baja
        
        df = pd.DataFrame(datos_matriz, columns=feature_names)
        df['label'] = labels
        
        print(f"   ✅ Dataset creado: {df.shape}")
        print(f"   📊 Clases: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def diagnosticar_dataset(self, dataset, contaminante):
        """Diagnóstica el dataset creado"""
        print(f"\n🔍 6. DIAGNÓSTICO DATASET...")
        
        # Separar features y labels
        feature_columns = [col for col in dataset.columns if col != 'label']
        X = dataset[feature_columns].values
        y = dataset['label'].values
        
        print(f"   📏 Shape X: {X.shape}")
        print(f"   📏 Shape y: {y.shape}")
        print(f"   📊 Clases únicas: {np.unique(y)}")
        
        # Verificar diferencias entre clases
        print(f"\n   🔍 DIFERENCIAS ENTRE CLASES:")
        
        clase_0 = X[y == 0][0]  # Primera muestra clase 0
        clase_1 = X[y == 1][0]  # Primera muestra clase 1
        
        diferencias_absolutas = np.abs(clase_1 - clase_0)
        diferencias_relativas = diferencias_absolutas / (np.abs(clase_1) + 1e-8)
        
        print(f"      Diferencia absoluta promedio: {diferencias_absolutas.mean():.6f}")
        print(f"      Diferencia relativa promedio: {diferencias_relativas.mean():.6f}")
        
        # Mostrar features con más diferencia
        indices_ordenados = np.argsort(diferencias_relativas)[::-1]
        print(f"\n   🎯 TOP 5 FEATURES MÁS DISCRIMINATIVOS:")
        for i in range(min(5, len(feature_columns))):
            idx = indices_ordenados[i]
            nombre = feature_columns[idx]
            val_0 = clase_0[idx]
            val_1 = clase_1[idx]
            diff_rel = diferencias_relativas[idx]
            print(f"      {i+1}. {nombre:15}: {val_0:.6f} vs {val_1:.6f} (Δ{diff_rel*100:.1f}%)")
        
        # Verificar features problemáticos
        features_problematicos = 0
        for i, nombre in enumerate(feature_columns):
            if diferencias_relativas[i] < 0.001:  # <0.1% diferencia
                print(f"   ⚠️ Feature sin discriminación: {nombre}")
                features_problematicos += 1
        
        if features_problematicos == 0:
            print(f"   ✅ Todos los features tienen alguna discriminación")
        else:
            print(f"   🚨 {features_problematicos} features problemáticos")
    
    def probar_separabilidad_clases(self, dataset, contaminante):
        """Prueba si las clases son separables"""
        print(f"\n🎯 7. PRUEBA DE SEPARABILIDAD...")
        
        # Separar datos
        feature_columns = [col for col in dataset.columns if col != 'label']
        X = dataset[feature_columns].values
        y = dataset['label'].values
        
        # Probar con modelo simple
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            
            # Escalado
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Modelo simple
            modelo_simple = LogisticRegression(random_state=42)
            modelo_simple.fit(X_scaled, y)
            
            # Predicción (train=test en este caso)
            y_pred = modelo_simple.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            
            print(f"   📊 Accuracy Logistic Regression: {accuracy:.4f}")
            
            if accuracy == 1.0:
                print(f"   ✅ CLASES PERFECTAMENTE SEPARABLES")
            elif accuracy >= 0.8:
                print(f"   ✅ CLASES BIEN SEPARABLES")
            elif accuracy >= 0.6:
                print(f"   💛 CLASES MODERADAMENTE SEPARABLES")
            else:
                print(f"   🚨 CLASES DIFÍCILMENTE SEPARABLES")
            
            # Mostrar coeficientes más importantes
            coeficientes = modelo_simple.coef_[0]
            indices_importantes = np.argsort(np.abs(coeficientes))[::-1]
            
            print(f"\n   🔑 FEATURES MÁS IMPORTANTES (LogReg):")
            for i in range(min(3, len(feature_columns))):
                idx = indices_importantes[i]
                nombre = feature_columns[idx]
                coef = coeficientes[idx]
                print(f"      {i+1}. {nombre:15}: {coef:+.4f}")
                
        except Exception as e:
            print(f"   ❌ Error en prueba separabilidad: {e}")
    
    def diagnosticar_multiples_contaminantes(self):
        """Diagnóstica múltiples contaminantes para comparar"""
        print(f"🔬 DIAGNÓSTICO MÚLTIPLES CONTAMINANTES")
        print(f"="*60)
        
        contaminantes = ['Nh4_Mg_L', 'Caffeine_Ng_L', 'Turbidity_Ntu']
        resultados = {}
        
        for contaminante in contaminantes:
            print(f"\n{'='*20} {contaminante} {'='*20}")
            resultado = self.diagnosticar_contaminante_completo(contaminante)
            resultados[contaminante] = resultado
        
        # Resumen comparativo
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN COMPARATIVO")
        print(f"{'='*60}")
        
        for contaminante, resultado in resultados.items():
            if resultado:
                features = resultado['features']
                ratio_mean = features.get('ratio_mean', 1.0)
                diff_mean = features.get('diff_mean', 0.0)
                
                print(f"{contaminante:15} | Ratio: {ratio_mean:6.3f} | Diff: {diff_mean:8.6f}")
        
        return resultados

def main_diagnostico():
    """Función principal de diagnóstico"""
    
    print("🔬 SISTEMA DE DIAGNÓSTICO DE FEATURES Y DATOS")
    print("="*55)
    print("🎯 Identificando por qué F1=0.0000 en todos los modelos")
    print()
    
    diagnostico = DiagnosticoFeaturesYDatos("todo/firmas_espectrales_csv")
    
    # Ejecutar diagnóstico múltiple
    resultados = diagnostico.diagnosticar_multiples_contaminantes()
    
    print(f"\n🏁 DIAGNÓSTICO COMPLETADO")
    print(f"   📊 Contaminantes analizados: {len([r for r in resultados.values() if r])}")
    
    return resultados

if __name__ == "__main__":
    main_diagnostico()