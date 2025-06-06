# entrenar_svm_firmas_espectrales_corregido.py
# Versión corregida del entrenador SVM

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
from todo.SVM.dataset_anti_overfitting import DatasetAntiOverfitting

class SVMFirmasEspectralesCorregido:
    """Versión corregida del entrenador SVM para firmas espectrales"""
    
    def __init__(self, firmas_dir="todo/firmas_espectrales_csv", output_dir="modelos_svm_firmas_corregido"):
        self.firmas_dir = firmas_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.contaminantes_disponibles = self.detectar_contaminantes_disponibles()
        
        print(f"🎯 SVMFirmasEspectrales Corregido inicializado")
        print(f"📁 Directorio firmas: {firmas_dir}")
        print(f"📁 Directorio modelos: {output_dir}")
        print(f"🧪 Contaminantes detectados: {len(self.contaminantes_disponibles)}")
    
    def detectar_contaminantes_disponibles(self):
        """Detecta automáticamente qué contaminantes tienen firmas espectrales"""
        contaminantes = {}
        
        if not os.path.exists(self.firmas_dir):
            print(f"❌ No se encontró directorio: {self.firmas_dir}")
            return contaminantes
        
        for archivo in os.listdir(self.firmas_dir):
            if archivo.endswith("_datos_espectrales.csv"):
                nombre_base = archivo.replace("_datos_espectrales.csv", "")
                ruta_completa = os.path.join(self.firmas_dir, archivo)
                
                try:
                    df_test = pd.read_csv(ruta_completa)
                    if len(df_test) > 0 and 'signature' in df_test.columns:
                        contaminantes[nombre_base] = {
                            'archivo_datos': ruta_completa,
                            'archivo_resumen': os.path.join(self.firmas_dir, f"{nombre_base}_resumen.txt"),
                            'tipo': 'inorganic' if nombre_base.lower() in ['doc_mg_l', 'nh4_mg_l', 'turbidity_ntu', 'po4_mg_l', 'so4_mg_l', 'nsol_mg_l'] else 'organic'
                        }
                        print(f"  ✅ {nombre_base} ({contaminantes[nombre_base]['tipo']})")
                except Exception as e:
                    print(f"  ❌ Error procesando {archivo}: {e}")
                    continue
        
        return contaminantes
    
    def cargar_firma_espectral(self, contaminante):
        """Carga la firma espectral de un contaminante"""
        if contaminante not in self.contaminantes_disponibles:
            raise ValueError(f"Contaminante {contaminante} no disponible")
        
        archivo_datos = self.contaminantes_disponibles[contaminante]['archivo_datos']
        print(f"📊 Cargando firma espectral: {contaminante}")
        
        df_firma = pd.read_csv(archivo_datos)
        print(f"   📈 Longitudes de onda: {len(df_firma)}")
        print(f"   🌊 Rango espectral: {df_firma['wavelength'].min()}-{df_firma['wavelength'].max()} nm")
        
        return df_firma
    
    def crear_dataset_desde_firma(self, df_firma, n_samples_por_clase=200, anti_overfitting=True):
        """Crea dataset sintético desde firma espectral"""
        generador = DatasetAntiOverfitting(df_firma)
        X, y = generador.generar_dataset_realista(n_samples_por_clase, anti_overfitting)
        return X, y, generador.wavelengths
    
    def crear_features_espectrales_avanzadas_corregidas(self, X, wavelengths):
        """Versión corregida para crear características espectrales avanzadas"""
        print(f"🔧 Creando features espectrales avanzadas (versión corregida)...")
        
        features_avanzadas = []
        nombres_features = []
        
        for i in range(X.shape[0]):
            espectro = X[i, :]
            features_muestra = []
            
            # 1. Estadísticas básicas (seguras)
            try:
                features_muestra.extend([
                    np.mean(espectro),
                    np.std(espectro),
                    np.max(espectro),
                    np.min(espectro),
                    np.max(espectro) - np.min(espectro),
                    np.median(espectro)
                ])
                
                if i == 0:
                    nombres_features.extend(['mean', 'std', 'max', 'min', 'range', 'median'])
            except Exception as e:
                print(f"   ⚠️ Error en estadísticas básicas: {e}")
                continue
            
            # 2. Momentos estadísticos (con manejo de errores)
            try:
                from scipy import stats
                skewness = stats.skew(espectro)
                kurtosis = stats.kurtosis(espectro)
                
                # Verificar que no sean NaN o infinitos
                if np.isfinite(skewness) and np.isfinite(kurtosis):
                    features_muestra.extend([skewness, kurtosis])
                    if i == 0:
                        nombres_features.extend(['skewness', 'kurtosis'])
                else:
                    features_muestra.extend([0.0, 0.0])
                    if i == 0:
                        nombres_features.extend(['skewness', 'kurtosis'])
            except Exception as e:
                features_muestra.extend([0.0, 0.0])
                if i == 0:
                    nombres_features.extend(['skewness', 'kurtosis'])
            
            # 3. Pendiente general
            try:
                x_vals = np.arange(len(espectro))
                slope, _ = np.polyfit(x_vals, espectro, 1)
                features_muestra.append(slope if np.isfinite(slope) else 0.0)
                if i == 0:
                    nombres_features.append('slope_general')
            except Exception:
                features_muestra.append(0.0)
                if i == 0:
                    nombres_features.append('slope_general')
            
            # 4. Características por regiones (simplificado)
            n_wavelengths = len(wavelengths)
            try:
                region_1 = espectro[:n_wavelengths//3]
                region_2 = espectro[n_wavelengths//3:2*n_wavelengths//3]
                region_3 = espectro[2*n_wavelengths//3:]
                
                for j, region in enumerate([region_1, region_2, region_3], 1):
                    if len(region) > 0:
                        mean_val = np.mean(region)
                        std_val = np.std(region)
                        max_val = np.max(region)
                        
                        features_muestra.extend([mean_val, std_val, max_val])
                        if i == 0:
                            nombres_features.extend([f'region{j}_mean', f'region{j}_std', f'region{j}_max'])
            except Exception as e:
                # Si hay error, agregar ceros
                for j in range(1, 4):
                    features_muestra.extend([0.0, 0.0, 0.0])
                    if i == 0:
                        nombres_features.extend([f'region{j}_mean', f'region{j}_std', f'region{j}_max'])
            
            # 5. Área bajo la curva
            try:
                area_total = np.trapz(espectro)
                features_muestra.append(area_total if np.isfinite(area_total) else 0.0)
                if i == 0:
                    nombres_features.append('area_total')
            except Exception:
                features_muestra.append(0.0)
                if i == 0:
                    nombres_features.append('area_total')
            
            # 6. Detección de picos (versión más robusta)
            try:
                from scipy.signal import find_peaks
                # Usar umbral más conservador
                umbral_picos = np.mean(espectro) + 0.5 * np.std(espectro)
                peaks, _ = find_peaks(espectro, height=umbral_picos)
                
                n_peaks = len(peaks)
                peak_max_height = np.max(espectro[peaks]) if n_peaks > 0 else 0.0
                peak_mean_height = np.mean(espectro[peaks]) if n_peaks > 0 else 0.0
                
                features_muestra.extend([n_peaks, peak_max_height, peak_mean_height])
                if i == 0:
                    nombres_features.extend(['n_peaks', 'peak_max_height', 'peak_mean_height'])
            except Exception as e:
                # Si hay error en detección de picos, usar valores por defecto
                features_muestra.extend([0.0, 0.0, 0.0])
                if i == 0:
                    nombres_features.extend(['n_peaks', 'peak_max_height', 'peak_mean_height'])
            
            features_avanzadas.append(features_muestra)
        
        X_features = np.array(features_avanzadas)
        
        # Verificar que no hay NaN o infinitos
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"   ✅ Features creadas: {X_features.shape[1]} características")
        print(f"   📝 Primeras features: {nombres_features[:5]}")
        
        return X_features, nombres_features
    
    def entrenar_svm_optimizado_corregido(self, X, y, nombres_features, contaminante, estrategia='auto'):
        """Versión corregida del entrenamiento SVM"""
        print(f"\n🤖 Entrenando SVM optimizado (CORREGIDO) para: {contaminante}")
        print(f"   📊 Dataset: {X.shape[0]} muestras x {X.shape[1]} features")
        
        # Determinar estrategia
        if estrategia == 'auto':
            if X.shape[0] < 50:
                estrategia = 'conservadora'
            elif X.shape[0] < 150:
                estrategia = 'moderada'
            else:
                estrategia = 'agresiva'
        
        print(f"   🎯 Estrategia seleccionada: {estrategia}")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalización
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Selección de características según estrategia (CORREGIDO)
        if estrategia == 'conservadora':
            # Usar SelectKBest (más seguro)
            selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
            cv_folds = 3
            
        elif estrategia == 'moderada':
            # Usar SelectKBest
            selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf', 'linear']
            }
            cv_folds = 5
            
        else:  # agresiva - CORREGIDO
            # Para RFE, primero entrenar un modelo base
            if X.shape[1] > 20:
                try:
                    # Entrenar SVM base con kernel lineal primero
                    base_svm = SVC(kernel='linear', C=1.0, random_state=42)
                    base_svm.fit(X_train_scaled, y_train)
                    
                    # Ahora usar RFE con el modelo entrenado
                    selector = RFE(estimator=base_svm, n_features_to_select=20, step=2)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                    X_test_selected = selector.transform(X_test_scaled)
                except Exception as e:
                    print(f"   ⚠️ Error con RFE, usando SelectKBest: {e}")
                    # Fallback a SelectKBest
                    selector = SelectKBest(score_func=f_classif, k=20)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                    X_test_selected = selector.transform(X_test_scaled)
            else:
                selector = None
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
            
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            cv_folds = 5
        
        print(f"   🔧 Features seleccionadas: {X_train_selected.shape[1]}")
        
        # Grid Search
        svm_base = SVC(random_state=42, probability=True)
        
        try:
            grid_search = GridSearchCV(
                svm_base, param_grid, 
                cv=cv_folds, scoring='f1', 
                n_jobs=-1, verbose=0  # Reducir verbosidad
            )
            
            print(f"   🔍 Ejecutando Grid Search...")
            grid_search.fit(X_train_selected, y_train)
            
            mejor_svm = grid_search.best_estimator_
            
            print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
            print(f"   📈 Mejor score CV: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Error en Grid Search, usando parámetros por defecto: {e}")
            # Fallback a modelo simple
            mejor_svm = SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42, probability=True)
            mejor_svm.fit(X_train_selected, y_train)
        
        # Evaluación
        y_pred = mejor_svm.predict(X_test_selected)
        y_pred_proba = mejor_svm.predict_proba(X_test_selected)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 RESULTADOS:")
        print(f"   🎯 Accuracy: {accuracy:.4f}")
        print(f"   🎯 F1-Score: {f1:.4f}")
        print(f"   🎯 AUC: {auc:.4f}")
        
        # Preparar resultados
        resultados = {
            'contaminante': contaminante,
            'estrategia': estrategia,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'auc_score': float(auc),
            'n_features_usadas': X_train_selected.shape[1],
            'fecha_entrenamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Guardar modelo
        modelo_info = {
            'modelo': mejor_svm,
            'scaler': scaler,
            'selector': selector,
            'resultados': resultados
        }
        
        modelo_path = os.path.join(self.output_dir, f"{contaminante}_svm_corregido.joblib")
        joblib.dump(modelo_info, modelo_path)
        
        # Guardar resultados JSON
        resultados_path = os.path.join(self.output_dir, f"{contaminante}_resultados_corregido.json")
        with open(resultados_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Modelo guardado: {modelo_path}")
        
        return resultados, modelo_info
    
    def entrenar_contaminante_corregido(self, contaminante, n_samples=150, estrategia='auto'):
        """Entrena SVM para un contaminante específico (versión corregida)"""
        
        print(f"\n{'='*60}")
        print(f"🧪 ENTRENANDO SVM (CORREGIDO): {contaminante.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Cargar firma espectral
            df_firma = self.cargar_firma_espectral(contaminante)
            
            # 2. Crear dataset sintético
            X_raw, y, wavelengths = self.crear_dataset_desde_firma(df_firma, n_samples_por_clase=n_samples//2)
            
            # 3. Crear características avanzadas (versión corregida)
            X_features, nombres_features = self.crear_features_espectrales_avanzadas_corregidas(X_raw, wavelengths)
            
            # 4. Entrenar SVM (versión corregida)
            resultados, modelo_info = self.entrenar_svm_optimizado_corregido(
                X_features, y, nombres_features, contaminante, estrategia
            )
            
            print(f"✅ ENTRENAMIENTO COMPLETADO: {contaminante}")
            
            return resultados, modelo_info
            
        except Exception as e:
            print(f"❌ Error entrenando {contaminante}: {str(e)}")
            return None, None
    
    def entrenar_contaminantes_prioritarios_corregido(self):
        """Entrena solo los contaminantes prioritarios de forma segura"""
        
        # Lista de contaminantes prioritarios
        prioritarios = [
            'Doc_Mg_L',
            'Nh4_Mg_L', 
            'Turbidity_Ntu',
            'Caffeine_Ng_L',
            'Acesulfame_Ng_L'
        ]
        
        print(f"🎯 ENTRENANDO CONTAMINANTES PRIORITARIOS (VERSIÓN CORREGIDA)")
        print(f"📋 Lista: {prioritarios}")
        
        disponibles = [c for c in prioritarios if c in self.contaminantes_disponibles]
        no_disponibles = [c for c in prioritarios if c not in self.contaminantes_disponibles]
        
        if no_disponibles:
            print(f"⚠️ No disponibles: {no_disponibles}")
        
        print(f"✅ Disponibles: {disponibles}")
        
        resultados_todos = {}
        
        for i, contaminante in enumerate(disponibles, 1):
            print(f"\n[{i}/{len(disponibles)}] Procesando: {contaminante}")
            
            resultado, _ = self.entrenar_contaminante_corregido(
                contaminante, n_samples=100, estrategia='moderada'
            )
            
            if resultado:
                resultados_todos[contaminante] = resultado
        
        # Generar reporte
        self.generar_reporte_corregido(resultados_todos)
        
        return resultados_todos
    
    def generar_reporte_corregido(self, resultados_todos):
        """Genera reporte de resultados"""
        
        print(f"\n{'='*60}")
        print(f"📊 REPORTE DE RESULTADOS (CORREGIDO)")
        print(f"{'='*60}")
        
        if not resultados_todos:
            print("❌ No hay resultados para reportar")
            return
        
        # Crear DataFrame
        datos_reporte = []
        for contaminante, resultado in resultados_todos.items():
            tipo = self.contaminantes_disponibles[contaminante]['tipo']
            datos_reporte.append({
                'Contaminante': contaminante,
                'Tipo': tipo,
                'Accuracy': resultado['accuracy'],
                'F1-Score': resultado['f1_score'],
                'AUC': resultado['auc_score'],
                'Features': resultado['n_features_usadas'],
                'Estrategia': resultado['estrategia']
            })
        
        df_reporte = pd.DataFrame(datos_reporte)
        
        # Guardar reporte
        reporte_path = os.path.join(self.output_dir, "reporte_svm_corregido.csv")
        df_reporte.to_csv(reporte_path, index=False, encoding='utf-8')
        
        print(f"📈 ESTADÍSTICAS:")
        print(f"   Modelos entrenados: {len(resultados_todos)}")
        print(f"   Accuracy promedio: {df_reporte['Accuracy'].mean():.4f}")
        print(f"   F1-Score promedio: {df_reporte['F1-Score'].mean():.4f}")
        print(f"   AUC promedio: {df_reporte['AUC'].mean():.4f}")
        
        # Top modelos
        top_3 = df_reporte.nlargest(3, 'F1-Score')
        print(f"\n🏆 TOP 3 MEJORES MODELOS:")
        for _, row in top_3.iterrows():
            print(f"   {row['Contaminante']:20} | F1: {row['F1-Score']:.4f} | Tipo: {row['Tipo']}")
        
        print(f"\n📁 Reporte guardado: {reporte_path}")
        
        return df_reporte

# Función principal para ejecutar entrenamiento corregido
def ejecutar_entrenamiento_corregido():
    """Función principal para ejecutar el entrenamiento corregido"""
    
    print(f"🌊 ENTRENADOR SVM CORREGIDO - DETECCIÓN DE CONTAMINANTES")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Crear entrenador
    entrenador = SVMFirmasEspectralesCorregido()
    
    # Entrenar contaminantes prioritarios
    resultados = entrenador.entrenar_contaminantes_prioritarios_corregido()
    
    print(f"\n✅ PROCESO COMPLETADO")
    print(f"📊 Modelos entrenados: {len(resultados)}")
    print(f"📁 Revisa la carpeta 'modelos_svm_firmas_corregido' para los resultados")
    
    return resultados

if __name__ == "__main__":
    # Ejecutar entrenamiento corregido
    resultados = ejecutar_entrenamiento_corregido()