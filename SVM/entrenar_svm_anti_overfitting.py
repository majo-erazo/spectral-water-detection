# entrenar_svm_anti_overfitting.py
# Versión robusta que integra técnicas anti-overfitting por defecto

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
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from todo.SVM.dataset_anti_overfitting import DatasetAntiOverfitting

class SVMAntiOverfitting:
    """Entrenador SVM con técnicas anti-overfitting integradas"""
    
    def __init__(self, firmas_dir="todo/firmas_espectrales_csv", output_dir="modelos_svm_anti_overfitting"):
        self.firmas_dir = firmas_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.contaminantes_disponibles = self.detectar_contaminantes_disponibles()
        
        # Parámetros anti-overfitting más conservadores
        self.config_anti_overfitting = {
            'test_size': 0.3,  # 30% para test (más estricto)
            'cv_folds': 10,    # Más folds para CV
            'max_features': 12, # Menos características
            'regularization_strong': True,  # Regularización fuerte por defecto
            'noise_factor': 0.08,  # 8% de ruido
            'samples_per_class': 200  # Más muestras para mejor generalización
        }
        
        print(f"🛡️ SVMAntiOverfitting inicializado")
        print(f"📁 Directorio firmas: {firmas_dir}")
        print(f"📁 Directorio modelos: {output_dir}")
        print(f"🧪 Contaminantes detectados: {len(self.contaminantes_disponibles)}")
        print(f"⚙️ Configuración anti-overfitting activada")
    
    def detectar_contaminantes_disponibles(self):
        """Detecta contaminantes disponibles"""
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
                            'tipo': 'inorganic' if nombre_base.lower() in ['doc_mg_l', 'nh4_mg_l', 'turbidity_ntu', 'po4_mg_l', 'so4_mg_l', 'nsol_mg_l'] else 'organic'
                        }
                        print(f"  ✅ {nombre_base} ({contaminantes[nombre_base]['tipo']})")
                except Exception as e:
                    print(f"  ❌ Error procesando {archivo}: {e}")
        
        return contaminantes
    
    def cargar_firma_espectral(self, contaminante):
        """Carga firma espectral de un contaminante"""
        if contaminante not in self.contaminantes_disponibles:
            raise ValueError(f"Contaminante {contaminante} no disponible")
        
        archivo_datos = self.contaminantes_disponibles[contaminante]['archivo_datos']
        print(f"📊 Cargando firma espectral: {contaminante}")
        
        df_firma = pd.read_csv(archivo_datos)
        print(f"   📈 Longitudes de onda: {len(df_firma)}")
        print(f"   🌊 Rango espectral: {df_firma['wavelength'].min()}-{df_firma['wavelength'].max()} nm")
        
        return df_firma
    
    def crear_dataset_anti_overfitting(self, df_firma, samples_per_class=None):
        """Crea dataset con técnicas anti-overfitting"""
        
        if samples_per_class is None:
            samples_per_class = self.config_anti_overfitting['samples_per_class'] // 2
        
        print(f"🛡️ Creando dataset anti-overfitting...")
        print(f"   📊 Muestras por clase: {samples_per_class}")
        print(f"   🔊 Factor de ruido: {self.config_anti_overfitting['noise_factor']}")
        
        generador = DatasetAntiOverfitting(df_firma)
        
        # USAR SIEMPRE anti_overfitting=True
        X, y = generador.generar_dataset_realista(
            n_samples_por_clase=samples_per_class, 
            anti_overfitting=True  # OBLIGATORIO
        )
        
        return X, y, generador.wavelengths
    
    def crear_features_robustas(self, X, wavelengths):
        """Crea características espectrales con enfoque conservador"""
        print(f"🔧 Creando features espectrales robustas...")
        
        features_robustas = []
        nombres_features = []
        
        for i in range(X.shape[0]):
            espectro = X[i, :]
            features_muestra = []
            
            # 1. Solo estadísticas básicas más robustas
            features_muestra.extend([
                np.mean(espectro),
                np.std(espectro),
                np.median(espectro),
                np.percentile(espectro, 25),  # Q1
                np.percentile(espectro, 75),  # Q3
            ])
            
            if i == 0:
                nombres_features.extend(['mean', 'std', 'median', 'q25', 'q75'])
            
            # 2. Características por regiones (simplificado)
            n_wavelengths = len(wavelengths)
            region_1 = espectro[:n_wavelengths//3]
            region_2 = espectro[n_wavelengths//3:2*n_wavelengths//3]
            region_3 = espectro[2*n_wavelengths//3:]
            
            for j, region in enumerate([region_1, region_2, region_3], 1):
                if len(region) > 0:
                    features_muestra.extend([
                        np.mean(region),
                        np.std(region)
                    ])
                    if i == 0:
                        nombres_features.extend([f'region{j}_mean', f'region{j}_std'])
            
            # 3. Área total (robusta)
            area_total = np.trapz(espectro)
            features_muestra.append(area_total)
            if i == 0:
                nombres_features.append('area_total')
            
            features_robustas.append(features_muestra)
        
        X_features = np.array(features_robustas)
        
        # Limpiar datos problemáticos
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"   ✅ Features robustas: {X_features.shape[1]} características")
        print(f"   📝 Limitado a {self.config_anti_overfitting['max_features']} features max")
        
        return X_features, nombres_features
    
    def entrenar_svm_robusto(self, X, y, nombres_features, contaminante):
        """Entrenamiento SVM con máxima robustez anti-overfitting"""
        
        print(f"\n🛡️ Entrenando SVM ANTI-OVERFITTING para: {contaminante}")
        print(f"   📊 Dataset: {X.shape[0]} muestras x {X.shape[1]} features")
        
        # División train/test MÁS ESTRICTA
        test_size = self.config_anti_overfitting['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   📊 División estricta: {len(X_train)} train ({100*(1-test_size):.0f}%), {len(X_test)} test ({100*test_size:.0f}%)")
        
        # Normalización
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Selección de características MÁS CONSERVADORA
        max_features = min(self.config_anti_overfitting['max_features'], X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=max_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        print(f"   🔧 Features seleccionadas: {X_train_selected.shape[1]} (max permitido: {max_features})")
        
        # Grid Search MÁS CONSERVADOR
        if self.config_anti_overfitting['regularization_strong']:
            param_grid = {
                'C': [0.01, 0.1, 1.0],  # Regularización MÁS fuerte
                'gamma': ['scale'],      # Solo scale
                'kernel': ['rbf']        # Solo RBF
            }
        else:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        
        cv_folds = self.config_anti_overfitting['cv_folds']
        
        print(f"   🎛️ Grid Search conservador: {len(param_grid['C'])} x {len(param_grid['gamma'])} x {len(param_grid['kernel'])} = {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])} combinaciones")
        print(f"   🔄 CV folds: {cv_folds}")
        
        svm_base = SVC(random_state=42, probability=True)
        
        try:
            # Usar StratifiedKFold explícito para más control
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                svm_base, param_grid, 
                cv=cv, scoring='f1',  # F1 para balancear precisión/recall
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train_selected, y_train)
            mejor_svm = grid_search.best_estimator_
            
            print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
            print(f"   📈 Mejor CV F1-Score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Error en Grid Search: {e}")
            # Modelo por defecto ultra-conservador
            mejor_svm = SVC(C=0.1, gamma='scale', kernel='rbf', random_state=42, probability=True)
            mejor_svm.fit(X_train_selected, y_train)
        
        # Evaluación COMPLETA
        y_train_pred = mejor_svm.predict(X_train_selected)
        y_test_pred = mejor_svm.predict(X_test_selected)
        y_test_proba = mejor_svm.predict_proba(X_test_selected)[:, 1]
        
        # Métricas train
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Métricas test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Análisis de GAPS (clave para detectar overfitting)
        accuracy_gap = train_accuracy - test_accuracy
        f1_gap = train_f1 - test_f1
        
        # Validación cruzada EXTENDIDA
        cv_accuracy = cross_val_score(mejor_svm, X_train_selected, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(mejor_svm, X_train_selected, y_train, cv=cv, scoring='f1')
        
        print(f"\n📊 RESULTADOS ANTI-OVERFITTING:")
        print(f"   🎯 Train Accuracy: {train_accuracy:.4f}")
        print(f"   🎯 Test Accuracy:  {test_accuracy:.4f} (Gap: {accuracy_gap:+.4f})")
        print(f"   🎯 Test F1-Score:  {test_f1:.4f} (Gap: {f1_gap:+.4f})")
        print(f"   🎯 Test AUC:       {test_auc:.4f}")
        print(f"   📊 CV Accuracy:    {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"   📊 CV F1:          {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        # DIAGNÓSTICO DE OVERFITTING
        overfitting_score = 0
        warnings_overfitting = []
        
        if accuracy_gap > 0.1:
            overfitting_score += 3
            warnings_overfitting.append(f"Accuracy gap alto: {accuracy_gap:.4f}")
        elif accuracy_gap > 0.05:
            overfitting_score += 1
            warnings_overfitting.append(f"Accuracy gap moderado: {accuracy_gap:.4f}")
        
        if test_accuracy > 0.98:
            overfitting_score += 2
            warnings_overfitting.append(f"Test accuracy sospechosamente alto: {test_accuracy:.4f}")
        
        if cv_accuracy.std() < 0.01:
            overfitting_score += 1
            warnings_overfitting.append(f"CV std muy baja: {cv_accuracy.std():.4f}")
        
        # Diagnóstico final
        if overfitting_score >= 4:
            diagnostico_of = "🚨 OVERFITTING SEVERO"
            confianza = "BAJA"
        elif overfitting_score >= 2:
            diagnostico_of = "⚠️ POSIBLE OVERFITTING"
            confianza = "MEDIA"
        else:
            diagnostico_of = "✅ MODELO ROBUSTO"
            confianza = "ALTA"
        
        print(f"\n🩺 DIAGNÓSTICO OVERFITTING: {diagnostico_of}")
        print(f"   Puntuación: {overfitting_score}/6")
        print(f"   Confianza: {confianza}")
        
        if warnings_overfitting:
            print(f"   ⚠️ Advertencias:")
            for warning in warnings_overfitting:
                print(f"     - {warning}")
        
        # Preparar resultados
        resultados = {
            'contaminante': contaminante,
            'tipo': self.contaminantes_disponibles[contaminante]['tipo'],
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'test_f1': float(test_f1),
            'test_auc': float(test_auc),
            'accuracy_gap': float(accuracy_gap),
            'f1_gap': float(f1_gap),
            'cv_accuracy_mean': float(cv_accuracy.mean()),
            'cv_accuracy_std': float(cv_accuracy.std()),
            'cv_f1_mean': float(cv_f1.mean()),
            'cv_f1_std': float(cv_f1.std()),
            'overfitting_score': overfitting_score,
            'diagnostico_overfitting': diagnostico_of,
            'confianza': confianza,
            'warnings': warnings_overfitting,
            'n_features_usadas': X_train_selected.shape[1],
            'config_anti_overfitting': self.config_anti_overfitting,
            'fecha_entrenamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Guardar modelo
        modelo_info = {
            'modelo': mejor_svm,
            'scaler': scaler,
            'selector': selector,
            'resultados': resultados
        }
        
        modelo_path = os.path.join(self.output_dir, f"{contaminante}_svm_anti_overfitting.joblib")
        joblib.dump(modelo_info, modelo_path)
        
        # Guardar resultados JSON
        resultados_path = os.path.join(self.output_dir, f"{contaminante}_resultados_anti_overfitting.json")
        with open(resultados_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Modelo guardado: {modelo_path}")
        
        return resultados, modelo_info
    
    def entrenar_contaminante_robusto(self, contaminante, samples_per_class=None):
        """Entrena un contaminante con máxima robustez"""
        
        print(f"\n{'='*70}")
        print(f"🛡️ ENTRENAMIENTO ANTI-OVERFITTING: {contaminante.upper()}")
        print(f"{'='*70}")
        
        try:
            # 1. Cargar firma espectral
            df_firma = self.cargar_firma_espectral(contaminante)
            
            # 2. Crear dataset anti-overfitting
            X_raw, y, wavelengths = self.crear_dataset_anti_overfitting(df_firma, samples_per_class)
            
            # 3. Crear características robustas
            X_features, nombres_features = self.crear_features_robustas(X_raw, wavelengths)
            
            # 4. Entrenar SVM robusto
            resultados, modelo_info = self.entrenar_svm_robusto(
                X_features, y, nombres_features, contaminante
            )
            
            print(f"✅ ENTRENAMIENTO ANTI-OVERFITTING COMPLETADO: {contaminante}")
            
            return resultados, modelo_info
            
        except Exception as e:
            print(f"❌ Error entrenando {contaminante}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def entrenar_prioritarios_anti_overfitting(self):
        """Entrena contaminantes prioritarios con anti-overfitting"""
        
        prioritarios = [
            'Doc_Mg_L',
            'Nh4_Mg_L', 
            'Turbidity_Ntu',
            'Caffeine_Ng_L',
            'Acesulfame_Ng_L'
        ]
        
        print(f"🛡️ ENTRENAMIENTO ANTI-OVERFITTING - CONTAMINANTES PRIORITARIOS")
        print(f"📋 Lista: {prioritarios}")
        
        disponibles = [c for c in prioritarios if c in self.contaminantes_disponibles]
        print(f"✅ Disponibles: {disponibles}")
        
        resultados_todos = {}
        
        for i, contaminante in enumerate(disponibles, 1):
            print(f"\n[{i}/{len(disponibles)}] Procesando: {contaminante}")
            
            resultado, _ = self.entrenar_contaminante_robusto(contaminante)
            
            if resultado:
                resultados_todos[contaminante] = resultado
        
        # Generar reporte anti-overfitting
        self.generar_reporte_anti_overfitting(resultados_todos)
        
        return resultados_todos
    
    def generar_reporte_anti_overfitting(self, resultados_todos):
        """Genera reporte especializado en anti-overfitting"""
        
        print(f"\n{'='*70}")
        print(f"📊 REPORTE ANTI-OVERFITTING - RESULTADOS ROBUSTOS")
        print(f"{'='*70}")
        
        if not resultados_todos:
            print("❌ No hay resultados para reportar")
            return
        
        # Crear DataFrame
        datos_reporte = []
        for contaminante, resultado in resultados_todos.items():
            datos_reporte.append({
                'Contaminante': contaminante,
                'Tipo': resultado['tipo'],
                'Test_Accuracy': resultado['test_accuracy'],
                'Test_F1': resultado['test_f1'],
                'Test_AUC': resultado['test_auc'],
                'Accuracy_Gap': resultado['accuracy_gap'],
                'F1_Gap': resultado['f1_gap'],
                'CV_Accuracy_Mean': resultado['cv_accuracy_mean'],
                'CV_Accuracy_Std': resultado['cv_accuracy_std'],
                'Overfitting_Score': resultado['overfitting_score'],
                'Diagnostico': resultado['diagnostico_overfitting'],
                'Confianza': resultado['confianza'],
                'Features': resultado['n_features_usadas']
            })
        
        df_reporte = pd.DataFrame(datos_reporte)
        
        # Guardar reporte
        reporte_path = os.path.join(self.output_dir, "reporte_anti_overfitting.csv")
        df_reporte.to_csv(reporte_path, index=False, encoding='utf-8')
        
        # Estadísticas
        print(f"📈 MÉTRICAS ROBUSTAS (ANTI-OVERFITTING):")
        print(f"   Modelos entrenados: {len(resultados_todos)}")
        print(f"   Test Accuracy promedio: {df_reporte['Test_Accuracy'].mean():.4f}")
        print(f"   Test F1-Score promedio: {df_reporte['Test_F1'].mean():.4f}")
        print(f"   Test AUC promedio: {df_reporte['Test_AUC'].mean():.4f}")
        print(f"   Accuracy Gap promedio: {df_reporte['Accuracy_Gap'].mean():.4f}")
        
        # Análisis de confianza
        alta_confianza = df_reporte[df_reporte['Confianza'] == 'ALTA']
        media_confianza = df_reporte[df_reporte['Confianza'] == 'MEDIA']
        baja_confianza = df_reporte[df_reporte['Confianza'] == 'BAJA']
        
        print(f"\n🎯 ANÁLISIS DE CONFIANZA:")
        print(f"   Alta confianza: {len(alta_confianza)} modelos")
        print(f"   Media confianza: {len(media_confianza)} modelos")
        print(f"   Baja confianza: {len(baja_confianza)} modelos")
        
        if len(alta_confianza) > 0:
            print(f"\n✅ MODELOS DE ALTA CONFIANZA:")
            for _, modelo in alta_confianza.iterrows():
                print(f"   {modelo['Contaminante']:20} | Accuracy: {modelo['Test_Accuracy']:.4f} | Gap: {modelo['Accuracy_Gap']:+.4f}")
        
        # Recomendaciones finales
        accuracy_promedio = df_reporte['Test_Accuracy'].mean()
        gap_promedio = df_reporte['Accuracy_Gap'].mean()
        
        print(f"\n💡 EVALUACIÓN FINAL:")
        
        if accuracy_promedio >= 0.85 and gap_promedio <= 0.05:
            evaluacion = "🌟 EXCELENTE - Sistema robusto y confiable"
            meta_anteproyecto = f"Usar accuracy promedio: {accuracy_promedio:.4f} como resultado principal"
        elif accuracy_promedio >= 0.80 and gap_promedio <= 0.10:
            evaluacion = "✅ BUENO - Sistema confiable con pequeñas reservas"
            meta_anteproyecto = f"Reportar accuracy {accuracy_promedio:.4f} con mención de validación robusta"
        elif accuracy_promedio >= 0.75:
            evaluacion = "📊 ACEPTABLE - Sistema funcional requiere optimización"
            meta_anteproyecto = f"Objetivo conservador: ≥75% accuracy (logrado: {accuracy_promedio:.4f})"
        else:
            evaluacion = "⚠️ REQUIERE MEJORA - Reconsiderar metodología"
            meta_anteproyecto = "Ajustar enfoque antes de anteproyecto"
        
        print(f"   {evaluacion}")
        print(f"   Gap promedio: {gap_promedio:.4f}")
        print(f"   Recomendación para anteproyecto: {meta_anteproyecto}")
        
        print(f"\n📁 Reporte guardado: {reporte_path}")
        
        return df_reporte

# Función principal
def ejecutar_entrenamiento_anti_overfitting():
    """Función principal para entrenamiento anti-overfitting"""
    
    print(f"🛡️ ENTRENADOR SVM ANTI-OVERFITTING")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 OBJETIVO: Resultados robustos y confiables para anteproyecto")
    
    # Crear entrenador
    entrenador = SVMAntiOverfitting()
    
    # Entrenar con anti-overfitting
    resultados = entrenador.entrenar_prioritarios_anti_overfitting()
    
    print(f"\n✅ ENTRENAMIENTO ANTI-OVERFITTING COMPLETADO")
    print(f"📊 Modelos robustos entrenados: {len(resultados)}")
    print(f"📁 Revisa 'modelos_svm_anti_overfitting' para resultados")
    
    return resultados

if __name__ == "__main__":
    resultados = ejecutar_entrenamiento_anti_overfitting()