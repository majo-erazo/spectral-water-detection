# entrenar_svm_firmas_espectrales.py
# Entrena modelos SVM usando las firmas espectrales generadas desde CSV

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

class SVMFirmasEspectrales:
    """Entrena modelos SVM usando las firmas espectrales generadas"""
    
    def __init__(self, firmas_dir="todo/firmas_espectrales_csv", output_dir="modelos_svm_firmas"):
        self.firmas_dir = firmas_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Contaminantes disponibles con sus archivos
        self.contaminantes_disponibles = self.detectar_contaminantes_disponibles()
        
        print(f"🎯 SVMFirmasEspectrales inicializado")
        print(f"📁 Directorio firmas: {firmas_dir}")
        print(f"📁 Directorio modelos: {output_dir}")
        print(f"🧪 Contaminantes detectados: {len(self.contaminantes_disponibles)}")
    
    def detectar_contaminantes_disponibles(self):
        """Detecta automáticamente qué contaminantes tienen firmas espectrales"""
        contaminantes = {}
        
        if not os.path.exists(self.firmas_dir):
            print(f"❌ No se encontró directorio: {self.firmas_dir}")
            return contaminantes
        
        # Buscar archivos CSV de datos espectrales
        for archivo in os.listdir(self.firmas_dir):
            if archivo.endswith("_datos_espectrales.csv"):
                # Extraer nombre del contaminante
                nombre_base = archivo.replace("_datos_espectrales.csv", "")
                ruta_completa = os.path.join(self.firmas_dir, archivo)
                
                # Verificar que el archivo tenga datos válidos
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
            raise ValueError(f"Contaminante {contaminante} no disponible. Disponibles: {list(self.contaminantes_disponibles.keys())}")
        
        archivo_datos = self.contaminantes_disponibles[contaminante]['archivo_datos']
        
        print(f"📊 Cargando firma espectral: {contaminante}")
        print(f"   📁 Archivo: {archivo_datos}")
        
        # Cargar datos espectrales
        df_firma = pd.read_csv(archivo_datos)
        
        print(f"   📈 Longitudes de onda: {len(df_firma)}")
        print(f"   🌊 Rango espectral: {df_firma['wavelength'].min()}-{df_firma['wavelength'].max()} nm")
        
        return df_firma
    
    
    def crear_dataset_desde_firma(self, df_firma, n_samples_por_clase=200, anti_overfitting=True):
        generador = DatasetAntiOverfitting(df_firma)
        X, y = generador.generar_dataset_realista(n_samples_por_clase, anti_overfitting)
        return X, y, generador.wavelengths
    
    def crear_features_espectrales_avanzadas(self, X, wavelengths):
        """Crea características espectrales avanzadas desde el espectro bruto"""
        print(f"🔧 Creando features espectrales avanzadas...")
        
        features_avanzadas = []
        nombres_features = []
        
        for i in range(X.shape[0]):
            espectro = X[i, :]
            features_muestra = []
            
            # 1. Estadísticas básicas
            features_muestra.extend([
                np.mean(espectro),           # Media
                np.std(espectro),            # Desviación estándar  
                np.max(espectro),            # Máximo
                np.min(espectro),            # Mínimo
                np.max(espectro) - np.min(espectro),  # Rango
                np.median(espectro)          # Mediana
            ])
            
            if i == 0:  # Solo añadir nombres una vez
                nombres_features.extend(['mean', 'std', 'max', 'min', 'range', 'median'])
            
            # 2. Momentos estadísticos
            from scipy import stats
            features_muestra.extend([
                stats.skew(espectro),        # Asimetría
                stats.kurtosis(espectro)     # Curtosis
            ])
            
            if i == 0:
                nombres_features.extend(['skewness', 'kurtosis'])
            
            # 3. Características espectrales específicas
            # Pendiente general
            x_vals = np.arange(len(espectro))
            slope, intercept = np.polyfit(x_vals, espectro, 1)
            features_muestra.append(slope)
            
            if i == 0:
                nombres_features.append('slope_general')
            
            # 4. Características por regiones espectrales
            # Dividir espectro en 3 regiones: UV-VIS (400-550nm), VIS (550-700nm), NIR (700-800nm)
            n_wavelengths = len(wavelengths)
            region_1 = espectro[:n_wavelengths//3]      # Primera región
            region_2 = espectro[n_wavelengths//3:2*n_wavelengths//3]  # Segunda región  
            region_3 = espectro[2*n_wavelengths//3:]    # Tercera región
            
            for j, region in enumerate([region_1, region_2, region_3], 1):
                if len(region) > 0:
                    features_muestra.extend([
                        np.mean(region),
                        np.std(region),
                        np.max(region)
                    ])
                    
                    if i == 0:
                        nombres_features.extend([f'region{j}_mean', f'region{j}_std', f'region{j}_max'])
            
            # 5. Ratios entre regiones
            if len(region_1) > 0 and len(region_2) > 0 and len(region_3) > 0:
                features_muestra.extend([
                    np.mean(region_2) / (np.mean(region_1) + 1e-8),  # VIS/UV
                    np.mean(region_3) / (np.mean(region_2) + 1e-8),  # NIR/VIS
                    np.mean(region_3) / (np.mean(region_1) + 1e-8)   # NIR/UV
                ])
                
                if i == 0:
                    nombres_features.extend(['ratio_vis_uv', 'ratio_nir_vis', 'ratio_nir_uv'])
            
            # 6. Área bajo la curva
            area_total = np.trapz(espectro)
            features_muestra.append(area_total)
            
            if i == 0:
                nombres_features.append('area_total')
            
            # 7. Detección de picos
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(espectro, height=np.mean(espectro))
            features_muestra.extend([
                len(peaks),                   # Número de picos
                np.max(espectro[peaks]) if len(peaks) > 0 else 0,  # Altura max pico
                np.mean(espectro[peaks]) if len(peaks) > 0 else 0  # Altura media picos
            ])
            
            if i == 0:
                nombres_features.extend(['n_peaks', 'peak_max_height', 'peak_mean_height'])
            
            features_avanzadas.append(features_muestra)
        
        X_features = np.array(features_avanzadas)
        
        print(f"   ✅ Features creadas: {X_features.shape[1]} características avanzadas")
        print(f"   📝 Features principales: {nombres_features[:10]}")
        
        return X_features, nombres_features
    
    def entrenar_svm_optimizado(self, X, y, nombres_features, contaminante, estrategia='auto'):
        """
        Entrena SVM optimizado para un contaminante específico
        
        Args:
            X: Características de entrada
            y: Etiquetas (0=baja, 1=alta concentración)
            nombres_features: Nombres de las características
            contaminante: Nombre del contaminante
            estrategia: 'conservadora', 'moderada', 'agresiva', 'auto'
        """
        print(f"\n🤖 Entrenando SVM optimizado para: {contaminante}")
        print(f"   📊 Dataset: {X.shape[0]} muestras x {X.shape[1]} features")
        print(f"   ⚖️ Estrategia: {estrategia}")
        
        # Determinar estrategia automáticamente
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
        
        print(f"   📊 División: {len(X_train)} train, {len(X_test)} test")
        
        # Normalización
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Selección de características según estrategia
        if estrategia == 'conservadora':
            # Seleccionar top 10 features
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
            # Seleccionar top 15 features
            selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            cv_folds = 5
            
        else:  # agresiva
            # Usar todas las features o RFE
            if X.shape[1] > 20:
                # Usar RFE para seleccionar mejores 20
                base_svm = SVC(kernel='rbf', random_state=42)
                selector = RFE(estimator=base_svm, n_features_to_select=20, step=2)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)
            else:
                selector = None
                X_train_selected = X_train_scaled
                X_test_selected = X_test_scaled
            
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            cv_folds = 5
        
        print(f"   🔧 Features seleccionadas: {X_train_selected.shape[1]}")
        print(f"   🎛️ Parámetros a probar: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
        
        # Grid Search con validación cruzada
        svm_base = SVC(random_state=42, probability=True)
        
        grid_search = GridSearchCV(
            svm_base, param_grid, 
            cv=cv_folds, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        print(f"   🔍 Ejecutando Grid Search...")
        grid_search.fit(X_train_selected, y_train)
        
        print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"   📈 Mejor score CV: {grid_search.best_score_:.4f}")
        
        # Modelo final
        mejor_svm = grid_search.best_estimator_
        
        # Evaluación en test
        y_pred = mejor_svm.predict(X_test_selected)
        y_pred_proba = mejor_svm.predict_proba(X_test_selected)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 RESULTADOS FINALES:")
        print(f"   🎯 Accuracy: {accuracy:.4f}")
        print(f"   🎯 F1-Score: {f1:.4f}")
        print(f"   🎯 AUC: {auc:.4f}")
        
        # Validación cruzada adicional
        cv_scores = cross_val_score(mejor_svm, X_train_selected, y_train, cv=5, scoring='f1')
        print(f"   📊 CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Análisis de overfitting
        train_score = mejor_svm.score(X_train_selected, y_train)
        test_score = accuracy
        gap = train_score - test_score
        
        if gap > 0.15:
            overfitting_nivel = "ALTO"
        elif gap > 0.08:
            overfitting_nivel = "MODERADO"
        elif gap > 0.03:
            overfitting_nivel = "BAJO"
        else:
            overfitting_nivel = "MÍNIMO"
        
        print(f"   ⚠️ Overfitting: {overfitting_nivel} (gap: {gap:.4f})")
        
        # Guardar modelo y resultados
        resultados = {
            'contaminante': contaminante,
            'estrategia': estrategia,
            'mejores_parametros': grid_search.best_params_,
            'mejor_score_cv': float(grid_search.best_score_),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'auc_score': float(auc),
            'cv_scores': cv_scores.tolist(),
            'overfitting_nivel': overfitting_nivel,
            'gap_train_test': float(gap),
            'n_features_usadas': X_train_selected.shape[1],
            'features_seleccionadas': [nombres_features[i] for i in selector.get_support(indices=True)] if selector else nombres_features,
            'fecha_entrenamiento': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Guardar modelo
        modelo_info = {
            'modelo': mejor_svm,
            'scaler': scaler,
            'selector': selector,
            'resultados': resultados
        }
        
        modelo_path = os.path.join(self.output_dir, f"{contaminante}_svm_firma_espectral.joblib")
        joblib.dump(modelo_info, modelo_path)
        
        # Guardar resultados JSON
        resultados_path = os.path.join(self.output_dir, f"{contaminante}_resultados.json")
        with open(resultados_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        
        # Generar visualizaciones
        self.generar_visualizaciones(mejor_svm, X_test_selected, y_test, y_pred, y_pred_proba, contaminante)
        
        print(f"   💾 Modelo guardado: {modelo_path}")
        print(f"   📄 Resultados guardados: {resultados_path}")
        
        return resultados, modelo_info
    
    def generar_visualizaciones(self, modelo, X_test, y_test, y_pred, y_pred_proba, contaminante):
        """Genera visualizaciones del modelo entrenado"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0,0].set_title(f'Matriz de Confusión - {contaminante}')
        
        # Añadir números a la matriz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0,0].text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        axes[0,0].set_xlabel('Predicción')
        axes[0,0].set_ylabel('Valor Real')
        
        # 2. Distribución de probabilidades
        axes[0,1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Baja concentración', color='blue')
        axes[0,1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Alta concentración', color='red')
        axes[0,1].set_title('Distribución de Probabilidades')
        axes[0,1].set_xlabel('Probabilidad de Alta Concentración')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
        
        # 3. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        axes[1,0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        axes[1,0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1,0].set_title('Curva ROC')
        axes[1,0].set_xlabel('Tasa de Falsos Positivos')
        axes[1,0].set_ylabel('Tasa de Verdaderos Positivos')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Importancia de características (si es SVM lineal)
        if hasattr(modelo, 'coef_') and modelo.coef_ is not None:
            coef_abs = np.abs(modelo.coef_[0])
            indices = np.argsort(coef_abs)[-10:]  # Top 10
            
            axes[1,1].barh(range(len(indices)), coef_abs[indices])
            axes[1,1].set_title('Importancia de Características (Top 10)')
            axes[1,1].set_xlabel('|Coeficiente|')
            axes[1,1].set_ylabel('Característica')
        else:
            # Para SVM no lineal, mostrar distribución de vectores de soporte
            n_support = modelo.n_support_
            axes[1,1].bar(['Baja Conc.', 'Alta Conc.'], n_support)
            axes[1,1].set_title('Vectores de Soporte por Clase')
            axes[1,1].set_ylabel('Número de Vectores de Soporte')
        
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = os.path.join(self.output_dir, f"{contaminante}_visualizaciones.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Visualizaciones guardadas: {viz_path}")
    
    def entrenar_contaminante(self, contaminante, n_samples=150, estrategia='auto'):
        """Entrena SVM para un contaminante específico usando su firma espectral"""
        
        print(f"\n{'='*70}")
        print(f"🧪 ENTRENANDO SVM PARA: {contaminante.upper()}")
        print(f"{'='*70}")
        
        try:
            # 1. Cargar firma espectral
            df_firma = self.cargar_firma_espectral(contaminante)
            
            # 2. Crear dataset sintético
            X_raw, y, wavelengths = self.crear_dataset_desde_firma(df_firma, n_samples_por_clase=n_samples//2)
            
            # 3. Crear características avanzadas
            X_features, nombres_features = self.crear_features_espectrales_avanzadas(X_raw, wavelengths)
            
            # 4. Entrenar SVM
            resultados, modelo_info = self.entrenar_svm_optimizado(
                X_features, y, nombres_features, contaminante, estrategia
            )
            
            print(f"✅ ENTRENAMIENTO COMPLETADO PARA {contaminante}")
            
            return resultados, modelo_info
            
        except Exception as e:
            print(f"❌ Error entrenando {contaminante}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def entrenar_todos_disponibles(self, n_samples=150, estrategia='auto', max_contaminantes=None):
        """Entrena SVM para todos los contaminantes disponibles"""
        
        print(f"\n{'='*80}")
        print(f"🚀 ENTRENANDO SVM PARA TODOS LOS CONTAMINANTES DISPONIBLES")
        print(f"{'='*80}")
        print(f"📊 Contaminantes detectados: {len(self.contaminantes_disponibles)}")
        
        if max_contaminantes:
            contaminantes_a_procesar = list(self.contaminantes_disponibles.keys())[:max_contaminantes]
            print(f"🎯 Procesando primeros {max_contaminantes} contaminantes")
        else:
            contaminantes_a_procesar = list(self.contaminantes_disponibles.keys())
        
        resultados_todos = {}
        
        for i, contaminante in enumerate(contaminantes_a_procesar, 1):
            print(f"\n[{i}/{len(contaminantes_a_procesar)}] Procesando: {contaminante}")
            
            resultados, modelo_info = self.entrenar_contaminante(contaminante, n_samples, estrategia)
            
            if resultados:
                resultados_todos[contaminante] = resultados
        
        # Generar reporte consolidado
        self.generar_reporte_consolidado(resultados_todos)
        
        return resultados_todos
    
    def generar_reporte_consolidado(self, resultados_todos):
        """Genera reporte consolidado de todos los entrenamientos"""
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE CONSOLIDADO DE ENTRENAMIENTOS SVM")
        print(f"{'='*80}")
        
        if not resultados_todos:
            print("❌ No hay resultados para reportar")
            return
        
        # Crear DataFrame con resultados
        datos_reporte = []
        for contaminante, resultado in resultados_todos.items():
            tipo = self.contaminantes_disponibles[contaminante]['tipo']
            datos_reporte.append({
                'Contaminante': contaminante,
                'Tipo': tipo,
                'Accuracy': resultado['accuracy'],
                'F1-Score': resultado['f1_score'],
                'AUC': resultado['auc_score'],
                'Overfitting': resultado['overfitting_nivel'],
                'Gap': resultado['gap_train_test'],
                'Features': resultado['n_features_usadas'],
                'Estrategia': resultado['estrategia']
            })
        
        df_reporte = pd.DataFrame(datos_reporte)
        
        # Guardar reporte
        reporte_path = os.path.join(self.output_dir, "reporte_consolidado_svm_firmas.csv")
        df_reporte.to_csv(reporte_path, index=False, encoding='utf-8')
        
        # Estadísticas por tipo
        print(f"📈 ESTADÍSTICAS GENERALES:")
        print(f"   Modelos entrenados: {len(resultados_todos)}")
        print(f"   Accuracy promedio: {df_reporte['Accuracy'].mean():.4f}")
        print(f"   F1-Score promedio: {df_reporte['F1-Score'].mean():.4f}")
        print(f"   AUC promedio: {df_reporte['AUC'].mean():.4f}")
        
        # Estadísticas por tipo
        for tipo in ['inorganic', 'organic']:
            df_tipo = df_reporte[df_reporte['Tipo'] == tipo]
            if len(df_tipo) > 0:
                print(f"\n🧪 CONTAMINANTES {tipo.upper()}:")
                print(f"   Cantidad: {len(df_tipo)}")
                print(f"   Accuracy promedio: {df_tipo['Accuracy'].mean():.4f}")
                print(f"   F1-Score promedio: {df_tipo['F1-Score'].mean():.4f}")
        
        # Top 5 mejores modelos
        top_5 = df_reporte.nlargest(5, 'F1-Score')
        print(f"\n🏆 TOP 5 MEJORES MODELOS (por F1-Score):")
        for _, row in top_5.iterrows():
            print(f"   {row['Contaminante']:20} | F1: {row['F1-Score']:.4f} | Accuracy: {row['Accuracy']:.4f} | {row['Tipo']}")
        
        # Casos con posible overfitting
        overfitting_casos = df_reporte[df_reporte['Overfitting'].isin(['ALTO', 'MODERADO'])]
        if len(overfitting_casos) > 0:
            print(f"\n⚠️ CASOS CON POSIBLE OVERFITTING:")
            for _, row in overfitting_casos.iterrows():
                print(f"   {row['Contaminante']:20} | Nivel: {row['Overfitting']} | Gap: {row['Gap']:.4f}")
        
        print(f"\n📁 Reporte guardado en: {reporte_path}")
        
        return df_reporte

# Funciones de conveniencia
def entrenar_svm_contaminante_especifico(contaminante, n_samples=150, estrategia='auto'):
    """Función de conveniencia para entrenar un contaminante específico"""
    entrenador = SVMFirmasEspectrales()
    return entrenador.entrenar_contaminante(contaminante, n_samples, estrategia)

def entrenar_svm_todos_los_contaminantes(n_samples=150, estrategia='auto', max_contaminantes=None):
    """Función de conveniencia para entrenar todos los contaminantes"""
    entrenador = SVMFirmasEspectrales()
    return entrenador.entrenar_todos_disponibles(n_samples, estrategia, max_contaminantes)

def entrenar_svm_prioritarios():
    """Entrena SVM para los contaminantes prioritarios (mejores firmas)"""
    
    # Contaminantes prioritarios basados en el reporte
    prioritarios = [
        'Doc_Mg_L',           # Inorgánico - excelente
        'Nh4_Mg_L',           # Inorgánico - excelente  
        'Turbidity_Ntu',      # Inorgánico - excelente
        'Caffeine_Ng_L',      # Orgánico - muy bueno
        'Acesulfame_Ng_L',    # Orgánico - muy bueno
        'Diclofenac_Ng_L'     # Orgánico - muy bueno
    ]
    
    print(f"🎯 ENTRENANDO SVM PARA CONTAMINANTES PRIORITARIOS")
    print(f"📋 Lista: {prioritarios}")
    
    entrenador = SVMFirmasEspectrales()
    
    # Verificar cuáles están disponibles
    disponibles = [c for c in prioritarios if c in entrenador.contaminantes_disponibles]
    no_disponibles = [c for c in prioritarios if c not in entrenador.contaminantes_disponibles]
    
    if no_disponibles:
        print(f"⚠️ No disponibles: {no_disponibles}")
    
    print(f"✅ Disponibles para entrenar: {disponibles}")
    
    resultados = {}
    for contaminante in disponibles:
        print(f"\n🧪 Entrenando: {contaminante}")
        resultado, _ = entrenador.entrenar_contaminante(contaminante, n_samples=200, estrategia='moderada')
        if resultado:
            resultados[contaminante] = resultado
    
    # Generar reporte específico
    entrenador.generar_reporte_consolidado(resultados)
    
    return resultados

# Ejecutar si es script principal
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar SVM usando firmas espectrales')
    parser.add_argument('--contaminante', type=str, help='Contaminante específico a entrenar')
    parser.add_argument('--todos', action='store_true', help='Entrenar todos los disponibles')
    parser.add_argument('--prioritarios', action='store_true', help='Entrenar solo prioritarios')
    parser.add_argument('--max', type=int, help='Máximo número de contaminantes (para pruebas)')
    parser.add_argument('--samples', type=int, default=150, help='Número de muestras sintéticas por clase')
    parser.add_argument('--estrategia', choices=['conservadora', 'moderada', 'agresiva', 'auto'], 
                       default='auto', help='Estrategia de entrenamiento')
    
    args = parser.parse_args()
    
    print(f"🌊 ENTRENADOR SVM CON FIRMAS ESPECTRALES")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Parámetros: muestras={args.samples}, estrategia={args.estrategia}")
    
    if args.prioritarios:
        # Entrenar solo prioritarios
        resultados = entrenar_svm_prioritarios()
        
    elif args.contaminante:
        # Entrenar contaminante específico
        resultados, modelo = entrenar_svm_contaminante_especifico(
            args.contaminante, args.samples, args.estrategia
        )
        
    elif args.todos:
        # Entrenar todos
        resultados = entrenar_svm_todos_los_contaminantes(
            args.samples, args.estrategia, args.max
        )
        
    else:
        # Modo interactivo
        print(f"\n🎛️ MODO INTERACTIVO")
        print(f"Opciones:")
        print(f"  1. Entrenar contaminantes prioritarios")
        print(f"  2. Entrenar todos los disponibles") 
        print(f"  3. Entrenar contaminante específico")
        
        opcion = input(f"\nSelecciona opción (1-3): ").strip()
        
        if opcion == '1':
            resultados = entrenar_svm_prioritarios()
        elif opcion == '2':
            resultados = entrenar_svm_todos_los_contaminantes()
        elif opcion == '3':
            # Mostrar disponibles
            entrenador = SVMFirmasEspectrales()
            print(f"\nContaminantes disponibles:")
            for i, cont in enumerate(entrenador.contaminantes_disponibles.keys(), 1):
                print(f"  {i}. {cont}")
            
            seleccion = input(f"\nEscribe el nombre del contaminante: ").strip()
            if seleccion in entrenador.contaminantes_disponibles:
                resultados, modelo = entrenar_svm_contaminante_especifico(seleccion)
            else:
                print(f"❌ Contaminante '{seleccion}' no encontrado")
        else:
            print(f"❌ Opción no válida")
    
    print(f"\n✅ PROCESO COMPLETADO")
    print(f"📁 Revisa la carpeta 'modelos_svm_firmas' para los resultados")