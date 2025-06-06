# sistema_hibrido_adaptativo.py
# Sistema que elige el modelo más apropiado según tamaño del dataset
# María José Erazo González - UDP

import os
import pandas as pd
import numpy as np
import json
import datetime
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from numpy import trapz

class SistemaHibridoAdaptativo:
    """
    Sistema híbrido que selecciona automáticamente el mejor modelo
    según el tamaño del dataset y tipo de contaminante
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.results_dir = "resultados_hibrido_adaptativo"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Mapeo estándar del proyecto
        self.mapeo_carpetas = {
            'Nh4_Mg_L': 'Nh4', 
            'Caffeine_Ng_L': 'Caffeine',
            'Turbidity_Ntu': 'Turbidity',
            'Doc_Mg_L': 'Doc',
            'Acesulfame_Ng_L': 'Acesulfame'
        }
        
        # Configuración de modelos por tamaño de dataset
        self.configuracion_modelos = {
            'extremo': {  # ≤ 4 muestras
                'modelo_primario': 'logistic_regression',
                'modelo_alternativo': 'svm_linear',
                'descripcion': 'Dataset extremo - Modelo lineal simple'
            },
            'pequeno': {  # 5-10 muestras
                'modelo_primario': 'random_forest',
                'modelo_alternativo': 'svm_rbf', 
                'descripcion': 'Dataset pequeño - Ensemble simple'
            },
            'mediano': {  # 11-50 muestras
                'modelo_primario': 'xgboost_conservador',
                'modelo_alternativo': 'random_forest',
                'descripcion': 'Dataset mediano - XGBoost conservador'
            },
            'grande': {  # >50 muestras
                'modelo_primario': 'xgboost_completo',
                'modelo_alternativo': 'xgboost_conservador',
                'descripcion': 'Dataset grande - XGBoost completo'
            }
        }
    
    def determinar_categoria_dataset(self, n_muestras):
        """Determina la categoría del dataset según número de muestras"""
        if n_muestras <= 4:
            return 'extremo'
        elif n_muestras <= 10:
            return 'pequeno'
        elif n_muestras <= 50:
            return 'mediano'
        else:
            return 'grande'
    
    def entrenar_hibrido_adaptativo(self, contaminante):
        """
        Entrenamiento híbrido que selecciona automáticamente el mejor modelo
        """
        print(f"\n{'='*70}")
        print(f"🔄 SISTEMA HÍBRIDO ADAPTATIVO")
        print(f"📋 Contaminante: {contaminante}")
        print(f"{'='*70}")
        
        inicio_tiempo = datetime.datetime.now()
        
        try:
            # 1. Preparar datos usando lógica exitosa del diagnóstico
            X, y, feature_names = self.preparar_datos_optimizados(contaminante)
            
            n_muestras = len(X)
            categoria = self.determinar_categoria_dataset(n_muestras)
            config = self.configuracion_modelos[categoria]
            
            print(f"📊 Dataset: {n_muestras} muestras → Categoría: {categoria}")
            print(f"🎯 Estrategia: {config['descripcion']}")
            print(f"🔧 Modelo primario: {config['modelo_primario']}")
            
            # 2. Entrenar modelo primario
            resultado_primario = self.entrenar_modelo_especifico(
                X, y, feature_names, config['modelo_primario'], contaminante
            )
            
            # 3. Si el primario falla, probar alternativo
            if not resultado_primario or resultado_primario['test_f1'] < 0.1:
                print(f"⚠️ Modelo primario insuficiente, probando alternativo...")
                resultado_alternativo = self.entrenar_modelo_especifico(
                    X, y, feature_names, config['modelo_alternativo'], contaminante
                )
                
                # Elegir el mejor resultado
                if resultado_alternativo and resultado_alternativo['test_f1'] > resultado_primario['test_f1']:
                    resultado_final = resultado_alternativo
                    resultado_final['modelo_usado'] = config['modelo_alternativo']
                    resultado_final['modelo_fue_alternativo'] = True
                else:
                    resultado_final = resultado_primario
                    resultado_final['modelo_usado'] = config['modelo_primario']
                    resultado_final['modelo_fue_alternativo'] = False
            else:
                resultado_final = resultado_primario
                resultado_final['modelo_usado'] = config['modelo_primario']
                resultado_final['modelo_fue_alternativo'] = False
            
            # 4. Agregar metadatos
            fin_tiempo = datetime.datetime.now()
            tiempo_total = (fin_tiempo - inicio_tiempo).total_seconds()
            
            resultado_final.update({
                'contaminante': contaminante,
                'metodo': 'sistema_hibrido_adaptativo',
                'categoria_dataset': categoria,
                'n_muestras_total': n_muestras,
                'tiempo_entrenamiento': tiempo_total,
                'features_utilizados': feature_names,
                'estrategia_usada': config['descripcion']
            })
            
            # 5. Mostrar y guardar resultados
            self._mostrar_resultados_hibridos(resultado_final)
            self._guardar_resultados_hibridos(contaminante, resultado_final)
            
            return resultado_final
            
        except Exception as e:
            print(f"❌ Error en sistema híbrido: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def preparar_datos_optimizados(self, contaminante):
        """
        Prepara datos usando la lógica exitosa del diagnóstico
        """
        print(f"📊 Preparando datos optimizados para {contaminante}...")
        
        # 1. Cargar datos crudos (misma lógica que diagnóstico exitoso)
        datos_espectrales = self.cargar_datos_crudos(contaminante)
        
        # 2. Extraer features (misma lógica que diagnóstico exitoso)  
        features = self.extraer_features_robustos(datos_espectrales)
        
        # 3. Crear dataset (misma lógica que diagnóstico exitoso)
        dataset = self.crear_dataset_robusto(features)
        
        # 4. Preparar para entrenamiento
        feature_columns = [col for col in dataset.columns if col != 'label']
        X = dataset[feature_columns].values
        y = dataset['label'].values
        
        print(f"   ✅ Dataset preparado: {X.shape}, Features: {len(feature_columns)}")
        
        return X, y, feature_columns
    
    def cargar_datos_crudos(self, contaminante):
        """Carga datos crudos (misma lógica exitosa del diagnóstico)"""
        carpeta = self.mapeo_carpetas[contaminante]
        ruta_carpeta = os.path.join(self.directorio_base, carpeta)
        
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) 
                              if f.endswith('_datos_espectrales.csv')]
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        datos = pd.read_csv(ruta_archivo)
        datos = datos.dropna().sort_values('wavelength').reset_index(drop=True)
        
        return datos
    
    def extraer_features_robustos(self, datos):
        """Extrae features robustos (misma lógica exitosa del diagnóstico)"""
        
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        features = {}
        
        # Features estadísticos básicos (que funcionaron en diagnóstico)
        for concentracion, response in [('high', high_response), ('low', low_response)]:
            features[f'{concentracion}_mean'] = np.mean(response)
            features[f'{concentracion}_std'] = np.std(response)
            features[f'{concentracion}_max'] = np.max(response)
            features[f'{concentracion}_min'] = np.min(response)
            features[f'{concentracion}_range'] = np.ptp(response)
            features[f'{concentracion}_auc'] = trapz(response, wavelengths)
            
            if len(wavelengths) > 1:
                slope, _ = np.polyfit(wavelengths, response, 1)
                features[f'{concentracion}_slope'] = slope
        
        # Features comparativos (los más discriminativos según diagnóstico)
        features['ratio_mean'] = features['high_mean'] / (features['low_mean'] + 1e-8)
        features['diff_mean'] = features['high_mean'] - features['low_mean']
        features['ratio_auc'] = features['high_auc'] / (features['low_auc'] + 1e-8)
        features['ratio_max'] = features['high_max'] / (features['low_max'] + 1e-8)
        
        return features
    
    def crear_dataset_robusto(self, features):
        """Crea dataset robusto (misma lógica exitosa del diagnóstico)"""
        
        # Muestra alta concentración (original)
        muestra_alta = list(features.values())
        
        # Muestra baja concentración (invertida - lógica que funcionó)
        muestra_baja = []
        for nombre, valor in features.items():
            if nombre.startswith('high_'):
                low_nombre = nombre.replace('high_', 'low_')
                if low_nombre in features:
                    muestra_baja.append(features[low_nombre])
                else:
                    muestra_baja.append(valor * 0.5)
            elif nombre.startswith('low_'):
                high_nombre = nombre.replace('low_', 'high_')
                if high_nombre in features:
                    muestra_baja.append(features[high_nombre])
                else:
                    muestra_baja.append(valor * 1.5)
            elif 'ratio' in nombre:
                muestra_baja.append(1 / (valor + 1e-8))
            elif 'diff' in nombre:
                muestra_baja.append(-valor)
            else:
                muestra_baja.append(valor)
        
        # Agregar algunas muestras con variabilidad controlada
        samples = [muestra_alta, muestra_baja]
        labels = [1, 0]
        
        # Agregar 2-4 muestras adicionales con noise conservador
        for i in range(2):
            # Alta concentración con noise
            noise_alta = [val * np.random.normal(1.0, 0.1) for val in muestra_alta]
            samples.append(noise_alta)
            labels.append(1)
            
            # Baja concentración con noise
            noise_baja = [val * np.random.normal(1.0, 0.1) for val in muestra_baja]
            samples.append(noise_baja)
            labels.append(0)
        
        # Crear DataFrame
        feature_names = list(features.keys())
        df = pd.DataFrame(samples, columns=feature_names)
        df['label'] = labels
        
        return df
    
    def entrenar_modelo_especifico(self, X, y, feature_names, tipo_modelo, contaminante):
        """
        Entrena un modelo específico según el tipo
        """
        print(f"🔧 Entrenando modelo: {tipo_modelo}")
        
        try:
            # Escalado de datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # División train/test (si hay suficientes datos)
            if len(X) >= 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
            else:
                # Para datasets extremos, train=test
                X_train = X_test = X_scaled
                y_train = y_test = y
            
            # Crear y entrenar modelo según tipo
            if tipo_modelo == 'logistic_regression':
                modelo = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
                
            elif tipo_modelo == 'svm_linear':
                modelo = SVC(kernel='linear', random_state=42, C=0.1, probability=True)
                
            elif tipo_modelo == 'svm_rbf':
                modelo = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale', probability=True)
                
            elif tipo_modelo == 'random_forest':
                modelo = RandomForestClassifier(
                    n_estimators=10, max_depth=3, random_state=42,
                    min_samples_split=2, min_samples_leaf=1
                )
                
            elif tipo_modelo == 'xgboost_conservador':
                modelo = xgb.XGBClassifier(
                    n_estimators=5, max_depth=2, learning_rate=0.1,
                    reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbosity=0
                )
                
            elif tipo_modelo == 'xgboost_completo':
                modelo = xgb.XGBClassifier(
                    n_estimators=20, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbosity=0
                )
            
            # Entrenar modelo
            modelo.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = modelo.predict(X_train)
            y_test_pred = modelo.predict(X_test)
            
            # Métricas
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            # AUC si es posible
            try:
                if hasattr(modelo, 'predict_proba'):
                    y_test_proba = modelo.predict_proba(X_test)[:, 1]
                    if len(set(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_test_proba)
                    else:
                        auc = 0.5
                else:
                    auc = 0.5
            except:
                auc = 0.5
            
            # Feature importance si está disponible
            feature_importance = {}
            if hasattr(modelo, 'feature_importances_'):
                for i, imp in enumerate(modelo.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(imp)
            elif hasattr(modelo, 'coef_'):
                coef = modelo.coef_[0] if len(modelo.coef_.shape) > 1 else modelo.coef_
                for i, imp in enumerate(np.abs(coef)):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(imp)
            
            # Diagnóstico de overfitting
            gap_f1 = train_f1 - test_f1
            gap_acc = train_acc - test_acc
            
            if gap_f1 > 0.2:
                diagnostico = "OVERFITTING_SEVERO"
            elif gap_f1 > 0.1:
                diagnostico = "OVERFITTING_MODERADO"
            elif gap_f1 > 0.05:
                diagnostico = "LEVE_OVERFITTING"
            else:
                diagnostico = "ROBUSTO"
            
            resultado = {
                'tipo_modelo': tipo_modelo,
                'test_accuracy': float(test_acc),
                'test_f1': float(test_f1),
                'train_accuracy': float(train_acc),
                'train_f1': float(train_f1),
                'gap_accuracy': float(gap_acc),
                'gap_f1': float(gap_f1),
                'auc': float(auc),
                'diagnostico_overfitting': diagnostico,
                'feature_importance': feature_importance,
                'n_muestras_train': len(X_train),
                'n_muestras_test': len(X_test),
                'exito': True
            }
            
            print(f"   ✅ {tipo_modelo}: F1={test_f1:.3f}, Acc={test_acc:.3f}, Gap={gap_f1:+.3f}")
            
            return resultado
            
        except Exception as e:
            print(f"   ❌ Error en {tipo_modelo}: {str(e)}")
            return {'tipo_modelo': tipo_modelo, 'test_f1': 0.0, 'exito': False, 'error': str(e)}
    
    def _mostrar_resultados_hibridos(self, resultado):
        """Muestra resultados del sistema híbrido"""
        
        print(f"\n{'='*70}")
        print(f"🔄 RESULTADOS SISTEMA HÍBRIDO")
        print(f"{'='*70}")
        
        print(f"📋 Contaminante: {resultado['contaminante']}")
        print(f"📊 Categoría dataset: {resultado['categoria_dataset']} ({resultado['n_muestras_total']} muestras)")
        print(f"🔧 Modelo usado: {resultado['modelo_usado']}")
        print(f"🎯 Estrategia: {resultado['estrategia_usada']}")
        
        if resultado.get('modelo_fue_alternativo'):
            print(f"⚠️ Se usó modelo alternativo (primario insuficiente)")
        
        print(f"\n📊 MÉTRICAS FINALES:")
        print(f"   🎯 Test F1:       {resultado['test_f1']:.4f}")
        print(f"   🎯 Test Accuracy: {resultado['test_accuracy']:.4f}")
        print(f"   🎯 AUC:           {resultado['auc']:.4f}")
        
        print(f"\n🔍 DIAGNÓSTICO:")
        diagnostico = resultado['diagnostico_overfitting']
        gap_f1 = resultado['gap_f1']
        
        emoji_diag = {
            'ROBUSTO': '✅',
            'LEVE_OVERFITTING': '💛',
            'OVERFITTING_MODERADO': '⚠️', 
            'OVERFITTING_SEVERO': '🚨'
        }.get(diagnostico, '❓')
        
        print(f"   {emoji_diag} Estado: {diagnostico}")
        print(f"   📊 Gap F1: {gap_f1:+.4f}")
        
        # Evaluación del resultado
        f1 = resultado['test_f1']
        if f1 >= 0.8:
            evaluacion = "🟢 EXCELENTE"
        elif f1 >= 0.6:
            evaluacion = "🟡 BUENO"
        elif f1 >= 0.4:
            evaluacion = "🟠 MODERADO"
        elif f1 > 0.1:
            evaluacion = "🔴 BAJO"
        else:
            evaluacion = "⚫ FALLO"
        
        print(f"   🏆 Evaluación: {evaluacion}")
        
        # Top features si están disponibles
        if resultado['feature_importance']:
            print(f"\n🔑 TOP FEATURES:")
            top_features = sorted(resultado['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        print(f"\n⏱️ Tiempo: {resultado['tiempo_entrenamiento']:.1f}s")
    
    def _guardar_resultados_hibridos(self, contaminante, resultado):
        """Guarda resultados del sistema híbrido"""
        try:
            dir_contaminante = os.path.join(self.results_dir, contaminante)
            os.makedirs(dir_contaminante, exist_ok=True)
            
            ruta_json = os.path.join(dir_contaminante, f"{contaminante}_hibrido.json")
            
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"   💾 Resultados guardados: {ruta_json}")
            
        except Exception as e:
            print(f"   ⚠️ Error guardando: {e}")

def probar_sistema_hibrido():
    """Prueba el sistema híbrido con los contaminantes de diagnóstico"""
    
    print("🔄 PRUEBA SISTEMA HÍBRIDO ADAPTATIVO")
    print("="*50)
    print("🎯 Usando datos exitosos del diagnóstico")
    print()
    
    sistema = SistemaHibridoAdaptativo("todo/firmas_espectrales_csv")
    
    contaminantes_test = ['Nh4_Mg_L', 'Caffeine_Ng_L', 'Turbidity_Ntu']
    resultados = {}
    
    for i, contaminante in enumerate(contaminantes_test, 1):
        print(f"\n[{i}/3] 🔬 PROCESANDO: {contaminante}")
        
        resultado = sistema.entrenar_hibrido_adaptativo(contaminante)
        
        if resultado:
            resultados[contaminante] = resultado
            
            # Mostrar resultado inmediato
            f1 = resultado['test_f1']
            modelo = resultado['modelo_usado']
            categoria = resultado['categoria_dataset']
            
            if f1 >= 0.6:
                emoji = "✅"
                estado = "ÉXITO"
            elif f1 >= 0.3:
                emoji = "💛"
                estado = "PARCIAL"
            else:
                emoji = "🚨"
                estado = "FALLO"
            
            print(f"      {emoji} RESULTADO: F1={f1:.3f} | Modelo={modelo} | Cat={categoria} | {estado}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN SISTEMA HÍBRIDO")
    print(f"{'='*60}")
    
    if resultados:
        print(f"✅ Resultados: {len(resultados)}/3")
        print()
        print(f"{'Contaminante':<15} | {'F1':<6} | {'Modelo':<12} | {'Categoría':<8}")
        print("-" * 50)
        
        f1_scores = []
        for cont, res in resultados.items():
            f1 = res['test_f1']
            modelo = res['modelo_usado'][:10]
            categoria = res['categoria_dataset']
            
            f1_scores.append(f1)
            
            if f1 >= 0.6:
                estado = "🟢"
            elif f1 >= 0.3:
                estado = "🟡"
            else:
                estado = "🔴"
            
            print(f"{cont:<15} | {f1:<6.3f} | {modelo:<12} | {categoria:<8} {estado}")
        
        print()
        print(f"📊 F1 promedio: {np.mean(f1_scores):.3f}")
        
        # Verificar mejora vs sistema anterior
        exitos = sum(1 for f1 in f1_scores if f1 >= 0.6)
        print(f"🎯 Casos exitosos: {exitos}/3")
        
        if exitos >= 2:
            print(f"🎉 SISTEMA HÍBRIDO EXITOSO!")
        elif exitos >= 1:
            print(f"💛 SISTEMA HÍBRIDO PARCIALMENTE EXITOSO")
        else:
            print(f"🔴 SISTEMA HÍBRIDO NECESITA AJUSTES")
    
    return resultados

if __name__ == "__main__":
    import numpy as np
    probar_sistema_hibrido()