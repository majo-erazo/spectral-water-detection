# lstm_optimizado_pocos_datos.py
# LSTM específicamente diseñado para datasets muy pequeños
# Soluciona problemas identificados en resultados previos

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l1_l2
    
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    KERAS_AVAILABLE = True
    
except ImportError:
    KERAS_AVAILABLE = False
    print("❌ TensorFlow no disponible")

class LSTMPocosData:
    """
    LSTM optimizado específicamente para datasets muy pequeños
    Soluciona problemas de overfitting y pocos datos
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        
        # Configuración para pocos datos
        self.min_samples_warning = 20  # Advertir si hay menos de 20 muestras
        self.max_epochs_small_data = 50  # Máximo epochs para datos pequeños
        
        # Configuración para reproducibilidad
        np.random.seed(42)
        if KERAS_AVAILABLE:
            tf.random.set_seed(42)
    
    def cargar_firma_espectral_optimizada(self, contaminante):
        """
        Carga y optimiza firma espectral para pocos datos
        """
        print(f"\n📊 Cargando firma espectral: {contaminante}")
        
        # Construir ruta
        ruta_carpeta = os.path.join(self.directorio_base, contaminante)
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
        
        if not archivos_espectrales:
            raise FileNotFoundError(f"No se encontró archivo espectral en {ruta_carpeta}")
        
        # Cargar datos
        ruta_archivo = os.path.join(ruta_carpeta, archivos_espectrales[0])
        datos = pd.read_csv(ruta_archivo)
        
        # Asegurar columnas correctas
        if 'wavelength' not in datos.columns:
            # Intentar mapeo automático
            posibles_wave = [col for col in datos.columns if 'wave' in col.lower() or 'nm' in col.lower()]
            if posibles_wave:
                datos = datos.rename(columns={posibles_wave[0]: 'wavelength'})
        
        if 'high_mean' not in datos.columns:
            posibles_high = [col for col in datos.columns if 'high' in col.lower()]
            if posibles_high:
                datos = datos.rename(columns={posibles_high[0]: 'high_mean'})
        
        if 'low_mean' not in datos.columns:
            posibles_low = [col for col in datos.columns if 'low' in col.lower()]
            if posibles_low:
                datos = datos.rename(columns={posibles_low[0]: 'low_mean'})
        
        # Limpiar y ordenar
        datos = datos.dropna()
        datos = datos.sort_values('wavelength').reset_index(drop=True)
        
        print(f"   ✅ Datos cargados: {datos.shape}")
        print(f"   🌈 Rango: {datos['wavelength'].min():.1f}-{datos['wavelength'].max():.1f} nm")
        
        return datos
    
    def data_augmentation_agresivo(self, datos, factor_aumento=20):
        """
        Data augmentation AGRESIVO específico para firmas espectrales
        Genera muchas más muestras para combatir overfitting
        """
        print(f"🔧 Data augmentation agresivo (factor: {factor_aumento})...")
        
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        # Normalizar wavelengths
        wavelength_norm = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
        
        # Crear muestras originales
        X_original = []
        y_original = []
        
        # Alta concentración
        for i in range(len(wavelengths)):
            X_original.append([wavelength_norm[i], high_response[i]])
            y_original.append(1)
        
        # Baja concentración  
        for i in range(len(wavelengths)):
            X_original.append([wavelength_norm[i], low_response[i]])
            y_original.append(0)
        
        X_original = np.array(X_original)
        y_original = np.array(y_original)
        
        # Data augmentation con múltiples técnicas
        X_augmented = [X_original]
        y_augmented = [y_original]
        
        for aug_iter in range(factor_aumento):
            X_aug = X_original.copy()
            
            # Técnica 1: Ruido gaussiano adaptativo
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, X_aug.shape)
            X_aug += noise
            
            # Técnica 2: Escalado de respuesta espectral
            scale_factor = np.random.uniform(0.9, 1.1)
            X_aug[:, 1] *= scale_factor
            
            # Técnica 3: Desplazamiento espectral
            shift = np.random.uniform(-0.02, 0.02)
            X_aug[:, 0] += shift
            X_aug[:, 0] = np.clip(X_aug[:, 0], 0, 1)
            
            # Técnica 4: Interpolación entre muestras
            if aug_iter % 5 == 0:  # Cada 5 iteraciones
                alpha = np.random.uniform(0.2, 0.8)
                indices = np.random.choice(len(X_original), len(X_original))
                X_aug = alpha * X_aug + (1 - alpha) * X_original[indices]
            
            # Técnica 5: Perturbación de baseline
            if aug_iter % 3 == 0:  # Cada 3 iteraciones
                baseline_shift = np.random.uniform(-0.01, 0.01)
                X_aug[:, 1] += baseline_shift
            
            X_augmented.append(X_aug)
            y_augmented.append(y_original)
        
        # Combinar todo
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        # Crear secuencias para LSTM
        n_samples_por_clase = len(X_final) // (2 * len(wavelengths))
        n_timesteps = len(wavelengths)
        
        X_sequences = X_final.reshape(n_samples_por_clase * 2, n_timesteps, 2)
        y_sequences = y_final[::n_timesteps]
        
        print(f"   📈 Muestras originales: 2")
        print(f"   📈 Muestras aumentadas: {len(X_sequences)}")
        print(f"   📊 Secuencias LSTM: {X_sequences.shape}")
        print(f"   🎯 Distribución: {np.bincount(y_sequences.astype(int))}")
        
        return X_sequences, y_sequences, wavelengths
    
    def crear_modelo_minimalista(self, input_shape, regularization_strength=0.01):
        """
        Crea modelo LSTM MINIMALISTA para pocos datos
        Máxima regularización, mínima complejidad
        """
        print(f"🧠 Creando modelo minimalista para pocos datos...")
        print(f"   📐 Input shape: {input_shape}")
        
        model = Sequential()
        
        # OPCIÓN 1: LSTM Ultra-simple
        model.add(LSTM(
            16,  # Solo 16 unidades 
            return_sequences=False,
            input_shape=input_shape,
            dropout=0.5,  # Dropout alto
            recurrent_dropout=0.5,
            kernel_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength),
            recurrent_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength)
        ))
        
        # Dropout agresivo
        model.add(Dropout(0.6))
        
        # Capa densa mínima
        model.add(Dense(
            8, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=regularization_strength, l2=regularization_strength)
        ))
        model.add(Dropout(0.5))
        
        # Salida
        model.add(Dense(1, activation='sigmoid'))
        
        # Compilar con learning rate bajo
        model.compile(
            optimizer=Adam(learning_rate=0.0001),  # LR muy bajo
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ Parámetros totales: {model.count_params():,}")
        
        return model
    
    def validacion_cruzada_robusta(self, X, y, n_splits=5):
        """
        Validación cruzada robusta para datasets pequeños
        """
        print(f"🔄 Validación cruzada robusta ({n_splits} folds)...")
        
        # Si hay muy pocos datos, usar Leave-One-Out
        if len(X) < 10:
            print("   ⚠️ Datos insuficientes, usando Leave-One-Out")
            cv = LeaveOneOut()
            n_splits = len(X)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"   📂 Fold {fold + 1}/{min(n_splits, len(X))}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Normalizar
            scaler = RobustScaler()  # Más robusto que StandardScaler
            X_train_scaled = X_train_fold.copy()
            X_val_scaled = X_val_fold.copy()
            
            # Escalar cada feature por separado
            for i in range(X_train_fold.shape[2]):
                X_flat_train = X_train_fold[:, :, i].reshape(-1, 1)
                X_flat_val = X_val_fold[:, :, i].reshape(-1, 1)
                
                X_scaled_train = scaler.fit_transform(X_flat_train)
                X_scaled_val = scaler.transform(X_flat_val)
                
                X_train_scaled[:, :, i] = X_scaled_train.reshape(X_train_fold.shape[0], X_train_fold.shape[1])
                X_val_scaled[:, :, i] = X_scaled_val.reshape(X_val_fold.shape[0], X_val_fold.shape[1])
            
            # Crear modelo
            modelo = self.crear_modelo_minimalista(X_train_scaled.shape[1:])
            
            # Entrenar con POCAS epochs
            history = modelo.fit(
                X_train_scaled, y_train_fold,
                validation_data=(X_val_scaled, y_val_fold),
                epochs=30,  # Pocas epochs
                batch_size=min(8, len(X_train_scaled) // 2),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                verbose=0
            )
            
            # Evaluar
            y_pred_proba = modelo.predict(X_val_scaled, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Métricas
            acc = accuracy_score(y_val_fold, y_pred)
            f1 = f1_score(y_val_fold, y_pred, average='binary', zero_division=0)
            
            try:
                auc = roc_auc_score(y_val_fold, y_pred_proba)
            except:
                auc = 0.5
            
            scores.append({
                'accuracy': acc,
                'f1_score': f1,
                'auc': auc,
                'epochs_trained': len(history.history['loss'])
            })
            
            print(f"      📊 Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Promediar resultados
        avg_scores = {
            'accuracy': np.mean([s['accuracy'] for s in scores]),
            'f1_score': np.mean([s['f1_score'] for s in scores]),
            'auc': np.mean([s['auc'] for s in scores]),
            'accuracy_std': np.std([s['accuracy'] for s in scores]),
            'f1_score_std': np.std([s['f1_score'] for s in scores]),
            'auc_std': np.std([s['auc'] for s in scores]),
            'fold_scores': scores
        }
        
        print(f"\n📈 RESULTADOS PROMEDIO:")
        print(f"   🎯 Accuracy: {avg_scores['accuracy']:.4f} ± {avg_scores['accuracy_std']:.4f}")
        print(f"   🎯 F1-score: {avg_scores['f1_score']:.4f} ± {avg_scores['f1_score_std']:.4f}")
        print(f"   🎯 AUC: {avg_scores['auc']:.4f} ± {avg_scores['auc_std']:.4f}")
        
        return avg_scores
    
    def entrenar_contaminante_optimizado(self, contaminante):
        """
        Entrena modelo optimizado para un contaminante con pocos datos
        """
        print(f"\n{'='*70}")
        print(f"🧬 ENTRENANDO LSTM OPTIMIZADO: {contaminante}")
        print(f"{'='*70}")
        
        if not KERAS_AVAILABLE:
            print("❌ Keras no disponible")
            return None
        
        try:
            inicio_tiempo = datetime.datetime.now()
            
            # 1. Cargar datos
            datos_espectrales = self.cargar_firma_espectral_optimizada(contaminante)
            
            # 2. Data augmentation agresivo
            X_sequences, y_sequences, wavelengths = self.data_augmentation_agresivo(
                datos_espectrales, factor_aumento=50  # MUY agresivo
            )
            
            # 3. Advertir si hay pocos datos
            if len(X_sequences) < self.min_samples_warning:
                print(f"⚠️ ADVERTENCIA: Solo {len(X_sequences)} muestras después de augmentation")
                print("   Considera usar modelos más simples (SVM, Random Forest)")
            
            # 4. Validación cruzada robusta
            resultados_cv = self.validacion_cruzada_robusta(X_sequences, y_sequences)
            
            fin_tiempo = datetime.datetime.now()
            tiempo_entrenamiento = (fin_tiempo - inicio_tiempo).total_seconds()
            
            # 5. Preparar resultados finales
            resultados = {
                'contaminante': contaminante,
                'metodo': 'lstm_optimizado_pocos_datos',
                'accuracy': resultados_cv['accuracy'],
                'f1_score': resultados_cv['f1_score'],
                'auc': resultados_cv['auc'],
                'accuracy_std': resultados_cv['accuracy_std'],
                'f1_score_std': resultados_cv['f1_score_std'],
                'auc_std': resultados_cv['auc_std'],
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'n_muestras_aumentadas': len(X_sequences),
                'n_wavelengths': len(wavelengths),
                'metodo_validacion': 'validacion_cruzada_robusta',
                'fold_scores': resultados_cv['fold_scores']
            }
            
            # 6. Clasificar rendimiento
            f1_score = resultados['f1_score']
            if f1_score >= 0.80:
                clasificacion = "🥇 EXCELENTE"
            elif f1_score >= 0.70:
                clasificacion = "🥈 BUENO"
            elif f1_score >= 0.60:
                clasificacion = "🥉 ACEPTABLE"
            else:
                clasificacion = "❌ PROBLEMÁTICO"
            
            print(f"\n{'='*50}")
            print(f"📊 RESULTADO FINAL - {contaminante}")
            print(f"{'='*50}")
            print(f"🎯 F1-score: {f1_score:.4f} ± {resultados['f1_score_std']:.4f}")
            print(f"🎯 Accuracy: {resultados['accuracy']:.4f} ± {resultados['accuracy_std']:.4f}")
            print(f"🎯 AUC: {resultados['auc']:.4f} ± {resultados['auc_std']:.4f}")
            print(f"⏱️ Tiempo: {tiempo_entrenamiento:.1f}s")
            print(f"📈 Clasificación: {clasificacion}")
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def entrenar_todos_optimizado(self, contaminantes_test=None):
        """
        Entrena modelos optimizados para varios contaminantes
        """
        if contaminantes_test is None:
            # Detectar contaminantes automáticamente
            contaminantes_test = []
            for carpeta in os.listdir(self.directorio_base):
                ruta_carpeta = os.path.join(self.directorio_base, carpeta)
                if os.path.isdir(ruta_carpeta):
                    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
                    if archivos:
                        contaminantes_test.append(carpeta)
        
        print(f"\n{'='*80}")
        print(f"🔬 ENTRENAMIENTO LSTM OPTIMIZADO - DATASET PEQUEÑO")
        print(f"{'='*80}")
        print(f"📋 Contaminantes: {len(contaminantes_test)}")
        print(f"🛠️ Optimizaciones: Data augmentation agresivo, modelo minimalista, validación robusta")
        
        resultados_todos = {}
        inicio_total = datetime.datetime.now()
        
        for i, contaminante in enumerate(contaminantes_test):
            print(f"\n[{i+1}/{len(contaminantes_test)}] 🔄 Procesando: {contaminante}")
            
            resultado = self.entrenar_contaminante_optimizado(contaminante)
            resultados_todos[contaminante] = resultado
        
        fin_total = datetime.datetime.now()
        tiempo_total = (fin_total - inicio_total).total_seconds()
        
        # Generar reporte comparativo
        self._generar_reporte_optimizado(resultados_todos, tiempo_total)
        
        return resultados_todos
    
    def _generar_reporte_optimizado(self, resultados, tiempo_total):
        """Genera reporte optimizado comparando con resultados previos"""
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE OPTIMIZADO - COMPARACIÓN CON RESULTADOS PREVIOS")
        print(f"{'='*80}")
        
        # Filtrar resultados exitosos
        exitosos = {k: v for k, v in resultados.items() if v is not None}
        
        if not exitosos:
            print("❌ No hay resultados exitosos")
            return
        
        # Estadísticas del modelo optimizado
        f1_scores = [r['f1_score'] for r in exitosos.values()]
        accuracies = [r['accuracy'] for r in exitosos.values()]
        aucs = [r['auc'] for r in exitosos.values()]
        
        # Estadísticas previas (del contexto anterior)
        f1_promedio_anterior = 0.348  # Del resultado anterior
        accuracy_promedio_anterior = 0.523
        auc_promedio_anterior = 0.591
        
        # Mejoras
        mejora_f1 = np.mean(f1_scores) - f1_promedio_anterior
        mejora_accuracy = np.mean(accuracies) - accuracy_promedio_anterior
        mejora_auc = np.mean(aucs) - auc_promedio_anterior
        
        print(f"✅ Modelos exitosos: {len(exitosos)}")
        print(f"⏱️ Tiempo total: {tiempo_total:.1f}s")
        print(f"\n📈 COMPARACIÓN DE RENDIMIENTO:")
        print(f"   F1-score:")
        print(f"     • Anterior: {f1_promedio_anterior:.4f}")
        print(f"     • Optimizado: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"     • Mejora: {mejora_f1:+.4f} ({'✅' if mejora_f1 > 0 else '❌'})")
        
        print(f"   Accuracy:")
        print(f"     • Anterior: {accuracy_promedio_anterior:.4f}")
        print(f"     • Optimizado: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"     • Mejora: {mejora_accuracy:+.4f} ({'✅' if mejora_accuracy > 0 else '❌'})")
        
        print(f"   AUC:")
        print(f"     • Anterior: {auc_promedio_anterior:.4f}")
        print(f"     • Optimizado: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"     • Mejora: {mejora_auc:+.4f} ({'✅' if mejora_auc > 0 else '❌'})")
        
        # Top 5 mejores
        print(f"\n🏆 TOP 5 CONTAMINANTES:")
        ranking = sorted(exitosos.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for i, (contaminante, resultado) in enumerate(ranking[:5], 1):
            f1 = resultado['f1_score']
            f1_std = resultado['f1_score_std']
            
            if f1 >= 0.8:
                emoji = "🥇"
            elif f1 >= 0.7:
                emoji = "🥈"
            elif f1 >= 0.6:
                emoji = "🥉"
            else:
                emoji = "📊"
            
            print(f"   {i}. {contaminante:15} | F1: {f1:.4f}±{f1_std:.3f} {emoji}")
        
        # Guardar resultados
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"lstm_optimizado_{timestamp}.json"
        
        resumen = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metodo': 'lstm_optimizado_pocos_datos',
            'mejoras_implementadas': [
                'Data augmentation agresivo (factor 50x)',
                'Modelo minimalista (16 LSTM units)',
                'Regularización fuerte (L1+L2 + Dropout 0.5-0.6)',
                'Validación cruzada robusta',
                'Learning rate bajo (0.0001)'
            ],
            'tiempo_total_segundos': tiempo_total,
            'comparacion_con_anterior': {
                'f1_score_anterior': f1_promedio_anterior,
                'f1_score_optimizado': float(np.mean(f1_scores)),
                'mejora_f1': float(mejora_f1),
                'accuracy_anterior': accuracy_promedio_anterior,
                'accuracy_optimizado': float(np.mean(accuracies)),
                'mejora_accuracy': float(mejora_accuracy)
            },
            'estadisticas': {
                'exitosos': len(exitosos),
                'f1_score_promedio': float(np.mean(f1_scores)),
                'f1_score_std': float(np.std(f1_scores)),
                'accuracy_promedio': float(np.mean(accuracies)),
                'auc_promedio': float(np.mean(aucs))
            },
            'resultados_detallados': exitosos
        }
        
        try:
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados guardados: {nombre_archivo}")
        except Exception as e:
            print(f"⚠️ Error guardando: {e}")
        
        return resumen

def main_optimizado():
    """Función principal del LSTM optimizado"""
    
    if not KERAS_AVAILABLE:
        print("❌ TensorFlow no disponible")
        return
    
    print("🔧 LSTM OPTIMIZADO PARA DATASETS PEQUEÑOS")
    print("="*60)
    print("🎯 Soluciona problemas identificados:")
    print("   • Overfitting severo")
    print("   • Dataset extremadamente pequeño") 
    print("   • Arquitectura demasiado compleja")
    print("   • Validación inadecuada")
    
    # Crear entrenador optimizado
    entrenador = LSTMPocosData("todo/firmas_espectrales_csv")
    
    # Probar con algunos contaminantes primero
    contaminantes_prueba = ["Caffeine", "Acesulfame", "Doc", "Nh4", "Turbidity"]
    
    print(f"\n🧪 Probando con {len(contaminantes_prueba)} contaminantes...")
    
    resultados = entrenador.entrenar_todos_optimizado(contaminantes_prueba)
    
    return resultados

if __name__ == "__main__":
    resultados = main_optimizado()