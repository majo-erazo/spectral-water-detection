# lstm_firmas_espectrales.py
# LSTM entrenado directamente con firmas espectrales
# Proyecto: María José Erazo González - UDP

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Importar Keras/TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
    from tensorflow.keras.layers import Input, Concatenate, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    
    # Configurar TensorFlow
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    KERAS_AVAILABLE = True
    print("✅ TensorFlow/Keras disponible")
    
except ImportError:
    print("❌ TensorFlow no disponible. Instalar con: pip install tensorflow")
    KERAS_AVAILABLE = False

class LSTMFirmasEspectrales:
    """
    LSTM entrenado directamente con firmas espectrales
    Optimizado para datos espectrales wavelength -> respuesta
    """
    
    def __init__(self, directorio_base="todo/firmas_espectrales_csv"):
        self.directorio_base = directorio_base
        self.contaminantes_disponibles = self._detectar_contaminantes()
        
        # Mapeo de nombres de carpetas a contaminantes
        self.mapeo_carpetas = {
            'Doc': 'Doc_Mg_L',
            'Turbidity': 'Turbidity_Ntu', 
            'Nh4': 'Nh4_Mg_L',
            'Acesulfame': 'Acesulfame_Ng_L',
            'Caffeine': 'Caffeine_Ng_L',
            'Benzotriazole': 'Benzotriazole_Ng_L',
            'Triclosan': 'Triclosan_Ng_L',
            'Diuron': 'Diuron_Ng_L',
            'Mecoprop': 'Mecoprop_Ng_L'
        }
        
        # Configuración para reproducibilidad
        np.random.seed(42)
        if KERAS_AVAILABLE:
            tf.random.set_seed(42)
    
    def _detectar_contaminantes(self):
        """Detecta automáticamente los contaminantes disponibles"""
        contaminantes = []
        
        if os.path.exists(self.directorio_base):
            for carpeta in os.listdir(self.directorio_base):
                ruta_carpeta = os.path.join(self.directorio_base, carpeta)
                if os.path.isdir(ruta_carpeta):
                    # Buscar archivos *_datos_espectrales.csv
                    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
                    if archivos:
                        contaminantes.append(carpeta)
        
        print(f"🔬 Contaminantes detectados: {len(contaminantes)}")
        for cont in sorted(contaminantes):
            print(f"   • {cont}")
        
        return sorted(contaminantes)
    
    def cargar_firma_espectral(self, contaminante):
        """
        Carga la firma espectral de un contaminante específico
        """
        print(f"\n📊 Cargando firma espectral: {contaminante}")
        
        # Construir ruta al archivo
        ruta_carpeta = os.path.join(self.directorio_base, contaminante)
        
        # Buscar archivo de datos espectrales
        archivos_espectrales = [f for f in os.listdir(ruta_carpeta) if f.endswith('_datos_espectrales.csv')]
        
        if not archivos_espectrales:
            raise FileNotFoundError(f"No se encontró archivo de datos espectrales en {ruta_carpeta}")
        
        archivo_espectral = archivos_espectrales[0]
        ruta_archivo = os.path.join(ruta_carpeta, archivo_espectral)
        
        # Cargar datos
        datos = pd.read_csv(ruta_archivo)
        
        print(f"   📁 Archivo: {archivo_espectral}")
        print(f"   📏 Dimensiones: {datos.shape}")
        print(f"   📋 Columnas: {list(datos.columns)}")
        
        # Verificar columnas esperadas
        columnas_requeridas = ['wavelength', 'high_mean', 'low_mean']
        columnas_faltantes = [col for col in columnas_requeridas if col not in datos.columns]
        
        if columnas_faltantes:
            print(f"   ⚠️ Columnas faltantes: {columnas_faltantes}")
            print(f"   📋 Columnas disponibles: {list(datos.columns)}")
            
            # Intentar mapear columnas automáticamente
            datos = self._mapear_columnas_automaticamente(datos)
        
        # Limpiar y ordenar datos
        datos = datos.dropna()
        datos = datos.sort_values('wavelength').reset_index(drop=True)
        
        print(f"   ✅ Datos limpiados: {datos.shape}")
        print(f"   🌈 Rango wavelength: {datos['wavelength'].min():.1f} - {datos['wavelength'].max():.1f} nm")
        
        return datos
    
    def _mapear_columnas_automaticamente(self, datos):
        """Mapea automáticamente columnas con nombres diferentes"""
        mapeo_alternativo = {
            'wavelength': ['wavelength', 'wave', 'nm', 'lambda'],
            'high_mean': ['high_mean', 'high', 'alta', 'concentration_high'],
            'low_mean': ['low_mean', 'low', 'baja', 'concentration_low']
        }
        
        nuevos_nombres = {}
        
        for col_objetivo, posibles_nombres in mapeo_alternativo.items():
            for col_datos in datos.columns:
                if col_datos.lower() in [nombre.lower() for nombre in posibles_nombres]:
                    nuevos_nombres[col_datos] = col_objetivo
                    break
        
        if nuevos_nombres:
            print(f"   🔄 Mapeando columnas: {nuevos_nombres}")
            datos = datos.rename(columns=nuevos_nombres)
        
        return datos
    
    def preparar_datos_lstm(self, datos, augmentar_datos=True):
        """
        Prepara los datos espectrales para entrenamiento LSTM
        
        Estrategia:
        1. Crear secuencias wavelength -> respuesta
        2. Dos clases: high_concentration vs low_concentration
        3. Data augmentation para balancear clases
        """
        print(f"🔧 Preparando datos para LSTM...")
        
        # Extraer wavelengths y respuestas
        wavelengths = datos['wavelength'].values
        high_response = datos['high_mean'].values
        low_response = datos['low_mean'].values
        
        # Normalizar wavelengths para usar como features adicionales
        wavelength_normalized = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
        
        # Crear secuencias para LSTM
        # Cada muestra tendrá: [wavelength_norm, respuesta_espectral]
        
        # Secuencias de alta concentración (clase 1)
        X_high = np.column_stack([wavelength_normalized, high_response])
        y_high = np.ones(len(X_high))  # Clase 1
        
        # Secuencias de baja concentración (clase 0) 
        X_low = np.column_stack([wavelength_normalized, low_response])
        y_low = np.zeros(len(X_low))  # Clase 0
        
        # Combinar datos
        X_combined = np.vstack([X_high, X_low])
        y_combined = np.hstack([y_high, y_low])
        
        print(f"   📊 Muestras creadas:")
        print(f"      • Alta concentración: {len(X_high)}")
        print(f"      • Baja concentración: {len(X_low)}")
        print(f"      • Total: {len(X_combined)}")
        
        # Aplicar data augmentation si se solicita
        if augmentar_datos:
            X_combined, y_combined = self._aplicar_augmentation_espectral(X_combined, y_combined, wavelengths)
        
        # Preparar para LSTM: reshape a (samples, timesteps, features)
        # Cada wavelength es un timestep, cada timestep tiene 2 features: [wavelength_norm, respuesta]
        n_samples = len(X_combined) // len(wavelengths)
        n_timesteps = len(wavelengths)
        n_features = 2
        
        X_sequences = X_combined.reshape(n_samples, n_timesteps, n_features)
        y_sequences = y_combined[::n_timesteps]  # Una etiqueta por secuencia
        
        print(f"   🔄 Secuencias LSTM: {X_sequences.shape}")
        print(f"   🎯 Etiquetas: {y_sequences.shape} - Distribución: {np.bincount(y_sequences.astype(int))}")
        
        return X_sequences, y_sequences, wavelengths
    
    def _aplicar_augmentation_espectral(self, X, y, wavelengths, factor_aumento=3):
        """
        Aplica data augmentation específico para datos espectrales
        """
        print(f"   🔧 Aplicando data augmentation (factor: {factor_aumento})...")
        
        n_original = len(X) // len(wavelengths)
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(factor_aumento):
            X_aug = X.copy()
            
            # Augmentation 1: Ruido gaussiano en la respuesta espectral
            ruido_respuesta = np.random.normal(0, 0.02, X_aug.shape[0])
            X_aug[:, 1] += ruido_respuesta  # Añadir ruido a la respuesta
            
            # Augmentation 2: Escalado ligero de la respuesta
            factor_escala = np.random.uniform(0.95, 1.05)
            X_aug[:, 1] *= factor_escala
            
            # Augmentation 3: Desplazamiento espectral ligero
            desplazamiento = np.random.uniform(-0.01, 0.01)
            X_aug[:, 0] += desplazamiento
            X_aug[:, 0] = np.clip(X_aug[:, 0], 0, 1)  # Mantener en rango [0,1]
            
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        print(f"      • Muestras originales: {n_original * 2}")
        print(f"      • Muestras aumentadas: {len(X_final) // len(wavelengths)}")
        
        return X_final, y_final
    
    def crear_modelo_lstm_espectral(self, input_shape, arquitectura='hibrido'):
        """
        Crea modelo LSTM optimizado para firmas espectrales
        
        Args:
            input_shape: (timesteps, features) 
            arquitectura: 'simple', 'bidireccional', 'hibrido'
        """
        print(f"🧠 Creando modelo LSTM ({arquitectura}) - Input: {input_shape}")
        
        model = Sequential()
        
        if arquitectura == 'simple':
            # LSTM simple para casos con pocos datos
            model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            
        elif arquitectura == 'bidireccional':
            # LSTM bidireccional para capturar patrones en ambas direcciones
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(32, return_sequences=False)))
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
            
        elif arquitectura == 'hibrido':
            # Enfoque híbrido: LSTM + CNN1D para capturar múltiples patrones
            
            # Input principal
            input_layer = Input(shape=input_shape)
            
            # Rama LSTM: Para patrones secuenciales
            lstm_branch = LSTM(64, return_sequences=True)(input_layer)
            lstm_branch = Dropout(0.3)(lstm_branch)
            lstm_branch = LSTM(32, return_sequences=False)(lstm_branch)
            lstm_branch = Dense(32, activation='relu')(lstm_branch)
            
            # Rama CNN: Para patrones locales en el espectro
            cnn_branch = Conv1D(filters=64, kernel_size=5, activation='relu')(input_layer)
            cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
            cnn_branch = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_branch)
            cnn_branch = GlobalMaxPooling1D()(cnn_branch)
            cnn_branch = Dense(32, activation='relu')(cnn_branch)
            
            # Combinar ramas
            combined = Concatenate()([lstm_branch, cnn_branch])
            combined = BatchNormalization()(combined)
            combined = Dropout(0.4)(combined)
            
            # Capas finales
            dense1 = Dense(64, activation='relu')(combined)
            dense1 = Dropout(0.3)(dense1)
            dense2 = Dense(32, activation='relu')(dense1)
            
            # Salida
            output = Dense(1, activation='sigmoid')(dense2)
            
            # Crear modelo
            model = Model(inputs=input_layer, outputs=output)
        
        # Para arquitecturas no híbridas, añadir capa de salida
        if arquitectura != 'hibrido':
            model.add(Dense(1, activation='sigmoid'))
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ Modelo creado - Parámetros: {model.count_params():,}")
        
        return model
    
    def entrenar_contaminante(self, contaminante, arquitectura='hibrido', epochs=150, validacion_cruzada=False):
        """
        Entrena modelo LSTM para un contaminante específico
        """
        print(f"\n{'='*70}")
        print(f"🧬 ENTRENANDO LSTM ESPECTRAL: {contaminante}")
        print(f"{'='*70}")
        
        if not KERAS_AVAILABLE:
            print("❌ Keras no disponible")
            return None
        
        try:
            inicio_tiempo = datetime.datetime.now()
            
            # 1. Cargar firma espectral
            datos_espectrales = self.cargar_firma_espectral(contaminante)
            
            # 2. Preparar datos para LSTM
            X_sequences, y_sequences, wavelengths = self.preparar_datos_lstm(datos_espectrales)
            
            # 3. Normalizar secuencias
            print(f"🔄 Normalizando secuencias...")
            scaler = StandardScaler()
            
            # Normalizar cada feature por separado
            X_normalized = X_sequences.copy()
            for i in range(X_sequences.shape[2]):  # Para cada feature
                X_flat = X_sequences[:, :, i].reshape(-1, 1)
                X_scaled = scaler.fit_transform(X_flat)
                X_normalized[:, :, i] = X_scaled.reshape(X_sequences.shape[0], X_sequences.shape[1])
            
            # 4. División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalized, y_sequences, 
                test_size=0.2, random_state=42, stratify=y_sequences
            )
            
            print(f"📊 División de datos:")
            print(f"   • Train: {X_train.shape[0]} muestras")
            print(f"   • Test: {X_test.shape[0]} muestras")
            print(f"   • Distribución train: {np.bincount(y_train.astype(int))}")
            print(f"   • Distribución test: {np.bincount(y_test.astype(int))}")
            
            # 5. Crear modelo
            modelo = self.crear_modelo_lstm_espectral(X_train.shape[1:], arquitectura)
            
            # 6. Configurar callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=20, 
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=10, 
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # 7. Entrenar modelo
            print(f"🚀 Iniciando entrenamiento ({epochs} epochs)...")
            
            history = modelo.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=min(32, X_train.shape[0] // 4),
                callbacks=callbacks,
                verbose=1
            )
            
            # 8. Evaluar modelo
            print(f"📈 Evaluando modelo...")
            
            # Predicciones
            y_pred_proba = modelo.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.0
            
            fin_tiempo = datetime.datetime.now()
            tiempo_entrenamiento = (fin_tiempo - inicio_tiempo).total_seconds()
            
            # 9. Resultados
            print(f"\n{'='*50}")
            print(f"📊 RESULTADOS - {contaminante}")
            print(f"{'='*50}")
            print(f"🎯 Accuracy: {accuracy:.4f}")
            print(f"🎯 F1-score: {f1:.4f}")
            print(f"🎯 AUC: {auc:.4f}")
            print(f"⏱️ Tiempo: {tiempo_entrenamiento:.1f}s")
            print(f"📐 Epochs entrenados: {len(history.history['loss'])}")
            
            # Informe detallado
            print(f"\n📋 Informe de clasificación:")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # 10. Preparar resultados
            resultados = {
                'contaminante': contaminante,
                'arquitectura': arquitectura,
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'auc': float(auc),
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'epochs_entrenados': len(history.history['loss']),
                'n_muestras_train': int(X_train.shape[0]),
                'n_muestras_test': int(X_test.shape[0]),
                'input_shape': list(X_train.shape[1:]),
                'n_wavelengths': len(wavelengths),
                'rango_wavelength': [float(wavelengths.min()), float(wavelengths.max())],
                'distribucion_clases': {
                    'train': np.bincount(y_train.astype(int)).tolist(),
                    'test': np.bincount(y_test.astype(int)).tolist()
                },
                'historia_entrenamiento': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
                }
            }
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error entrenando {contaminante}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def entrenar_todos_contaminantes(self, arquitectura='hibrido', epochs=150):
        """
        Entrena modelos LSTM para todos los contaminantes disponibles
        """
        print(f"\n{'='*80}")
        print(f"🔬 ENTRENAMIENTO MASIVO LSTM ESPECTRAL")
        print(f"{'='*80}")
        print(f"📋 Contaminantes a procesar: {len(self.contaminantes_disponibles)}")
        print(f"🏗️ Arquitectura: {arquitectura}")
        print(f"📅 Epochs: {epochs}")
        
        resultados_todos = {}
        inicio_total = datetime.datetime.now()
        
        for i, contaminante in enumerate(self.contaminantes_disponibles):
            print(f"\n[{i+1}/{len(self.contaminantes_disponibles)}] 🔄 Procesando: {contaminante}")
            
            resultado = self.entrenar_contaminante(contaminante, arquitectura, epochs)
            resultados_todos[contaminante] = resultado
        
        fin_total = datetime.datetime.now()
        tiempo_total = (fin_total - inicio_total).total_seconds()
        
        # Generar reporte consolidado
        self._generar_reporte_consolidado(resultados_todos, tiempo_total, arquitectura)
        
        return resultados_todos
    
    def _generar_reporte_consolidado(self, resultados, tiempo_total, arquitectura):
        """Genera reporte consolidado de todos los entrenamientos"""
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE CONSOLIDADO - LSTM ESPECTRAL")
        print(f"{'='*80}")
        
        # Filtrar resultados exitosos
        exitosos = {k: v for k, v in resultados.items() if v is not None}
        fallidos = [k for k, v in resultados.items() if v is None]
        
        if not exitosos:
            print("❌ No se obtuvieron resultados exitosos")
            return
        
        # Estadísticas generales
        accuracies = [r['accuracy'] for r in exitosos.values()]
        f1_scores = [r['f1_score'] for r in exitosos.values()]
        aucs = [r['auc'] for r in exitosos.values()]
        
        print(f"✅ Exitosos: {len(exitosos)}/{len(resultados)}")
        if fallidos:
            print(f"❌ Fallidos: {fallidos}")
        
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   🎯 Accuracy promedio: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   🎯 F1-score promedio: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"   🎯 AUC promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"   ⏱️ Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
        
        # Ranking de contaminantes
        print(f"\n🏆 RANKING POR F1-SCORE:")
        ranking = sorted(exitosos.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for i, (contaminante, resultado) in enumerate(ranking[:10], 1):
            f1 = resultado['f1_score']
            acc = resultado['accuracy']
            auc = resultado['auc']
            
            # Determinar calidad
            if f1 >= 0.85:
                calidad = "🥇 EXCELENTE"
            elif f1 >= 0.75:
                calidad = "🥈 BUENO"
            elif f1 >= 0.65:
                calidad = "🥉 REGULAR"
            else:
                calidad = "⚠️ BAJO"
            
            print(f"   {i:2d}. {contaminante:15} | F1: {f1:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | {calidad}")
        
        # Contaminantes problemáticos
        problematicos = [(k, v) for k, v in exitosos.items() if v['f1_score'] < 0.70]
        if problematicos:
            print(f"\n⚠️ CONTAMINANTES PROBLEMÁTICOS (F1 < 0.70):")
            for contaminante, resultado in problematicos:
                print(f"   • {contaminante}: F1 = {resultado['f1_score']:.4f}")
        
        # Guardar resultados
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"lstm_espectral_{arquitectura}_{timestamp}.json"
        
        resumen = {
            'timestamp': datetime.datetime.now().isoformat(),
            'arquitectura': arquitectura,
            'tiempo_total_segundos': tiempo_total,
            'estadisticas': {
                'exitosos': len(exitosos),
                'fallidos': len(fallidos),
                'accuracy_promedio': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'f1_score_promedio': float(np.mean(f1_scores)),
                'f1_score_std': float(np.std(f1_scores)),
                'auc_promedio': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs))
            },
            'ranking': [(k, v['f1_score']) for k, v in ranking],
            'resultados_detallados': exitosos
        }
        
        # Guardar JSON
        try:
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados guardados: {nombre_archivo}")
        except Exception as e:
            print(f"⚠️ Error guardando resultados: {e}")
        
        return resumen

def main():
    """Función principal"""
    
    if not KERAS_AVAILABLE:
        print("❌ TensorFlow/Keras no disponible. Instalar con:")
        print("   pip install tensorflow")
        return
    
    print("🧬 LSTM PARA FIRMAS ESPECTRALES")
    print("="*60)
    
    # Crear entrenador
    entrenador = LSTMFirmasEspectrales("todo/firmas_espectrales_csv")
    
    # Opciones de uso
    print("\n🎮 OPCIONES DE USO:")
    print("1. 🧪 Entrenar un contaminante específico")
    print("2. 🔬 Entrenar todos los contaminantes")
    print("3. 📊 Ver contaminantes disponibles")
    
    # Para desarrollo, entrenar todos con arquitectura híbrida
    print("\n🚀 Iniciando entrenamiento automático (arquitectura híbrida)...")
    
    resultados = entrenador.entrenar_todos_contaminantes(
        arquitectura='hibrido',
        epochs=150
    )
    
    print("\n✅ Entrenamiento completado!")
    return resultados

if __name__ == "__main__":
    resultados = main()