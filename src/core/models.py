"""
Módulo de modelos de Machine Learning para detección de contaminantes.

Este módulo contiene las clases base y específicas para todos los modelos
de machine learning utilizados en el sistema.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

# Importar XGBoost si está disponible
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Importar TensorFlow si está disponible  
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BaseContaminantModel(ABC):
    """Clase base abstracta para todos los modelos de detección de contaminantes."""
    
    def predict(self, X):
        """Realiza predicciones."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Obtiene la importancia de las características."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancia")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar en formato nativo de XGBoost
        if filepath.endswith('.json'):
            self.model.save_model(filepath)
        else:
            # Guardar con joblib incluyendo metadatos
            joblib.dump({
                'model': self.model,
                'contaminant_name': self.contaminant_name,
                'config': self.config,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'best_params': self.best_params
            }, filepath)
        
        print(f"Modelo XGBoost guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        if filepath.endswith('.json'):
            # Cargar formato nativo de XGBoost
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
        else:
            # Cargar con joblib
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.contaminant_name = model_data.get('contaminant_name', self.contaminant_name)
            self.config = model_data.get('config', {})
            self.feature_names = model_data.get('feature_names')
            self.training_history = model_data.get('training_history', {})
            self.best_params = model_data.get('best_params')
        
        self.is_trained = True
        print(f"Modelo XGBoost cargado desde: {filepath}")


class LSTMContaminantModel(BaseContaminantModel):
    """Modelo LSTM para análisis temporal de contaminantes."""
    
    def __init__(self, contaminant_name, config=None):
        super().__init__(contaminant_name, config)
        self.sequence_length = self.config.get('sequence_length', 10)
        self.history = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
    
    def _prepare_sequences(self, X, y=None):
        """Prepara secuencias temporales para LSTM."""
        n_samples, n_features = X.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"No hay suficientes muestras para crear secuencias de longitud {self.sequence_length}")
        
        # Crear secuencias
        X_sequences = []
        y_sequences = []
        
        for i in range(n_samples - self.sequence_length + 1):
            X_sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                y_sequences.append(y[i + self.sequence_length - 1])  # Último valor de la secuencia
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        
        return X_sequences
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena el modelo LSTM.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            
        Returns:
            dict: Resultados del entrenamiento
        """
        print(f"Entrenando modelo LSTM para {self.contaminant_name}...")
        
        # Convertir a numpy si es necesario
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # Preparar secuencias
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        # Configurar arquitectura del modelo
        n_features = X_train_seq.shape[2]
        n_classes = len(np.unique(y_train_seq))
        
        self.model = Sequential([
            LSTM(
                units=self.config.get('lstm_units', 32),
                return_sequences=False,
                input_shape=(self.sequence_length, n_features),
                dropout=self.config.get('dropout_rate', 0.3),
                recurrent_dropout=self.config.get('recurrent_dropout', 0.2)
            ),
            Dropout(self.config.get('dropout_rate', 0.3)),
            Dense(
                units=n_classes if n_classes > 2 else 1,
                activation='softmax' if n_classes > 2 else 'sigmoid'
            )
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get('learning_rate', 0.001)
            ),
            loss='sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Configurar callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Entrenar modelo
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 16),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Guardar información del entrenamiento
        self.training_history = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train_seq),
            'n_features': n_features,
            'sequence_length': self.sequence_length,
            'class_distribution': pd.Series(y_train_seq).value_counts().to_dict(),
            'final_loss': float(self.history.history['loss'][-1]),
            'final_accuracy': float(self.history.history['accuracy'][-1])
        }
        
        if validation_data:
            self.training_history.update({
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1])
            })
        
        print(f"Entrenamiento LSTM completado para {self.contaminant_name}")
        return self.training_history
    
    def predict(self, X):
        """Realiza predicciones."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_seq = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq)
        
        # Convertir probabilidades a clases
        if predictions.shape[1] == 1:
            # Binario
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multiclase
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_seq = self._prepare_sequences(X)
        return self.model.predict(X_seq)
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo de TensorFlow
        self.model.save(filepath)
        
        # Guardar metadatos por separado
        metadata_file = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'contaminant_name': self.contaminant_name,
                'config': self.config,
                'sequence_length': self.sequence_length,
                'training_history': self.training_history
            }, f, indent=2)
        
        print(f"Modelo LSTM guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Cargar modelo de TensorFlow
        self.model = tf.keras.models.load_model(filepath)
        
        # Cargar metadatos
        metadata_file = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.contaminant_name = metadata.get('contaminant_name', self.contaminant_name)
                self.config = metadata.get('config', {})
                self.sequence_length = metadata.get('sequence_length', 10)
                self.training_history = metadata.get('training_history', {})
        
        self.is_trained = True
        print(f"Modelo LSTM cargado desde: {filepath}")


class EnsembleContaminantModel(BaseContaminantModel):
    """Modelo ensemble que combina múltiples algoritmos."""
    
    def __init__(self, contaminant_name, config=None):
        super().__init__(contaminant_name, config)
        self.models = {}
        self.weights = None
        self.voting_method = self.config.get('voting_method', 'soft')
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena múltiples modelos y los combina en un ensemble.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            
        Returns:
            dict: Resultados del entrenamiento
        """
        print(f"Entrenando ensemble para {self.contaminant_name}...")
        
        # Configurar modelos base
        base_models = {
            'svm': SVMContaminantModel(self.contaminant_name, self.config.get('svm_config', {})),
            'xgboost': XGBoostContaminantModel(self.contaminant_name, self.config.get('xgboost_config', {}))
        }
        
        # Entrenar cada modelo base
        results = {}
        for name, model in base_models.items():
            print(f"Entrenando modelo base: {name}")
            try:
                model_result = model.train(X_train, y_train, X_val, y_val)
                self.models[name] = model
                results[name] = model_result
                print(f"Modelo {name} entrenado exitosamente")
            except Exception as e:
                print(f"Error entrenando modelo {name}: {e}")
                results[name] = {'error': str(e)}
        
        if not self.models:
            raise ValueError("No se pudo entrenar ningún modelo base")
        
        # Calcular pesos si se usa votación ponderada
        if self.voting_method == 'weighted' and X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        
        self.is_trained = True
        
        # Guardar información del entrenamiento
        self.training_history = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'class_distribution': pd.Series(y_train).value_counts().to_dict(),
            'base_models': list(self.models.keys()),
            'voting_method': self.voting_method,
            'individual_results': results
        }
        
        print(f"Ensemble entrenado para {self.contaminant_name} con {len(self.models)} modelos")
        return results
    
    def _calculate_weights(self, X_val, y_val):
        """Calcula pesos basados en el rendimiento en validación."""
        accuracies = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                accuracies[name] = acc
            except:
                accuracies[name] = 0.0
        
        # Normalizar pesos
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            self.weights = {name: acc / total_acc for name, acc in accuracies.items()}
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        
        print(f"Pesos calculados: {self.weights}")
    
    def predict(self, X):
        """Realiza predicciones usando el ensemble."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        if self.voting_method == 'hard':
            # Votación por mayoría
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)
            
            # Votar por clase más común
            predictions = np.array(predictions).T
            final_pred = []
            for i in range(len(predictions)):
                votes = predictions[i]
                final_pred.append(np.bincount(votes).argmax())
            
            return np.array(final_pred)
        
        else:
            # Votación suave (promedio de probabilidades)
            probabilities = self.predict_proba(X)
            return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas usando el ensemble."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        all_probas = []
        model_weights = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                all_probas.append(proba)
                
                if self.weights:
                    model_weights.append(self.weights[name])
                else:
                    model_weights.append(1.0 / len(self.models))
            except:
                continue
        
        if not all_probas:
            raise ValueError("No se pudieron obtener predicciones de ningún modelo")
        
        # Promediar probabilidades ponderadas
        weighted_probas = np.zeros_like(all_probas[0])
        total_weight = sum(model_weights)
        
        for proba, weight in zip(all_probas, model_weights):
            weighted_probas += (weight / total_weight) * proba
        
        return weighted_probas
    
    def save_model(self, filepath):
        """Guarda el modelo ensemble."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Crear directorio para el ensemble
        ensemble_dir = filepath.replace('.joblib', '_ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Guardar cada modelo base
        model_paths = {}
        for name, model in self.models.items():
            model_path = os.path.join(ensemble_dir, f"{name}.joblib")
            model.save_model(model_path)
            model_paths[name] = model_path
        
        # Guardar metadatos del ensemble
        ensemble_metadata = {
            'contaminant_name': self.contaminant_name,
            'config': self.config,
            'voting_method': self.voting_method,
            'weights': self.weights,
            'model_paths': model_paths,
            'training_history': self.training_history
        }
        
        metadata_path = os.path.join(ensemble_dir, 'ensemble_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        print(f"Ensemble guardado en: {ensemble_dir}")
    
    def load_model(self, filepath):
        """Carga un modelo ensemble."""
        ensemble_dir = filepath.replace('.joblib', '_ensemble')
        metadata_path = os.path.join(ensemble_dir, 'ensemble_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadatos del ensemble no encontrados: {metadata_path}")
        
        # Cargar metadatos
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.contaminant_name = metadata.get('contaminant_name', self.contaminant_name)
        self.config = metadata.get('config', {})
        self.voting_method = metadata.get('voting_method', 'soft')
        self.weights = metadata.get('weights')
        self.training_history = metadata.get('training_history', {})
        
        # Cargar modelos base
        model_paths = metadata.get('model_paths', {})
        self.models = {}
        
        for name, model_path in model_paths.items():
            if name == 'svm':
                model = SVMContaminantModel(self.contaminant_name)
            elif name == 'xgboost':
                model = XGBoostContaminantModel(self.contaminant_name)
            else:
                continue
            
            try:
                model.load_model(model_path)
                self.models[name] = model
            except Exception as e:
                print(f"Error cargando modelo {name}: {e}")
        
        if not self.models:
            raise ValueError("No se pudo cargar ningún modelo base del ensemble")
        
        self.is_trained = True
        print(f"Ensemble cargado desde: {ensemble_dir} con {len(self.models)} modelos") 
    
    def __init__(self, contaminant_name, config=None):
        """
        Inicializa el modelo base.
        
        Args:
            contaminant_name (str): Nombre del contaminante
            config (dict): Configuración del modelo
        """
        self.contaminant_name = contaminant_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena el modelo con los datos proporcionados."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Realiza predicciones en nuevos datos."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas."""
        pass
    
    @abstractmethod
    def save_model(self, filepath):
        """Guarda el modelo entrenado."""
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado."""
        pass
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en datos de prueba.
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        # F1-score adaptativo según número de clases
        if len(np.unique(y_test)) > 2:
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        
        return metrics


class SVMContaminantModel(BaseContaminantModel):
    """Modelo SVM optimizado para detección de contaminantes."""
    
    def __init__(self, contaminant_name, config=None):
        super().__init__(contaminant_name, config)
        self.pipeline = None
        self.best_params = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena el modelo SVM con optimización de hiperparámetros.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            
        Returns:
            dict: Resultados del entrenamiento
        """
        print(f"Entrenando modelo SVM para {self.contaminant_name}...")
        
        # Guardar nombres de características
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Configurar pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                probability=True,
                class_weight='balanced',
                random_state=self.config.get('random_state', 42)
            ))
        ])
        
        # Determinar si optimizar hiperparámetros
        optimize = self.config.get('optimize_hyperparameters', True)
        
        if optimize:
            results = self._optimize_hyperparameters(X_train, y_train)
        else:
            # Entrenar con parámetros por defecto
            self.pipeline.fit(X_train, y_train)
            results = {'best_score': None, 'best_params': None}
        
        self.model = self.pipeline
        self.is_trained = True
        
        # Guardar información del entrenamiento
        self.training_history = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'class_distribution': pd.Series(y_train).value_counts().to_dict(),
            'optimization_results': results
        }
        
        print(f"Entrenamiento completado para {self.contaminant_name}")
        return results
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimiza hiperparámetros usando GridSearchCV."""
        print("Optimizando hiperparámetros...")
        
        # Espacio de búsqueda adaptativo según tamaño del dataset
        n_samples = len(X_train)
        
        if n_samples < 50:
            param_grid = {
                'svm__C': [0.1, 1.0, 10.0],
                'svm__gamma': ['scale', 'auto'],
                'svm__kernel': ['rbf', 'linear']
            }
        elif n_samples < 200:
            param_grid = {
                'svm__C': [0.1, 1.0, 10.0, 100.0],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1],
                'svm__kernel': ['rbf', 'linear', 'poly']
            }
        else:
            param_grid = {
                'svm__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'svm__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
            }
        
        # Configurar validación cruzada
        min_samples_per_class = min(np.bincount(y_train)[np.nonzero(np.bincount(y_train))])
        n_folds = max(2, min(5, min_samples_per_class // 2))
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Realizar búsqueda
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Mejores parámetros: {self.best_params}")
        print(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        return {
            'best_score': float(grid_search.best_score_),
            'best_params': self.best_params,
            'cv_results': grid_search.cv_results_
        }
    
    def predict(self, X):
        """Realiza predicciones."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Realiza predicciones probabilísticas."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        return self.pipeline.predict_proba(X)
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo
        joblib.dump({
            'pipeline': self.pipeline,
            'contaminant_name': self.contaminant_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'best_params': self.best_params
        }, filepath)
        
        print(f"Modelo SVM guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo previamente entrenado."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.pipeline = model_data['pipeline']
        self.model = self.pipeline
        self.contaminant_name = model_data.get('contaminant_name', self.contaminant_name)
        self.config = model_data.get('config', {})
        self.feature_names = model_data.get('feature_names')
        self.training_history = model_data.get('training_history', {})
        self.best_params = model_data.get('best_params')
        self.is_trained = True
        
        print(f"Modelo SVM cargado desde: {filepath}")


class XGBoostContaminantModel(BaseContaminantModel):
    """Modelo XGBoost adaptativo para detección de contaminantes."""
    
    def __init__(self, contaminant_name, config=None):
        super().__init__(contaminant_name, config)
        self.best_params = None
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no está disponible. Instala con: pip install xgboost")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena el modelo XGBoost con optimización adaptativa.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            
        Returns:
            dict: Resultados del entrenamiento
        """
        print(f"Entrenando modelo XGBoost para {self.contaminant_name}...")
        
        # Guardar nombres de características
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Configurar parámetros base adaptativos
        base_params = self._get_adaptive_params(X_train, y_train)
        
        # Determinar si optimizar hiperparámetros
        optimize = self.config.get('optimize_hyperparameters', True)
        
        if optimize:
            results = self._optimize_hyperparameters(X_train, y_train, base_params)
        else:
            # Entrenar con parámetros base
            self.model = xgb.XGBClassifier(**base_params)
            self.model.fit(X_train, y_train)
            results = {'best_score': None, 'best_params': base_params}
        
        self.is_trained = True
        
        # Guardar información del entrenamiento
        self.training_history = {
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'class_distribution': pd.Series(y_train).value_counts().to_dict(),
            'optimization_results': results
        }
        
        print(f"Entrenamiento completado para {self.contaminant_name}")
        return results
    
    def _get_adaptive_params(self, X_train, y_train):
        """Obtiene parámetros adaptativos según las características del dataset."""
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        
        # Parámetros base
        params = {
            'random_state': self.config.get('random_state', 42),
            'verbosity': 0
        }
        
        # Configuración según tipo de problema
        if n_classes > 2:
            params.update({
                'objective': 'multi:softprob',
                'num_class': n_classes
            })
        else:
            params.update({
                'objective': 'binary:logistic'
            })
        
        # Configuración según tamaño del dataset
        if n_samples < 50:
            # Dataset pequeño - evitar sobreajuste
            params.update({
                'learning_rate': 0.01,
                'max_depth': 2,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_estimators': 50
            })
        elif n_samples < 200:
            # Dataset mediano
            params.update({
                'learning_rate': 0.03,
                'max_depth': 3,
                'min_child_weight': 2,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.05,
                'reg_lambda': 0.5,
                'n_estimators': 100
            })
        else:
            # Dataset grande
            params.update({
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 1,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'reg_alpha': 0.01,
                'reg_lambda': 0.1,
                'n_estimators': 200
            })
        
        return params
    
    def _optimize_hyperparameters(self, X_train, y_train, base_params):
        """Optimiza hiperparámetros usando GridSearchCV."""
        print("Optimizando hiperparámetros XGBoost...")
        
        n_samples = len(X_train)
        
        # Espacio de búsqueda adaptativo
        if n_samples < 50:
            param_grid = {
                'learning_rate': [0.01, 0.03],
                'max_depth': [2, 3],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }
        elif n_samples < 200:
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [2, 3, 4],
                'min_child_weight': [1, 3],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.6, 0.7, 0.8]
            }
        else:
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'n_estimators': [100, 200]
            }
        
        # Configurar validación cruzada
        min_samples_per_class = min(np.bincount(y_train)[np.nonzero(np.bincount(y_train))])
        n_folds = max(2, min(5, min_samples_per_class // 2))
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Crear estimador base
        estimator = xgb.XGBClassifier(
            **{k: v for k, v in base_params.items() if k not in param_grid.keys()}
        )
        
        # Realizar búsqueda
        grid_search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)
        
        # Actualizar parámetros con los mejores encontrados
        final_params = base_params.copy()
        final_params.update(grid_search.best_params_)
        
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(X_train, y_train)
        self.best_params = final_params
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        return {
            'best_score': float(grid_search.best_score_),
            'best_params': final_params,
            'cv_results': grid_search.cv_results_
        }
    
    