import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from src.models.base_model import BaseContaminantModel

# Importar TensorFlow si está disponible  
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


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
