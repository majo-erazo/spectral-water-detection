import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.models.base_model import BaseContaminantModel

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
