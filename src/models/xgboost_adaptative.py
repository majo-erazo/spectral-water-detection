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

# Importar XGBoost si está disponible
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


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