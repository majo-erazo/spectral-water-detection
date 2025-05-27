import os
import joblib
from abc import ABC, abstractmethod
import xgboost as xgb


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
