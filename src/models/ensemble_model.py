import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from abc import abstractmethod
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.models.base_model import BaseContaminantModel
from src.models.svm_detector import SVMContaminantModel
from src.models.xgboost_adaptative import XGBoostContaminantModel



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
