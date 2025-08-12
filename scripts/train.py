#!/usr/bin/env python3

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Machine Learning Core
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                   StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                           recall_score, roc_auc_score, classification_report,
                           confusion_matrix, mean_squared_error, r2_score)

# Supresión de warnings
warnings.filterwarnings('ignore')

# Imports opcionales con manejo de errores
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost no disponible - instalar con: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Configurar TensorFlow para evitar warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow no disponible - instalar con: pip install tensorflow")


class ContaminantTrainer:
    """
    Sistema completo de entrenamiento para detección de contaminantes
    """
    
    def __init__(self, datasets_dir: str = "integrated_datasets", 
                 output_dir: str = "model_results", random_state: int = 42):
        """
        Inicializar trainer
        
        Args:
            datasets_dir: Directorio con datasets integrados
            output_dir: Directorio para resultados
            random_state: Semilla para reproducibilidad
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectorios para resultados
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        self.random_state = random_state
        self.results = {}
        
        # Configuraciones específicas por contaminante
        self.contaminant_configs = self._load_contaminant_configs()
        
        print("Sistema de Entrenamiento Inicializado")
        print(f"Datasets dir: {self.datasets_dir}")
        print(f"Output dir: {self.output_dir}")
    
    def _load_contaminant_configs(self) -> Dict:
        """Cargar configuraciones específicas por contaminante"""
        
        # Configuraciones basadas en los resultados de tu investigación
        configs = {
            # Contaminantes con alta separabilidad espectral
            'diuron': {
                'priority': 'high',
                'best_algorithms': ['xgboost', 'svm'],
                'target_strategies': ['binary', 'detected_only'],
                'grid_search_intensive': True
            },
            'benzotriazole': {
                'priority': 'high', 
                'best_algorithms': ['lstm', 'svm'],
                'target_strategies': ['binary', 'ternary'],
                'grid_search_intensive': True
            },
            'hydrochlorthiazide': {
                'priority': 'high',
                'best_algorithms': ['svm', 'xgboost'],
                'target_strategies': ['binary', 'detected_only'],
                'grid_search_intensive': True
            },
            # Farmacéuticos aromáticos
            'diclofenac': {
                'priority': 'medium',
                'best_algorithms': ['svm', 'xgboost'],
                'target_strategies': ['binary', 'ternary']
            },
            'candesartan': {
                'priority': 'medium',
                'best_algorithms': ['svm', 'lstm'],
                'target_strategies': ['binary', 'detected_only']
            },
            'citalopram': {
                'priority': 'medium',
                'best_algorithms': ['xgboost', 'svm'],
                'target_strategies': ['binary', 'ternary']
            },
            # Parámetros fisicoquímicos
            'turbidity': {
                'priority': 'medium',
                'best_algorithms': ['lstm', 'xgboost'],
                'target_strategies': ['original', 'ternary']
            },
            # Casos especiales con baja detectabilidad
            'oit': {
                'priority': 'low',
                'best_algorithms': ['svm'],
                'target_strategies': ['binary'],
                'special_handling': True
            }
        }
        
        return configs
    
    def discover_datasets(self) -> Dict[str, Dict]:
        """Descubrir y catalogar datasets disponibles"""
        
        print("\nDescubriendo datasets disponibles...")
        
        datasets_info = {}
        
        for metadata_file in self.datasets_dir.glob("*_integrated_metadata.json"):
            contaminant_name = metadata_file.stem.replace('_integrated_metadata', '')
            
            # Cargar metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Verificar archivos correspondientes
            classical_file = self.datasets_dir / f"{contaminant_name}_integrated_classical.npz"
            lstm_file = self.datasets_dir / f"{contaminant_name}_integrated_lstm.npz"
            
            if classical_file.exists():
                datasets_info[contaminant_name] = {
                    'metadata': metadata,
                    'classical_file': classical_file,
                    'lstm_file': lstm_file if lstm_file.exists() else None,
                    'priority': self.contaminant_configs.get(contaminant_name, {}).get('priority', 'medium'),
                    'n_samples': metadata['n_samples'],
                    'detectability': metadata['detectability_category'],
                    'has_loq': metadata['has_loq'],
                    'strategies': metadata['strategies_available']
                }
        
        print(f"Encontrados {len(datasets_info)} contaminantes:")
        
        # Mostrar por prioridad
        for priority in ['high', 'medium', 'low']:
            priority_datasets = [name for name, info in datasets_info.items() 
                               if info['priority'] == priority]
            if priority_datasets:
                print(f"  {priority.upper()}: {len(priority_datasets)} contaminantes")
                for name in priority_datasets:
                    info = datasets_info[name]
                    print(f"    - {name}: {info['n_samples']} muestras, "
                          f"{info['detectability']}, {len(info['strategies'])} estrategias")
        
        return datasets_info
    
    def load_dataset(self, contaminant_name: str, dataset_type: str = 'classical') -> Optional[Dict]:
        """Cargar dataset específico"""
        
        if dataset_type == 'classical':
            file_path = self.datasets_dir / f"{contaminant_name}_integrated_classical.npz"
        else:
            file_path = self.datasets_dir / f"{contaminant_name}_integrated_lstm.npz"
        
        if not file_path.exists():
            print(f"Dataset no encontrado: {file_path}")
            return None
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Convertir a dict regular
            dataset = {}
            for key in data.files:
                dataset[key] = data[key]
            
            return dataset
            
        except Exception as e:
            print(f"Error cargando dataset {contaminant_name}: {e}")
            return None
    
    def train_svm_model(self, dataset: Dict, contaminant_name: str, 
                       target_strategy: str = 'binary') -> Dict:
        """Entrenar modelo SVM optimizado"""
        
        print(f"    Entrenando SVM con estrategia '{target_strategy}'...")
        
        # Seleccionar targets según estrategia
        y_train_key = f'y_train_{target_strategy}'
        y_val_key = f'y_val_{target_strategy}'
        y_test_key = f'y_test_{target_strategy}'
        
        if y_train_key not in dataset:
            print(f"      Estrategia '{target_strategy}' no disponible")
            return {'status': 'failed', 'reason': 'strategy_not_available'}
        
        X_train = dataset['X_train']
        X_val = dataset['X_val'] 
        X_test = dataset['X_test']
        y_train = dataset[y_train_key]
        y_val = dataset[y_val_key]
        y_test = dataset[y_test_key]
        
        # Verificar balance de clases
        unique_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
        if len(unique_classes) < 2:
            return {'status': 'failed', 'reason': 'insufficient_classes'}
        
        # Combinar train y val para grid search
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Configurar grid search
        config = self.contaminant_configs.get(contaminant_name, {})
        intensive_search = config.get('grid_search_intensive', False)
        
        if intensive_search:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            cv_folds = 5
        else:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf', 'poly']
            }
            cv_folds = 3
        
        # Grid search con validación cruzada estratificada
        svm_base = SVC(random_state=self.random_state, probability=True)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            svm_base, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_full, y_train_full)
        
        # Mejor modelo
        best_svm = grid_search.best_estimator_
        
        # Predicciones
        y_train_pred = best_svm.predict(X_train_full)
        y_test_pred = best_svm.predict(X_test)
        
        if hasattr(best_svm, 'predict_proba'):
            y_train_proba = best_svm.predict_proba(X_train_full)[:, 1]
            y_test_proba = best_svm.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = best_svm.decision_function(X_train_full)
            y_test_proba = best_svm.decision_function(X_test)
        
        # Métricas
        results = self._calculate_metrics(
            y_train_full, y_train_pred, y_train_proba,
            y_test, y_test_pred, y_test_proba,
            task_type='classification'
        )
        
        results.update({
            'model': best_svm,
            'model_type': 'SVM',
            'target_strategy': target_strategy,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'contaminant': contaminant_name,
            'status': 'success'
        })
        
        print(f"      SVM: Acc={results['test_accuracy']:.3f}, "
              f"AUC={results['test_auc']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        
        return results
    
    def train_xgboost_model(self, dataset: Dict, contaminant_name: str,
                           target_strategy: str = 'binary') -> Dict:
        """Entrenar modelo XGBoost optimizado"""
        
        if not XGBOOST_AVAILABLE:
            return {'status': 'failed', 'reason': 'xgboost_not_available'}
        
        print(f"    Entrenando XGBoost con estrategia '{target_strategy}'...")
        
        # Seleccionar targets
        y_train_key = f'y_train_{target_strategy}'
        y_val_key = f'y_val_{target_strategy}'
        y_test_key = f'y_test_{target_strategy}'
        
        if y_train_key not in dataset:
            return {'status': 'failed', 'reason': 'strategy_not_available'}
        
        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test'] 
        y_train = dataset[y_train_key]
        y_val = dataset[y_val_key]
        y_test = dataset[y_test_key]
        
        # Verificar clases
        unique_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
        if len(unique_classes) < 2:
            return {'status': 'failed', 'reason': 'insufficient_classes'}
        
        # Combinar para entrenamiento
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Calcular peso de clases
        class_counts = np.bincount(y_train_full.astype(int))
        if len(class_counts) > 1 and class_counts[1] > 0:
            scale_pos_weight = class_counts[0] / class_counts[1]
        else:
            scale_pos_weight = 1.0
        
        # Grid search para XGBoost
        config = self.contaminant_configs.get(contaminant_name, {})
        intensive_search = config.get('grid_search_intensive', False)
        
        if intensive_search:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        xgb_base = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            verbosity=0
        )
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=cv, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_full, y_train_full)
        
        # Mejor modelo
        best_xgb = grid_search.best_estimator_
        
        # Predicciones
        y_train_pred = best_xgb.predict(X_train_full)
        y_test_pred = best_xgb.predict(X_test)
        y_train_proba = best_xgb.predict_proba(X_train_full)[:, 1]
        y_test_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        # Métricas
        results = self._calculate_metrics(
            y_train_full, y_train_pred, y_train_proba,
            y_test, y_test_pred, y_test_proba,
            task_type='classification'
        )
        
        results.update({
            'model': best_xgb,
            'model_type': 'XGBoost',
            'target_strategy': target_strategy,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'contaminant': contaminant_name,
            'feature_importance': best_xgb.feature_importances_.tolist(),
            'status': 'success'
        })
        
        print(f"      XGBoost: Acc={results['test_accuracy']:.3f}, "
              f"AUC={results['test_auc']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        
        return results
    
    def train_lstm_model(self, dataset: Dict, contaminant_name: str,
                        target_strategy: str = 'binary') -> Dict:
        """Entrenar modelo LSTM optimizado"""
        
        if not TENSORFLOW_AVAILABLE:
            return {'status': 'failed', 'reason': 'tensorflow_not_available'}
        
        print(f"    Entrenando LSTM con estrategia '{target_strategy}'...")
        
        # Cargar dataset LSTM
        lstm_dataset = self.load_dataset(contaminant_name, 'lstm')
        if lstm_dataset is None:
            return {'status': 'failed', 'reason': 'lstm_dataset_not_found'}
        
        # Seleccionar targets
        y_train_key = f'y_train_{target_strategy}'
        y_val_key = f'y_val_{target_strategy}'
        y_test_key = f'y_test_{target_strategy}'
        
        if y_train_key not in lstm_dataset:
            return {'status': 'failed', 'reason': 'strategy_not_available'}
        
        X_train = lstm_dataset['X_train']
        X_val = lstm_dataset['X_val']
        X_test = lstm_dataset['X_test']
        y_train = lstm_dataset[y_train_key]
        y_val = lstm_dataset[y_val_key]
        y_test = lstm_dataset[y_test_key]
        
        # Verificar clases
        unique_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
        if len(unique_classes) < 2:
            return {'status': 'failed', 'reason': 'insufficient_classes'}
        
        # Arquitectura LSTM
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=0
        )
        
        # Entrenar
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predicciones
        y_train_proba = model.predict(X_train, verbose=0).flatten()
        y_test_proba = model.predict(X_test, verbose=0).flatten()
        y_train_pred = (y_train_proba > 0.5).astype(int)
        y_test_pred = (y_test_proba > 0.5).astype(int)
        
        # Métricas
        results = self._calculate_metrics(
            y_train, y_train_pred, y_train_proba,
            y_test, y_test_pred, y_test_proba,
            task_type='classification'
        )
        
        results.update({
            'model': model,
            'model_type': 'LSTM',
            'target_strategy': target_strategy,
            'contaminant': contaminant_name,
            'training_history': {
                'epochs': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            },
            'status': 'success'
        })
        
        print(f"      LSTM: Acc={results['test_accuracy']:.3f}, "
              f"AUC={results['test_auc']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        
        return results
    
    def _calculate_metrics(self, y_train_true, y_train_pred, y_train_proba,
                          y_test_true, y_test_pred, y_test_proba,
                          task_type='classification') -> Dict:
        """Calcular métricas comprehensivas"""
        
        metrics = {}
        
        if task_type == 'classification':
            # Métricas de clasificación
            metrics.update({
                'train_accuracy': float(accuracy_score(y_train_true, y_train_pred)),
                'test_accuracy': float(accuracy_score(y_test_true, y_test_pred)),
                'train_f1': float(f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
                'test_f1': float(f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)),
                'train_precision': float(precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
                'test_precision': float(precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)),
                'train_recall': float(recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0)),
                'test_recall': float(recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0))
            })
            
            # AUC si es binario
            if len(np.unique(y_test_true)) == 2:
                try:
                    metrics['train_auc'] = float(roc_auc_score(y_train_true, y_train_proba))
                    metrics['test_auc'] = float(roc_auc_score(y_test_true, y_test_proba))
                except:
                    metrics['train_auc'] = 0.5
                    metrics['test_auc'] = 0.5
            
        else:  # regresión
            metrics.update({
                'train_rmse': float(np.sqrt(mean_squared_error(y_train_true, y_train_pred))),
                'test_rmse': float(np.sqrt(mean_squared_error(y_test_true, y_test_pred))),
                'train_r2': float(r2_score(y_train_true, y_train_pred)),
                'test_r2': float(r2_score(y_test_true, y_test_pred))
            })
        
        # Gaps de overfitting
        if 'train_accuracy' in metrics:
            metrics['accuracy_gap'] = abs(metrics['train_accuracy'] - metrics['test_accuracy']) * 100
        if 'train_f1' in metrics:
            metrics['f1_gap'] = abs(metrics['train_f1'] - metrics['test_f1']) * 100
        
        # Distribuciones de predicciones
        train_unique, train_counts = np.unique(y_train_pred, return_counts=True)
        test_unique, test_counts = np.unique(y_test_pred, return_counts=True)
        
        metrics['train_pred_distribution'] = dict(zip(train_unique.astype(str), train_counts.astype(int)))
        metrics['test_pred_distribution'] = dict(zip(test_unique.astype(str), test_counts.astype(int)))
        
        # Distribuciones reales
        train_true_unique, train_true_counts = np.unique(y_train_true, return_counts=True)
        test_true_unique, test_true_counts = np.unique(y_test_true, return_counts=True)
        
        metrics['train_true_distribution'] = dict(zip(train_true_unique.astype(str), train_true_counts.astype(int)))
        metrics['test_true_distribution'] = dict(zip(test_true_unique.astype(str), test_true_counts.astype(int)))
        
        return metrics
    
    def train_contaminant(self, contaminant_name: str, datasets_info: Dict) -> Dict:
        """Entrenar todos los modelos para un contaminante"""
        
        print(f"\nEntrenando modelos para: {contaminant_name}")
        
        info = datasets_info[contaminant_name]
        config = self.contaminant_configs.get(contaminant_name, {})
        
        # Cargar dataset clásico
        dataset = self.load_dataset(contaminant_name, 'classical')
        if dataset is None:
            return {'status': 'failed', 'reason': 'dataset_not_found'}
        
        # Determinar estrategias a usar
        available_strategies = info['strategies']
        preferred_strategies = config.get('target_strategies', ['binary'])
        
        strategies_to_use = []
        for strategy in preferred_strategies:
            if strategy in available_strategies:
                strategies_to_use.append(strategy)
        
        if not strategies_to_use:
            strategies_to_use = ['binary'] if 'binary' in available_strategies else available_strategies[:1]
        
        print(f"  Estrategias: {strategies_to_use}")
        print(f"  Muestras: {info['n_samples']}, Detectabilidad: {info['detectability']}")
        
        # Entrenar modelos
        contaminant_results = {}
        
        for strategy in strategies_to_use:
            print(f"  Estrategia: {strategy}")
            
            strategy_results = {}
            
            # Determinar algoritmos a usar
            preferred_algorithms = config.get('best_algorithms', ['svm', 'xgboost', 'lstm'])
            
            for algorithm in preferred_algorithms:
                try:
                    if algorithm == 'svm':
                        result = self.train_svm_model(dataset, contaminant_name, strategy)
                    elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
                        result = self.train_xgboost_model(dataset, contaminant_name, strategy)
                    elif algorithm == 'lstm' and TENSORFLOW_AVAILABLE:
                        result = self.train_lstm_model(dataset, contaminant_name, strategy)
                    else:
                        continue
                    
                    if result['status'] == 'success':
                        strategy_results[algorithm] = result
                    
                except Exception as e:
                    print(f"      Error en {algorithm}: {e}")
                    continue
            
            if strategy_results:
                contaminant_results[strategy] = strategy_results
        
        return contaminant_results
    
    def save_results(self, contaminant_name: str, results: Dict):
        """Guardar resultados del entrenamiento"""
        
        for strategy, strategy_results in results.items():
            for algorithm, result in strategy_results.items():
                
                # Preparar reporte
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'contaminant': contaminant_name,
                    'algorithm': algorithm,
                    'target_strategy': strategy,
                    'status': result['status']
                }
                
                if result['status'] == 'success':
                    # Métricas
                    report.update({
                        'train_accuracy': result['train_accuracy'],
                        'test_accuracy': result['test_accuracy'],
                        'train_f1': result['train_f1'],
                        'test_f1': result['test_f1'],
                        'accuracy_gap': result['accuracy_gap'],
                        'f1_gap': result['f1_gap']
                    })
                    
                    if 'test_auc' in result:
                        report['test_auc'] = result['test_auc']
                    
                    if 'best_params' in result:
                        report['best_params'] = result['best_params']
                    
                    if 'cv_score' in result:
                        report['cv_score'] = result['cv_score']
                    
                    # Distribuciones
                    report['train_pred_distribution'] = result['train_pred_distribution']
                    report['test_pred_distribution'] = result['test_pred_distribution']
                    report['train_true_distribution'] = result['train_true_distribution']
                    report['test_true_distribution'] = result['test_true_distribution']
                    
                    # Guardar modelo
                    model_filename = f"{contaminant_name}_{algorithm}_{strategy}_model.pkl"
                    model_path = self.output_dir / "models" / model_filename
                    
                    try:
                        if algorithm == 'lstm':
                            # Para LSTM guardar en formato TensorFlow
                            model_dir = self.output_dir / "models" / f"{contaminant_name}_{algorithm}_{strategy}_model"
                            result['model'].save(model_dir)
                            report['model_path'] = str(model_dir)
                        else:
                            # Para SVM y XGBoost usar pickle
                            import pickle
                            with open(model_path, 'wb') as f:
                                pickle.dump(result['model'], f)
                            report['model_path'] = str(model_path)
                    except Exception as e:
                        print(f"      Warning: No se pudo guardar modelo: {e}")
                        report['model_path'] = None
                
                # Guardar reporte JSON
                report_filename = f"{contaminant_name}_{algorithm}_{strategy}_report.json"
                report_path = self.output_dir / "reports" / report_filename
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
    
    def run_training_pipeline(self, contaminants_filter: List[str] = None, 
                             priority_filter: List[str] = None):
        """Ejecutar pipeline completo de entrenamiento"""
        
        print("SISTEMA DE ENTRENAMIENTO DE CONTAMINANTES")
        print("=" * 60)
        
        # Descubrir datasets
        datasets_info = self.discover_datasets()
        
        if not datasets_info:
            print("No se encontraron datasets para entrenar")
            return
        
        # Filtrar contaminantes si se especifica
        if contaminants_filter:
            datasets_info = {name: info for name, info in datasets_info.items() 
                           if name in contaminants_filter}
        
        if priority_filter:
            datasets_info = {name: info for name, info in datasets_info.items() 
                           if info['priority'] in priority_filter}
        
        if not datasets_info:
            print("No quedan contaminantes después de aplicar filtros")
            return
        
        print(f"\nEntrenando {len(datasets_info)} contaminantes...")
        
        # Entrenar por prioridad
        training_order = sorted(datasets_info.items(), 
                              key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x[1]['priority']], 
                              reverse=True)
        
        all_results = {}
        successful_trainings = 0
        total_models = 0
        
        start_time = time.time()
        
        for i, (contaminant_name, info) in enumerate(training_order, 1):
            print(f"\n[{i}/{len(training_order)}] Procesando: {contaminant_name}")
            print(f"  Prioridad: {info['priority']}, Detectabilidad: {info['detectability']}")
            
            try:
                results = self.train_contaminant(contaminant_name, datasets_info)
                
                if results and any(results.values()):
                    # Guardar resultados
                    self.save_results(contaminant_name, results)
                    all_results[contaminant_name] = results
                    successful_trainings += 1
                    
                    # Contar modelos exitosos
                    for strategy_results in results.values():
                        total_models += len([r for r in strategy_results.values() 
                                           if r['status'] == 'success'])
                    
                    print(f"   Completado: {len(results)} estrategias")
                else:
                    print(f"   Falló el entrenamiento")
                
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        # Resumen final
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("RESUMEN DE ENTRENAMIENTO")
        print("=" * 60)
        print(f"Tiempo total: {duration/60:.1f} minutos")
        print(f"Contaminantes procesados: {successful_trainings}/{len(training_order)}")
        print(f"Modelos entrenados exitosamente: {total_models}")
        
        # Estadísticas por algoritmo
        algorithm_stats = defaultdict(int)
        strategy_stats = defaultdict(int)
        
        for results in all_results.values():
            for strategy, strategy_results in results.items():
                strategy_stats[strategy] += len(strategy_results)
                for algorithm in strategy_results.keys():
                    algorithm_stats[algorithm] += 1
        
        print(f"\nModelos por algoritmo:")
        for algorithm, count in algorithm_stats.items():
            print(f"  {algorithm.upper()}: {count} modelos")
        
        print(f"\nEstrategias utilizadas:")
        for strategy, count in strategy_stats.items():
            print(f"  {strategy}: {count} modelos")
        
        # Mejores resultados
        print(f"\nMejores resultados por contaminante:")
        
        for contaminant_name, results in all_results.items():
            best_accuracy = 0
            best_combo = None
            
            for strategy, strategy_results in results.items():
                for algorithm, result in strategy_results.items():
                    if result['status'] == 'success':
                        acc = result['test_accuracy']
                        if acc > best_accuracy:
                            best_accuracy = acc
                            best_combo = (algorithm, strategy)
            
            if best_combo:
                algorithm, strategy = best_combo
                info = datasets_info[contaminant_name]
                print(f"  {contaminant_name}: {best_accuracy:.3f} ({algorithm.upper()}, {strategy}) "
                      f"[{info['detectability']}]")
        
        print(f"\nArchivos guardados en: {self.output_dir}")
        print("  - models/: Modelos entrenados")
        print("  - reports/: Reportes detallados JSON")
        
        return all_results
    
    def generate_comparison_report(self):
        """Generar reporte de comparación de modelos"""
        
        print("\nGenerando reporte de comparación...")
        
        # Cargar todos los reportes
        reports_dir = self.output_dir / "reports"
        all_reports = []
        
        for report_file in reports_dir.glob("*_report.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    all_reports.append(report)
            except Exception as e:
                print(f"Error cargando {report_file}: {e}")
        
        if not all_reports:
            print("No se encontraron reportes para comparar")
            return
        
        # Convertir a DataFrame para análisis
        df_reports = pd.DataFrame(all_reports)
        
        # Filtrar solo éxitos
        df_success = df_reports[df_reports['status'] == 'success'].copy()
        
        if len(df_success) == 0:
            print("No hay modelos exitosos para comparar")
            return
        
        # Estadísticas por algoritmo
        print("\nEstadísticas por algoritmo:")
        for algorithm in df_success['algorithm'].unique():
            algo_data = df_success[df_success['algorithm'] == algorithm]
            
            print(f"\n{algorithm.upper()}:")
            print(f"  Modelos: {len(algo_data)}")
            print(f"  Accuracy promedio: {algo_data['test_accuracy'].mean():.3f} ± {algo_data['test_accuracy'].std():.3f}")
            print(f"  F1 promedio: {algo_data['test_f1'].mean():.3f} ± {algo_data['test_f1'].std():.3f}")
            if 'test_auc' in algo_data.columns:
                auc_data = algo_data['test_auc'].dropna()
                if len(auc_data) > 0:
                    print(f"  AUC promedio: {auc_data.mean():.3f} ± {auc_data.std():.3f}")
            print(f"  Gap promedio: {algo_data['accuracy_gap'].mean():.1f}% ± {algo_data['accuracy_gap'].std():.1f}%")
        
        # Top 10 modelos
        print(f"\nTop 10 modelos por accuracy:")
        top_models = df_success.nlargest(10, 'test_accuracy')
        
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            print(f"  {i:2d}. {row['contaminant']} ({row['algorithm'].upper()}, {row['target_strategy']}): "
                  f"Acc={row['test_accuracy']:.3f}, Gap={row['accuracy_gap']:.1f}%")
        
        # Estadísticas por estrategia
        print(f"\nEstadísticas por estrategia:")
        for strategy in df_success['target_strategy'].unique():
            strategy_data = df_success[df_success['target_strategy'] == strategy]
            print(f"  {strategy}: {len(strategy_data)} modelos, "
                  f"Acc promedio: {strategy_data['test_accuracy'].mean():.3f}")
        
        # Modelos con bajo overfitting (gap < 10%)
        low_overfitting = df_success[df_success['accuracy_gap'] < 10]
        print(f"\nModelos con bajo overfitting (gap < 10%): {len(low_overfitting)}")
        
        if len(low_overfitting) > 0:
            best_generalization = low_overfitting.nlargest(5, 'test_accuracy')
            print("Top 5 con mejor generalización:")
            for i, (_, row) in enumerate(best_generalization.iterrows(), 1):
                print(f"  {i}. {row['contaminant']} ({row['algorithm'].upper()}): "
                      f"Acc={row['test_accuracy']:.3f}, Gap={row['accuracy_gap']:.1f}%")
        
        # Guardar reporte de comparación
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(df_success),
            'algorithms_used': df_success['algorithm'].unique().tolist(),
            'strategies_used': df_success['target_strategy'].unique().tolist(),
            'contaminants_trained': df_success['contaminant'].unique().tolist(),
            'statistics': {
                'mean_accuracy': float(df_success['test_accuracy'].mean()),
                'std_accuracy': float(df_success['test_accuracy'].std()),
                'mean_f1': float(df_success['test_f1'].mean()),
                'std_f1': float(df_success['test_f1'].std()),
                'mean_gap': float(df_success['accuracy_gap'].mean()),
                'std_gap': float(df_success['accuracy_gap'].std())
            },
            'top_models': top_models[['contaminant', 'algorithm', 'target_strategy', 
                                    'test_accuracy', 'accuracy_gap']].to_dict('records'),
            'low_overfitting_count': len(low_overfitting)
        }
        
        comparison_file = self.output_dir / "model_comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        print(f"\nReporte de comparación guardado: {comparison_file}")


def main():
    """Función principal"""
    
    print("SISTEMA DE ENTRENAMIENTO DE CONTAMINANTES")
    print("=" * 60)
    
    # Verificar dependencias
    print("\nVerificando dependencias:")
    print(f"  XGBoost: {' Disponible' if XGBOOST_AVAILABLE else ' No disponible'}")
    print(f"  TensorFlow: {' Disponible' if TENSORFLOW_AVAILABLE else ' No disponible'}")
    
    try:
        # Crear trainer
        trainer = ContaminantTrainer(
            datasets_dir="integrated_datasets",
            output_dir="training_results"
        )
        
        # Opción 1: Entrenar solo contaminantes de alta prioridad
        print("\n¿Qué deseas entrenar?")
        print("1. Solo alta prioridad (recomendado)")
        print("2. Alta y media prioridad")
        print("3. Todos los contaminantes")
        print("4. Contaminantes específicos")
        
        choice = input("\nSelecciona opción (1-4) [1]: ").strip() or "1"
        
        if choice == "1":
            results = trainer.run_training_pipeline(priority_filter=['high'])
        elif choice == "2":
            results = trainer.run_training_pipeline(priority_filter=['high', 'medium'])
        elif choice == "3":
            results = trainer.run_training_pipeline()
        elif choice == "4":
            print("\nContaminantes disponibles:")
            datasets_info = trainer.discover_datasets()
            contaminant_names = list(datasets_info.keys())
            
            for i, name in enumerate(contaminant_names, 1):
                info = datasets_info[name]
                print(f"  {i:2d}. {name} ({info['priority']}, {info['detectability']})")
            
            selected = input("\nIngresa números separados por comas (ej: 1,3,5): ").strip()
            try:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                selected_contaminants = [contaminant_names[i] for i in indices if 0 <= i < len(contaminant_names)]
                
                if selected_contaminants:
                    results = trainer.run_training_pipeline(contaminants_filter=selected_contaminants)
                else:
                    print("Selección inválida")
                    return
            except:
                print("Formato inválido")
                return
        else:
            print("Opción inválida")
            return
        
        # Generar reporte de comparación
        if results:
            trainer.generate_comparison_report()
            
            print(f"\n ENTRENAMIENTO COMPLETADO")
            print(f"Revisa los resultados en: training_results/")
        else:
            print(" No se completó ningún entrenamiento")
    
    except KeyboardInterrupt:
        print("\n  Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()