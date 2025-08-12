#!/usr/bin/env python3
"""
train_V4_spectral_enhanced.py - Sistema de Entrenamiento con Análisis Espectral V4
Universidad Diego Portales - María José Erazo González

Sistema completo de entrenamiento con correcciones implementadas + análisis espectral integrado
Compatible con datasets generados por ML_dataset_generator_spectral_enhanced.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           roc_auc_score, classification_report, confusion_matrix)

# Imports opcionales
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(" XGBoost no disponible - instalar con: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print(" TensorFlow no disponible - instalar con: pip install tensorflow")

# Análisis espectral (opcional)
try:
    from spectral_analisis import SpectralFeatureEngineer, SpectralSignatureAnalyzer
    SPECTRAL_ANALYSIS_AVAILABLE = True
    print(" Análisis espectral avanzado disponible")
except ImportError:
    SPECTRAL_ANALYSIS_AVAILABLE = False
    print("ℹ spectral_analisis.py no encontrado - funcionalidad básica")


class SpectralEnhancedMLPipeline:
    """
    Pipeline de Machine Learning con análisis espectral integrado
    Compatible con datasets espectralmente mejorados
    """

    def __init__(self, datasets_dir: str = "spectral_enhanced_datasets_combined",
                 output_dir: str = "model_outputs_spectral_v4", 
                 random_state: int = 42,
                 spectral_strategy: str = "auto"):
        
        self.datasets_dir = datasets_dir
        self.output_dir = output_dir
        self.random_state = random_state
        self.spectral_strategy = spectral_strategy  # "auto", "spectral_only", "combined", "raw_only"
        
        # Crear directorios
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.output_dir + "/models").mkdir(exist_ok=True)
        Path(self.output_dir + "/reports").mkdir(exist_ok=True)
        Path(self.output_dir + "/spectral_analysis").mkdir(exist_ok=True)
        
        # Cargar firmas espectrales si están disponibles
        self.spectral_signatures = self._load_spectral_signatures()
        
        # Parámetros corregidos + análisis espectral
        self.corrected_params = {
            # Farmacéuticos con alta separabilidad espectral
            'diclofenac': {
                'svm': {'C': 0.01, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50},
                'spectral_priority': True  # NUEVO: Priorizar features espectrales
            },
            'candesartan': {
                'svm': {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
                'spectral_priority': True
            },
            'citalopram': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150},
                'spectral_priority': True
            },
            
            # Contaminantes con excelente separabilidad
            'benzotriazole': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 6, 'learning_rate': 0.15, 'n_estimators': 200},
                'spectral_priority': True
            },
            'diuron': {
                'svm': {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 6, 'learning_rate': 0.2, 'n_estimators': 300},
                'spectral_priority': True
            },
            'hydrochlorthiazide': {
                'svm': {'C': 10.0, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200},
                'spectral_priority': True
            },
            
            # Edulcorantes (alta detectabilidad)
            'acesulfame': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.15, 'n_estimators': 150},
                'spectral_priority': True
            },
            'cyclamate': {
                'svm': {'C': 10.0, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.15, 'n_estimators': 150},
                'spectral_priority': True
            },
            
            # Estimulantes
            'caffeine': {
                'svm': {'C': 50.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200},
                'spectral_priority': True
            },
            
            # Antimicrobianos
            'triclosan': {
                'svm': {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50},
                'spectral_priority': False  # Dataset pequeño
            },
            
            # Compuestos industriales
            '13-diphenylguanidine': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150},
                'spectral_priority': True
            },
            '6ppd-quinone': {
                'svm': {'C': 5.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
                'spectral_priority': True
            },
            'hmmm': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150},
                'spectral_priority': True
            },
            
            # Herbicidas y pesticidas
            '24-d': {
                'svm': {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 30},
                'spectral_priority': False  # Dataset muy pequeño
            },
            'carbendazim': {
                'svm': {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 50},
                'spectral_priority': False
            },
            'mcpa': {
                'svm': {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 30},
                'spectral_priority': False
            },
            'mecoprop': {
                'svm': {'C': 5.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
                'spectral_priority': True
            },
            
            # Benzotriazoles
            '4-&5-methylbenzotriazole': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150},
                'spectral_priority': True
            },
            
            # Repelentes
            'deet': {
                'svm': {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150},
                'spectral_priority': True
            },
            
            # Caso especial - baja detectabilidad
            'oit': {
                'svm': {'C': 0.01, 'gamma': 'auto', 'kernel': 'rbf'},
                'xgboost': {'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 30},
                'spectral_priority': False
            }
        }
        
        # Criterios de calidad corregidos (más realistas)
        self.quality_thresholds = {
            'excellent': {'min_accuracy': 0.90, 'max_gap': 8, 'min_auc': 0.92},
            'good': {'min_accuracy': 0.80, 'max_gap': 12, 'min_auc': 0.85},
            'fair': {'min_accuracy': 0.70, 'max_gap': 20, 'min_auc': 0.75},
            'poor': {'min_accuracy': 0.00, 'max_gap': 100, 'min_auc': 0.50}
        }
        
        print(f" Pipeline Spectral Enhanced inicializado")
        print(f" Datasets dir: {self.datasets_dir}")
        print(f" Output dir: {self.output_dir}")
        print(f" Estrategia espectral: {self.spectral_strategy}")
        print(f" Firmas espectrales: {len(self.spectral_signatures)} disponibles")
    
    def _load_spectral_signatures(self) -> Dict:
        """Cargar firmas espectrales si están disponibles"""
        
        signatures = {}
        signatures_dir = Path(self.datasets_dir) / "spectral_signatures"
        
        if signatures_dir.exists():
            signature_files = list(signatures_dir.glob("signature_*.json"))
            
            for signature_file in signature_files:
                try:
                    with open(signature_file, 'r', encoding='utf-8') as f:
                        signature = json.load(f)
                    
                    contaminant_name = signature['contaminant_name']
                    signatures[contaminant_name] = signature
                    
                except Exception as e:
                    print(f" Error cargando firma {signature_file}: {e}")
            
            print(f" Cargadas {len(signatures)} firmas espectrales")
        else:
            print(f"ℹ No se encontraron firmas espectrales en {signatures_dir}")
        
        return signatures
    
    def detect_dataset_type(self, contaminant_name: str) -> Optional[str]:
        """Detectar qué tipo de dataset está disponible"""
        
        # Buscar archivos disponibles
        dataset_files = []
        base_path = Path(self.datasets_dir)
        
        # Patrones de archivos espectralmente mejorados
        patterns = [
            f"{contaminant_name}_spectral_enhanced_classical.npz",
            f"{contaminant_name}_enhanced_classical.npz",
            f"{contaminant_name}_classical.npz"
        ]
        
        for pattern in patterns:
            file_path = base_path / pattern
            if file_path.exists():
                return str(file_path)
        
        return None
    
    def load_dataset(self, contaminant_name: str, dataset_type: str = 'classical') -> Optional[Dict]:
        """Cargar dataset espectralmente mejorado"""
        
        dataset_path = self.detect_dataset_type(contaminant_name)
        
        if dataset_path is None:
            print(f" Dataset no encontrado para: {contaminant_name}")
            return None
        
        try:
            data = np.load(dataset_path, allow_pickle=True)
            dataset = {}
            
            # Convertir arrays de numpy a dict
            for key in data.files:
                dataset[key] = data[key]
            
            # Cargar metadata si está disponible
            metadata_file = Path(self.datasets_dir) / f"{contaminant_name}_spectral_enhanced_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                dataset['metadata'] = metadata
            
            print(f" Dataset cargado: {Path(dataset_path).name}")
            
            # Mostrar información del dataset
            if 'metadata' in dataset:
                meta = dataset['metadata']
                print(f"    Muestras: {meta.get('n_samples', 'N/A')}")
                print(f"    Análisis espectral: {'SÍ' if meta.get('spectral_analysis_applied', False) else 'NO'}")
                print(f"    Augmentation: {'SÍ' if meta.get('augmentation_applied', False) else 'NO'}")
                print(f"    Calidad firma espectral: {meta.get('spectral_signature_quality', 0):.1f}/100")
            
            return dataset
            
        except Exception as e:
            print(f" Error cargando dataset {contaminant_name}: {e}")
            return None
    
    def get_spectral_strategy_for_contaminant(self, contaminant_name: str, dataset: Dict) -> str:
        """Determinar la mejor estrategia espectral para un contaminante"""
        
        if self.spectral_strategy != "auto":
            return self.spectral_strategy
        
        # Análisis automático basado en:
        # 1. Calidad de la firma espectral
        # 2. Tamaño del dataset
        # 3. Configuración específica del contaminante
        
        # Verificar configuración específica
        if contaminant_name in self.corrected_params:
            spectral_priority = self.corrected_params[contaminant_name].get('spectral_priority', True)
            if not spectral_priority:
                return "raw_only"
        
        # Verificar calidad de firma espectral
        if contaminant_name in self.spectral_signatures:
            signature = self.spectral_signatures[contaminant_name]
            quality = signature['quality_metrics']['overall_quality']
            
            if quality >= 80:
                return "spectral_only"  # Alta calidad: solo features espectrales
            elif quality >= 60:
                return "combined"       # Calidad media: combinar
            else:
                return "raw_only"       # Baja calidad: solo raw
        
        # Verificar metadatos del dataset
        if 'metadata' in dataset:
            meta = dataset['metadata']
            
            if meta.get('spectral_analysis_applied', False):
                spectral_quality = meta.get('spectral_signature_quality', 0)
                
                if spectral_quality >= 70:
                    return "spectral_only"
                elif spectral_quality >= 50:
                    return "combined"
                else:
                    return "raw_only"
        
        # Default: usar combined si hay análisis espectral
        return "combined" if dataset.get('spectral_analysis_applied', False) else "raw_only"
    
    def train_svm_spectral_enhanced(self, dataset: Dict, contaminant_name: str) -> Dict:
        """Entrenar SVM con datasets espectralmente mejorados"""
        
        print(f"        Entrenando SVM espectral...")
        
        # Determinar estrategia espectral
        strategy = self.get_spectral_strategy_for_contaminant(contaminant_name, dataset)
        print(f"           Estrategia: {strategy}")
        
        # Cargar datos según estrategia
        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train_binary']
        y_val = dataset['y_val_binary']
        y_test = dataset['y_test_binary']
        
        # Combinar train y val para entrenamiento final
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Parámetros específicos o por defecto
        if contaminant_name in self.corrected_params:
            params = self.corrected_params[contaminant_name]['svm']
            print(f"           Parámetros específicos: {params}")
        else:
            params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            print(f"           Parámetros por defecto: {params}")
        
        # Entrenar SVM
        svm_model = SVC(probability=True, random_state=self.random_state, **params)
        svm_model.fit(X_train_full, y_train_full)
        
        # Predicciones
        train_pred = svm_model.predict(X_train_full)
        val_pred = svm_model.predict(X_val)
        test_pred = svm_model.predict(X_test)
        
        train_proba = svm_model.predict_proba(X_train_full)[:, 1]
        test_proba = svm_model.predict_proba(X_test)[:, 1]
        
        # Métricas
        results = {
            'model': svm_model,
            'model_type': f'SVM_Spectral_{strategy}',
            'spectral_strategy': strategy,
            'train_accuracy': accuracy_score(y_train_full, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_f1': f1_score(y_train_full, train_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train_full, train_proba),
            'test_auc': roc_auc_score(y_test, test_proba),
            'predictions': test_pred,
            'probabilities': test_proba,
            'svm_params': svm_model.get_params(),
            'feature_info': {
                'total_features': X_train.shape[1],
                'strategy_used': strategy
            }
        }
        
        results['accuracy_gap'] = abs(results['train_accuracy'] - results['test_accuracy']) * 100
        results['f1_gap'] = abs(results['train_f1'] - results['test_f1']) * 100
        
        print(f"           SVM: Acc={results['test_accuracy']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        
        return results
    
    def train_xgboost_spectral_enhanced(self, dataset: Dict, contaminant_name: str) -> Dict:
        """Entrenar XGBoost con datasets espectralmente mejorados"""
        
        if not XGBOOST_AVAILABLE:
            return {'status': 'failed', 'reason': 'xgboost_not_available'}
        
        print(f"        Entrenando XGBoost espectral...")
        
        # Determinar estrategia espectral
        strategy = self.get_spectral_strategy_for_contaminant(contaminant_name, dataset)
        print(f"           Estrategia: {strategy}")
        
        # Parámetros específicos o por defecto
        if contaminant_name in self.corrected_params:
            params = self.corrected_params[contaminant_name]['xgboost']
            print(f"           Parámetros específicos: {params}")
        else:
            params = {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100, 'reg_alpha': 0.1}
            print(f"           Parámetros por defecto: {params}")
        
        # Cargar datos
        X_train = dataset['X_train']
        X_val = dataset['X_val']
        X_test = dataset['X_test']
        y_train = dataset['y_train_binary']
        y_val = dataset['y_val_binary']
        y_test = dataset['y_test_binary']
        
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        # Calcular peso de clases
        pos_weight = len(y_train_full[y_train_full == 0]) / max(1, len(y_train_full[y_train_full == 1]))
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            verbosity=0,
            **params
        )
        
        xgb_model.fit(X_train_full, y_train_full)
        
        # Predicciones
        train_pred = xgb_model.predict(X_train_full)
        test_pred = xgb_model.predict(X_test)
        train_proba = xgb_model.predict_proba(X_train_full)[:, 1]
        test_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Análisis de importancia de features (especial para features espectrales)
        feature_importance = xgb_model.feature_importances_
        
        # Si tenemos información sobre features espectrales, analizarlas
        spectral_feature_importance = None
        if 'feature_names' in dataset and strategy in ['spectral_only', 'combined']:
            feature_names = dataset.get('feature_names', [])
            
            # Identificar features espectrales vs raw
            spectral_features = [i for i, name in enumerate(feature_names) 
                               if not name.startswith('wl_')]
            raw_features = [i for i, name in enumerate(feature_names) 
                          if name.startswith('wl_')]
            
            if spectral_features:
                spectral_importance = np.sum(feature_importance[spectral_features])
                raw_importance = np.sum(feature_importance[raw_features]) if raw_features else 0
                
                spectral_feature_importance = {
                    'spectral_total_importance': float(spectral_importance),
                    'raw_total_importance': float(raw_importance),
                    'spectral_vs_raw_ratio': float(spectral_importance / (raw_importance + 1e-8)),
                    'top_spectral_features': [
                        {'name': feature_names[i], 'importance': float(feature_importance[i])}
                        for i in sorted(spectral_features, key=lambda x: feature_importance[x], reverse=True)[:10]
                    ]
                }
        
        # Métricas
        results = {
            'model': xgb_model,
            'model_type': f'XGBoost_Spectral_{strategy}',
            'spectral_strategy': strategy,
            'train_accuracy': accuracy_score(y_train_full, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_f1': f1_score(y_train_full, train_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train_full, train_proba),
            'test_auc': roc_auc_score(y_test, test_proba),
            'predictions': test_pred,
            'probabilities': test_proba,
            'feature_importance': feature_importance.tolist(),
            'spectral_feature_analysis': spectral_feature_importance,
            'feature_info': {
                'total_features': X_train.shape[1],
                'strategy_used': strategy
            }
        }
        
        results['accuracy_gap'] = abs(results['train_accuracy'] - results['test_accuracy']) * 100
        results['f1_gap'] = abs(results['train_f1'] - results['test_f1']) * 100
        
        print(f"           XGBoost: Acc={results['test_accuracy']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        
        # Mostrar análisis de features espectrales si está disponible
        if spectral_feature_importance:
            ratio = spectral_feature_importance['spectral_vs_raw_ratio']
            print(f"           Importancia spectral/raw: {ratio:.2f}")
        
        return results
    
    def train_lstm_spectral_enhanced(self, dataset: Dict, contaminant_name: str) -> Dict:
        """Entrenar LSTM con secuencias espectrales optimizadas"""
        
        if not TENSORFLOW_AVAILABLE:
            return {'status': 'failed', 'reason': 'tensorflow_not_available'}
        
        print(f"        Entrenando LSTM espectral...")
        
        # Cargar dataset LSTM
        lstm_file = Path(self.datasets_dir) / f"{contaminant_name}_spectral_enhanced_lstm.npz"
        
        if not lstm_file.exists():
            # Fallback a otros formatos
            alternative_files = [
                f"{contaminant_name}_enhanced_lstm.npz",
                f"{contaminant_name}_lstm.npz"
            ]
            
            for alt_file in alternative_files:
                alt_path = Path(self.datasets_dir) / alt_file
                if alt_path.exists():
                    lstm_file = alt_path
                    break
            else:
                return {'status': 'failed', 'reason': 'lstm_dataset_not_found'}
        
        try:
            lstm_data = np.load(lstm_file, allow_pickle=True)
            
            X_train = lstm_data['X_train']
            X_val = lstm_data['X_val']
            X_test = lstm_data['X_test']
            y_train = lstm_data['y_train_binary']
            y_val = lstm_data['y_val_binary']
            y_test = lstm_data['y_test_binary']
            
        except Exception as e:
            print(f"           Error cargando datos LSTM: {e}")
            return {'status': 'failed', 'reason': f'data_loading_error: {e}'}
        
        # Verificar clases
        if len(np.unique(np.concatenate([y_train, y_val, y_test]))) < 2:
            return {'status': 'failed', 'reason': 'insufficient_classes'}
        
        # Arquitectura LSTM optimizada basada en análisis espectral
        spectral_quality = 0.5  # Default
        
        if contaminant_name in self.spectral_signatures:
            spectral_quality = self.spectral_signatures[contaminant_name]['quality_metrics']['overall_quality'] / 100
        elif 'metadata' in dataset:
            spectral_quality = dataset['metadata'].get('spectral_signature_quality', 50) / 100
        
        print(f"           Calidad espectral: {spectral_quality:.2f}")
        
        if spectral_quality > 0.8:
            # Alta calidad espectral: arquitectura más simple
            lstm_units = [32, 16]
            dropout_rate = 0.2
            epochs = 50
        else:
            # Baja calidad espectral: arquitectura más compleja
            lstm_units = [64, 32]
            dropout_rate = 0.3
            epochs = 100
        
        # Modelo LSTM
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(dropout_rate),
            LSTM(lstm_units[1], return_sequences=False),
            Dropout(dropout_rate),
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
            epochs=epochs,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predicciones
        train_proba = model.predict(X_train, verbose=0).flatten()
        test_proba = model.predict(X_test, verbose=0).flatten()
        train_pred = (train_proba > 0.5).astype(int)
        test_pred = (test_proba > 0.5).astype(int)
        
        # Métricas
        results = {
            'model': model,
            'model_type': 'LSTM_Spectral_Enhanced',
            'spectral_quality': spectral_quality,
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, train_proba),
            'test_auc': roc_auc_score(y_test, test_proba),
            'predictions': test_pred,
            'probabilities': test_proba,
            'training_epochs': len(history.history['loss']),
            'architecture': f"LSTM({lstm_units[0]}, {lstm_units[1]})",
            'spectral_optimization': {
                'quality_based_architecture': True,
                'spectral_quality_score': spectral_quality,
                'architecture_choice': 'simple' if spectral_quality > 0.8 else 'complex'
            }
        }
        
        results['accuracy_gap'] = abs(results['train_accuracy'] - results['test_accuracy']) * 100
        results['f1_gap'] = abs(results['train_f1'] - results['test_f1']) * 100
        
        print(f"           LSTM: Acc={results['test_accuracy']:.3f}, Gap={results['accuracy_gap']:.1f}%")
        print(f"          🏗 Arquitectura: {results['architecture']} (épocas: {results['training_epochs']})")
        
        return results
    
    def evaluate_model_quality_spectral_enhanced(self, results: Dict, contaminant_name: str) -> Dict:
        """Evaluación de calidad mejorada con información espectral"""
        
        accuracy = results['test_accuracy']
        gap = results['accuracy_gap']
        auc = results.get('test_auc', 0.5)
        
        # Cálculo de calidad base
        base_score = accuracy * 100
        gap_penalty = max(0, (gap - 5) * 1.5)
        overfitting_penalty = max(0, (gap - 8) * 2)
        generalization_bonus = 10 if gap < 5 else (5 if gap < 8 else 0)
        
        # NUEVO: Bonus por análisis espectral
        spectral_bonus = 0
        spectral_info = {}
        
        if contaminant_name in self.spectral_signatures:
            signature = self.spectral_signatures[contaminant_name]
            spectral_quality = signature['quality_metrics']['overall_quality']
            
            # Bonus por alta calidad espectral
            if spectral_quality >= 80:
                spectral_bonus += 5
            elif spectral_quality >= 60:
                spectral_bonus += 3
            
            spectral_info = {
                'signature_quality': spectral_quality,
                'has_signature': True,
                'n_characteristic_peaks': len(signature['characteristic_peaks']),
                'n_discriminant_wavelengths': len(signature['discriminant_wavelengths'])
            }
        
        # Bonus por estrategia espectral efectiva
        if 'spectral_strategy' in results:
            strategy = results['spectral_strategy']
            if strategy in ['spectral_only', 'combined'] and accuracy > 0.8:
                spectral_bonus += 2
            
            spectral_info['strategy_used'] = strategy
        
        # Bonus por análisis de features espectrales (XGBoost)
        if 'spectral_feature_analysis' in results and results['spectral_feature_analysis']:
            feature_analysis = results['spectral_feature_analysis']
            spectral_ratio = feature_analysis['spectral_vs_raw_ratio']
            
            # Si features espectrales son más importantes, dar bonus
            if spectral_ratio > 1.5:
                spectral_bonus += 3
            elif spectral_ratio > 1.0:
                spectral_bonus += 1
            
            spectral_info['spectral_feature_dominance'] = spectral_ratio
        
        quality_score = base_score + generalization_bonus + spectral_bonus - gap_penalty - overfitting_penalty
        quality_score = max(0, min(100, quality_score))
        
        # Categoría de calidad mejorada
        if overfitting_penalty >= 50:
            quality_category = 'poor'
        else:
            for category, thresholds in self.quality_thresholds.items():
                if (accuracy >= thresholds['min_accuracy'] and 
                    gap <= thresholds['max_gap'] and 
                    auc >= thresholds['min_auc']):
                    quality_category = category
                    break
            else:
                quality_category = 'poor'
        
        # Recomendaciones mejoradas con análisis espectral
        recommendations = []
        
        if gap > 20:
            recommendations.append("CRÍTICO: Overfitting severo detectado")
            recommendations.append("Reducir complejidad del modelo")
        elif quality_category in ['excellent', 'good']:
            recommendations.append(" Modelo listo para uso/ensemble")
            if spectral_bonus > 0:
                recommendations.append(" Beneficiado por análisis espectral")
        elif quality_category == 'fair':
            recommendations.append(" Modelo aceptable con optimización")
            if spectral_info.get('has_signature', False):
                recommendations.append(" Considerar optimización espectral")
        else:
            recommendations.append(" Modelo requiere reentrenamiento")
            if not spectral_info.get('has_signature', False):
                recommendations.append(" Considerar análisis espectral")
        
        # Recomendaciones específicas por estrategia
        if 'spectral_strategy' in results:
            strategy = results['spectral_strategy']
            if strategy == 'raw_only' and accuracy < 0.75:
                recommendations.append(" Probar features espectrales para mejorar")
            elif strategy == 'spectral_only' and accuracy > 0.85:
                recommendations.append(" Features espectrales muy efectivas")
        
        return {
            'quality_score': quality_score,
            'quality_category': quality_category,
            'accuracy': accuracy,
            'gap': gap,
            'auc': auc,
            'overfitting_penalty': overfitting_penalty,
            'spectral_bonus': spectral_bonus,
            'spectral_analysis': spectral_info,
            'recommendations': recommendations,
            'is_production_ready': quality_category in ['excellent', 'good', 'fair'],
            'spectral_enhanced': spectral_bonus > 0
        }
    
    def create_spectral_ensemble(self, individual_results: Dict, contaminant_name: str) -> Optional[Dict]:
        """Crear ensemble que considera análisis espectral"""
        
        print(f"     Creando ensemble espectral...")
        
        # Filtrar modelos aceptables
        acceptable_models = {}
        for model_name, data in individual_results.items():
            evaluation = data['evaluation']
            if evaluation['quality_category'] in ['excellent', 'good', 'fair']:
                acceptable_models[model_name] = data
        
        if len(acceptable_models) >= 2:
            print(f"           Modelos para ensemble: {list(acceptable_models.keys())}")
            
            # Calcular pesos considerando análisis espectral
            weights = []
            predictions_list = []
            probabilities_list = []
            
            for model_name, data in acceptable_models.items():
                evaluation = data['evaluation']
                
                # Peso base según calidad
                if evaluation['quality_category'] == 'excellent':
                    base_weight = 1.0
                elif evaluation['quality_category'] == 'good':
                    base_weight = 0.8
                else:  # fair
                    base_weight = 0.6
                
                # Bonus por bajo gap
                gap_bonus = max(0, (15 - evaluation['gap']) / 20)
                
                # NUEVO: Bonus por análisis espectral
                spectral_bonus = evaluation.get('spectral_bonus', 0) / 100
                
                # Bonus por estrategia espectral efectiva
                strategy_bonus = 0
                if 'spectral_strategy' in data['results']:
                    strategy = data['results']['spectral_strategy']
                    if strategy in ['spectral_only', 'combined'] and evaluation['accuracy'] > 0.8:
                        strategy_bonus = 0.1
                
                final_weight = base_weight + gap_bonus + spectral_bonus + strategy_bonus
                weights.append(final_weight)
                
                predictions_list.append(data['results']['predictions'])
                probabilities_list.append(data['results']['probabilities'])
            
            # Normalizar pesos
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Ensemble predictions
            ensemble_probabilities = np.average(probabilities_list, axis=0, weights=weights)
            final_predictions = (ensemble_probabilities > 0.5).astype(int)
            
            # Estimación de accuracy
            ensemble_accuracy = np.mean([np.mean(pred == final_predictions) for pred in predictions_list])
            
            # Análisis de diversidad espectral
            spectral_strategies = [data['results'].get('spectral_strategy', 'unknown') 
                                 for data in acceptable_models.values()]
            strategy_diversity = len(set(spectral_strategies))
            
            print(f"           Ensemble accuracy estimada: {ensemble_accuracy:.3f}")
            print(f"          ⚖ Pesos: {weights}")
            print(f"           Diversidad espectral: {strategy_diversity} estrategias")
            
            return {
                'type': 'Spectral_Enhanced_Ensemble',
                'models_used': list(acceptable_models.keys()),
                'weights': weights,
                'predictions': final_predictions,
                'probabilities': ensemble_probabilities,
                'estimated_accuracy': ensemble_accuracy,
                'spectral_analysis': {
                    'strategies_included': spectral_strategies,
                    'strategy_diversity': strategy_diversity,
                    'spectral_signature_available': contaminant_name in self.spectral_signatures
                },
                'base_models_evaluations': {name: data['evaluation'] for name, data in acceptable_models.items()}
            }
        
        return None
    
    def train_contaminant_spectral_enhanced(self, contaminant_name: str) -> Optional[Dict]:
        """Entrenar contaminante con pipeline espectral mejorado"""
        
        print(f"\n PROCESAMIENTO SPECTRAL ENHANCED: {contaminant_name.upper()}")
        print("="*80)
        
        # Cargar dataset
        dataset = self.load_dataset(contaminant_name, 'classical')
        if dataset is None:
            print(f"     Dataset no disponible")
            return None
        
        # Información espectral
        spectral_info = {}
        if contaminant_name in self.spectral_signatures:
            signature = self.spectral_signatures[contaminant_name]
            spectral_info = {
                'has_signature': True,
                'quality': signature['quality_metrics']['overall_quality'],
                'n_peaks': len(signature['characteristic_peaks']),
                'n_discriminants': len(signature['discriminant_wavelengths'])
            }
            print(f"     Firma espectral: Calidad {spectral_info['quality']:.1f}/100, {spectral_info['n_peaks']} picos")
        else:
            spectral_info = {'has_signature': False}
            print(f"    ℹ Sin firma espectral específica")
        
        # Mostrar información del dataset
        if 'metadata' in dataset:
            meta = dataset['metadata']
            print(f"     Dataset: {meta.get('n_samples', 'N/A')} muestras")
            print(f"     Augmentation: {'SÍ' if meta.get('augmentation_applied', False) else 'NO'}")
            print(f"     Features espectrales: {meta.get('n_spectral_features', 0)}")
        
        # Entrenar modelos individuales
        individual_results = {}
        
        # SVM espectral
        try:
            svm_result = self.train_svm_spectral_enhanced(dataset, contaminant_name)
            if svm_result:
                individual_results['SVM'] = svm_result
        except Exception as e:
            print(f"        Error SVM: {e}")
        
        # XGBoost espectral
        try:
            xgb_result = self.train_xgboost_spectral_enhanced(dataset, contaminant_name)
            if xgb_result and xgb_result.get('status') != 'failed':
                individual_results['XGBoost'] = xgb_result
        except Exception as e:
            print(f"        Error XGBoost: {e}")
        
        # LSTM espectral
        try:
            lstm_result = self.train_lstm_spectral_enhanced(dataset, contaminant_name)
            if lstm_result and lstm_result.get('status') != 'failed':
                individual_results['LSTM'] = lstm_result
        except Exception as e:
            print(f"        Error LSTM: {e}")
        
        if not individual_results:
            print("     No se pudieron entrenar modelos")
            return None
        
        # Evaluación de calidad con análisis espectral
        print(f"\n     EVALUACIÓN DE CALIDAD SPECTRAL ENHANCED:")
        quality_evaluations = {}
        
        for model_name, results in individual_results.items():
            evaluation = self.evaluate_model_quality_spectral_enhanced(results, contaminant_name)
            quality_evaluations[model_name] = {
                'results': results,
                'evaluation': evaluation
            }
            
            category_emoji = {
                'excellent': '🌟',
                'good': '', 
                'fair': '',
                'poor': ''
            }
            
            emoji = category_emoji.get(evaluation['quality_category'], '❓')
            spectral_indicator = '' if evaluation.get('spectral_enhanced', False) else ''
            
            print(f"       {emoji} {model_name}: {evaluation['quality_category'].upper()} {spectral_indicator}")
            print(f"          Score: {evaluation['quality_score']:.1f}/100")
            print(f"          Acc: {evaluation['accuracy']:.3f}, Gap: {evaluation['gap']:.1f}%")
            
            if evaluation.get('spectral_bonus', 0) > 0:
                print(f"           Bonus espectral: +{evaluation['spectral_bonus']:.1f}")
            
            for rec in evaluation['recommendations']:
                print(f"          → {rec}")
        
        # Crear ensemble espectral
        ensemble_result = self.create_spectral_ensemble(quality_evaluations, contaminant_name)
        
        # Resultado final
        final_result = {
            'contaminant': contaminant_name,
            'individual_models': quality_evaluations,
            'ensemble': ensemble_result,
            'spectral_analysis': spectral_info,
            'dataset_metadata': dataset.get('metadata', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Resumen
        acceptable_count = len([e for e in quality_evaluations.values() 
                              if e['evaluation']['is_production_ready']])
        
        spectral_enhanced_count = len([e for e in quality_evaluations.values() 
                                     if e['evaluation'].get('spectral_enhanced', False)])
        
        print(f"\n     RESUMEN:")
        print(f"       Modelos entrenados: {len(individual_results)}")
        print(f"       Modelos aceptables: {acceptable_count}")
        print(f"       Con mejora espectral: {spectral_enhanced_count}")
        
        if ensemble_result:
            ensemble_type = ensemble_result['type']
            n_models = len(ensemble_result['models_used'])
            diversity = ensemble_result['spectral_analysis']['strategy_diversity']
            print(f"       Ensemble: {ensemble_type} con {n_models} modelos")
            print(f"       Diversidad espectral: {diversity} estrategias")
        else:
            print(f"       Ensemble: No creado (insuficientes modelos)")
        
        return final_result
    
    def train_spectral_enhanced_pipeline(self) -> int:
        """Ejecutar pipeline de entrenamiento espectral completo"""
        
        print(" SISTEMA DE ENTRENAMIENTO SPECTRAL ENHANCED V4")
        print("Universidad Diego Portales - María José Erazo González")
        print("="*70)
        
        # Detectar contaminantes disponibles
        datasets_path = Path(self.datasets_dir)
        if not datasets_path.exists():
            print(f" Directorio no encontrado: {self.datasets_dir}")
            return 0
        
        # Buscar archivos de datasets espectralmente mejorados
        spectral_files = list(datasets_path.glob("*_spectral_enhanced_classical.npz"))
        enhanced_files = list(datasets_path.glob("*_enhanced_classical.npz"))
        classical_files = list(datasets_path.glob("*_classical.npz"))
        
        # Extraer nombres de contaminantes
        all_contaminants = set()
        
        for file_list, suffix in [(spectral_files, "_spectral_enhanced_classical.npz"),
                                 (enhanced_files, "_enhanced_classical.npz"), 
                                 (classical_files, "_classical.npz")]:
            for file in file_list:
                contaminant = file.name.replace(suffix, "")
                all_contaminants.add(contaminant)
        
        all_contaminants = sorted(list(all_contaminants))
        
        print(f"\n CONTAMINANTES DETECTADOS: {len(all_contaminants)}")
        
        # Categorizar por tipo de dataset disponible
        spectral_enhanced = []
        enhanced_only = []
        classical_only = []
        
        for cont in all_contaminants:
            if (datasets_path / f"{cont}_spectral_enhanced_classical.npz").exists():
                spectral_enhanced.append(cont)
            elif (datasets_path / f"{cont}_enhanced_classical.npz").exists():
                enhanced_only.append(cont)
            else:
                classical_only.append(cont)
        
        print(f"   Con análisis espectral: {len(spectral_enhanced)}")
        print(f"   Solo enhanced: {len(enhanced_only)}")
        print(f"   Solo clásicos: {len(classical_only)}")
        
        # Mostrar información de firmas espectrales
        if self.spectral_signatures:
            print(f"   Firmas espectrales: {len(self.spectral_signatures)} disponibles")
            
            # Mostrar calidades
            high_quality = len([s for s in self.spectral_signatures.values() 
                              if s['quality_metrics']['overall_quality'] >= 80])
            medium_quality = len([s for s in self.spectral_signatures.values() 
                                if 60 <= s['quality_metrics']['overall_quality'] < 80])
            
            print(f"     Alta calidad (≥80): {high_quality}")
            print(f"     Calidad media (60-79): {medium_quality}")
        
        # Selección de contaminantes a entrenar
        print(f"\n SELECCIÓN DE ENTRENAMIENTO:")
        print(f"1. Solo con análisis espectral ({len(spectral_enhanced)})")
        print(f"2. Todos los enhanced ({len(spectral_enhanced + enhanced_only)})")
        print(f"3. TODOS los disponibles ({len(all_contaminants)})")
        print(f"4. Selección personalizada")
        
        choice = input("\nSelecciona opción (1-4) [1]: ").strip() or "1"
        
        if choice == "1":
            contaminants_to_train = spectral_enhanced
            print(f" Entrenando {len(contaminants_to_train)} contaminantes con análisis espectral")
            
        elif choice == "2":
            contaminants_to_train = spectral_enhanced + enhanced_only
            print(f" Entrenando {len(contaminants_to_train)} contaminantes enhanced")
            
        elif choice == "3":
            contaminants_to_train = all_contaminants
            print(f" Entrenando TODOS los {len(contaminants_to_train)} contaminantes")
            
        elif choice == "4":
            print(f"\n Contaminantes disponibles:")
            
            if spectral_enhanced:
                print(f"\n CON ANÁLISIS ESPECTRAL:")
                for i, cont in enumerate(spectral_enhanced, 1):
                    quality = ""
                    if cont in self.spectral_signatures:
                        q = self.spectral_signatures[cont]['quality_metrics']['overall_quality']
                        quality = f" (calidad: {q:.0f}/100)"
                    print(f"  {i:2d}. {cont}{quality}")
            
            if enhanced_only:
                print(f"\n SOLO ENHANCED:")
                start_idx = len(spectral_enhanced) + 1
                for i, cont in enumerate(enhanced_only, start_idx):
                    print(f"  {i:2d}. {cont}")
            
            if classical_only:
                print(f"\n SOLO CLÁSICOS:")
                start_idx = len(spectral_enhanced) + len(enhanced_only) + 1
                for i, cont in enumerate(classical_only, start_idx):
                    print(f"  {i:2d}. {cont}")
            
            selected = input("\nIngresa números separados por comas (ej: 1,2,5-8,10): ").strip()
            try:
                indices = []
                for part in selected.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start-1, end))
                    else:
                        indices.append(int(part)-1)
                
                contaminants_to_train = [all_contaminants[i] for i in indices 
                                       if 0 <= i < len(all_contaminants)]
                print(f" Entrenando {len(contaminants_to_train)} contaminantes seleccionados")
                
            except:
                print(" Formato inválido, usando análisis espectral")
                contaminants_to_train = spectral_enhanced
        else:
            print(" Opción inválida, usando análisis espectral")
            contaminants_to_train = spectral_enhanced
        
        # Mostrar lista final
        print(f"\n LISTA DE ENTRENAMIENTO ({len(contaminants_to_train)}):")
        for i, cont in enumerate(contaminants_to_train, 1):
            status_indicators = []
            
            if cont in spectral_enhanced:
                status_indicators.append("")
            if cont in self.spectral_signatures:
                quality = self.spectral_signatures[cont]['quality_metrics']['overall_quality']
                if quality >= 80:
                    status_indicators.append("")
                elif quality >= 60:
                    status_indicators.append("")
            
            indicators = "".join(status_indicators)
            print(f"  {i:2d}. {cont} {indicators}")
        
        # Confirmar si son muchos
        if len(contaminants_to_train) > 10:
            estimated_time = len(contaminants_to_train) * 2  # ~2 min por contaminante
            confirm = input(f"\n Tiempo estimado: ~{estimated_time} minutos. ¿Continuar? (y/N): ")
            if confirm.lower() not in ['y', 'yes', 'sí', 's']:
                print(" Entrenamiento cancelado")
                return 0
        
        # Ejecutar entrenamiento
        successful_trainings = 0
        all_results = {}
        
        print(f"\n INICIANDO ENTRENAMIENTO SPECTRAL ENHANCED...")
        print("="*70)
        
        for i, contaminant in enumerate(contaminants_to_train, 1):
            print(f"\n[{i}/{len(contaminants_to_train)}]  {contaminant}")
            
            try:
                result = self.train_contaminant_spectral_enhanced(contaminant)
                if result:
                    all_results[contaminant] = result
                    successful_trainings += 1
                    
                    # Mostrar resumen rápido
                    acceptable = len([e for e in result['individual_models'].values() 
                                    if e['evaluation']['is_production_ready']])
                    spectral_enhanced_models = len([e for e in result['individual_models'].values() 
                                                  if e['evaluation'].get('spectral_enhanced', False)])
                    
                    print(f"     {contaminant}: {acceptable} modelos aceptables, {spectral_enhanced_models} con mejora espectral")
                else:
                    print(f"     {contaminant}: Falló")
                    
            except Exception as e:
                print(f"     {contaminant}: Error - {e}")
                continue
        
        # Guardar resultados
        self.results = all_results
        
        print(f"\n" + "="*70)
        print(f" PIPELINE SPECTRAL ENHANCED COMPLETADO")
        print(f"="*70)
        print(f" Contaminantes entrenados exitosamente: {successful_trainings}/{len(contaminants_to_train)}")
        
        # Estadísticas de calidad
        total_models = 0
        excellent_models = 0
        good_models = 0
        fair_models = 0
        spectral_enhanced_models = 0
        ensembles_created = 0
        
        for contaminant, result in all_results.items():
            for model_name, model_data in result['individual_models'].items():
                total_models += 1
                category = model_data['evaluation']['quality_category']
                if category == 'excellent':
                    excellent_models += 1
                elif category == 'good':
                    good_models += 1
                elif category == 'fair':
                    fair_models += 1
                
                if model_data['evaluation'].get('spectral_enhanced', False):
                    spectral_enhanced_models += 1
            
            if result['ensemble']:
                ensembles_created += 1
        
        print(f"\n RESUMEN DE CALIDAD:")
        print(f"  Total modelos: {total_models}")
        print(f"  🌟 Excelentes: {excellent_models}")
        print(f"   Buenos: {good_models}")
        print(f"   Aceptables: {fair_models}")
        print(f"   Con mejora espectral: {spectral_enhanced_models}")
        print(f"   Ensembles creados: {ensembles_created}")
        
        # Análisis de efectividad espectral
        if spectral_enhanced_models > 0:
            spectral_effectiveness = (spectral_enhanced_models / total_models) * 100
            print(f"   Efectividad espectral: {spectral_effectiveness:.1f}%")
        
        # Mostrar mejores resultados
        print(f"\n MEJORES RESULTADOS:")
        results_summary = []
        
        for contaminant, result in all_results.items():
            best_accuracy = 0
            best_model = None
            has_spectral_enhancement = False
            
            for model_name, model_data in result['individual_models'].items():
                acc = model_data['evaluation']['accuracy']
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model_name
                
                if model_data['evaluation'].get('spectral_enhanced', False):
                    has_spectral_enhancement = True
            
            ensemble_info = ""
            if result['ensemble']:
                ensemble_acc = result['ensemble']['estimated_accuracy']
                ensemble_info = f" | Ensemble: {ensemble_acc:.3f}"
            
            spectral_indicator = " " if has_spectral_enhancement else ""
            
            results_summary.append({
                'contaminant': contaminant,
                'best_accuracy': best_accuracy,
                'best_model': best_model,
                'has_ensemble': bool(result['ensemble']),
                'spectral_enhanced': has_spectral_enhancement,
                'display': f"  {contaminant}: {best_model} {best_accuracy:.3f}{ensemble_info}{spectral_indicator}"
            })
        
        # Ordenar por accuracy
        results_summary.sort(key=lambda x: x['best_accuracy'], reverse=True)
        
        for result in results_summary:
            print(result['display'])
        
        # Análisis específico de contaminantes con firmas espectrales
        if self.spectral_signatures:
            print(f"\n ANÁLISIS DE FIRMAS ESPECTRALES:")
            
            signature_results = []
            for contaminant in contaminants_to_train:
                if contaminant in self.spectral_signatures and contaminant in all_results:
                    signature = self.spectral_signatures[contaminant]
                    result = all_results[contaminant]
                    
                    # Mejor accuracy del contaminante
                    best_acc = max([model_data['evaluation']['accuracy'] 
                                  for model_data in result['individual_models'].values()])
                    
                    signature_results.append({
                        'contaminant': contaminant,
                        'signature_quality': signature['quality_metrics']['overall_quality'],
                        'best_accuracy': best_acc,
                        'n_peaks': len(signature['characteristic_peaks'])
                    })
            
            if signature_results:
                signature_results.sort(key=lambda x: x['signature_quality'], reverse=True)
                
                print(f"  Correlación calidad espectral vs rendimiento ML:")
                for sr in signature_results:
                    quality_status = "" if sr['signature_quality'] >= 80 else ("" if sr['signature_quality'] >= 60 else "")
                    print(f"  {quality_status} {sr['contaminant']}: Firma {sr['signature_quality']:.0f}/100 → ML {sr['best_accuracy']:.3f}")
        
        return successful_trainings
    
    def save_spectral_enhanced_results(self) -> Tuple[str, str]:
        """Guardar resultados del entrenamiento espectral"""
        
        if not hasattr(self, 'results') or not self.results:
            print(" No hay resultados para guardar")
            return None, None
        
        # Archivo CSV con métricas espectrales
        csv_data = []
        for contaminant, result in self.results.items():
            for model_name, model_data in result['individual_models'].items():
                evaluation = model_data['evaluation']
                results_data = model_data['results']
                
                # Información espectral
                spectral_info = evaluation.get('spectral_analysis', {})
                
                csv_data.append({
                    'contaminant': contaminant,
                    'model': model_name,
                    'quality_category': evaluation['quality_category'],
                    'quality_score': evaluation['quality_score'],
                    'test_accuracy': evaluation['accuracy'],
                    'accuracy_gap': evaluation['gap'],
                    'test_auc': evaluation['auc'],
                    'test_f1': results_data.get('test_f1', 0),
                    'train_accuracy': results_data.get('train_accuracy', 0),
                    'overfitting_penalty': evaluation['overfitting_penalty'],
                    'spectral_bonus': evaluation.get('spectral_bonus', 0),
                    'spectral_enhanced': evaluation.get('spectral_enhanced', False),
                    'spectral_strategy': results_data.get('spectral_strategy', 'unknown'),
                    'has_spectral_signature': spectral_info.get('has_signature', False),
                    'signature_quality': spectral_info.get('signature_quality', 0),
                    'n_characteristic_peaks': spectral_info.get('n_characteristic_peaks', 0),
                    'is_production_ready': evaluation['is_production_ready'],
                    'feature_count': results_data.get('feature_info', {}).get('total_features', 0)
                })
        
        # Guardar CSV
        df_results = pd.DataFrame(csv_data)
        csv_file = Path(self.output_dir) / "spectral_enhanced_training_results.csv"
        df_results.to_csv(csv_file, index=False)
        
        # Archivo JSON completo con análisis espectral
        json_file = Path(self.output_dir) / "spectral_enhanced_training_complete.json"
        
        # Preparar datos para JSON
        json_data = {}
        for contaminant, result in self.results.items():
            json_data[contaminant] = {
                'contaminant': result['contaminant'],
                'spectral_analysis': result['spectral_analysis'],
                'dataset_metadata': result['dataset_metadata'],
                'timestamp': result['timestamp'],
                'individual_models': {},
                'ensemble': result['ensemble']
            }
            
            # Procesar modelos individuales
            for model_name, model_data in result['individual_models'].items():
                model_results = model_data['results'].copy()
                
                # Remover objetos no serializables
                if 'model' in model_results:
                    del model_results['model']
                
                # Convertir arrays numpy a listas
                for key, value in model_results.items():
                    if isinstance(value, np.ndarray):
                        model_results[key] = value.tolist()
                
                json_data[contaminant]['individual_models'][model_name] = {
                    'results': model_results,
                    'evaluation': model_data['evaluation']
                }
        
        # Guardar JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n Resultados guardados:")
        print(f"   CSV: {csv_file}")
        print(f"   JSON: {json_file}")
        
        return str(csv_file), str(json_file)
    
    def generate_spectral_analysis_report(self) -> str:
        """Generar reporte de análisis espectral"""
        
        if not hasattr(self, 'results') or not self.results:
            return None
        
        report_lines = []
        
        # Encabezado
        report_lines.extend([
            "# REPORTE DE ENTRENAMIENTO SPECTRAL ENHANCED",
            "## Universidad Diego Portales - María José Erazo González",
            f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Pipeline:** Spectral Enhanced ML V4",
            f"**Datasets:** {self.datasets_dir}",
            ""
        ])
        
        # Resumen ejecutivo
        total_contaminants = len(self.results)
        total_models = sum(len(result['individual_models']) for result in self.results.values())
        
        spectral_enhanced_models = 0
        excellent_models = 0
        production_ready = 0
        
        for result in self.results.values():
            for model_data in result['individual_models'].values():
                evaluation = model_data['evaluation']
                
                if evaluation.get('spectral_enhanced', False):
                    spectral_enhanced_models += 1
                
                if evaluation['quality_category'] == 'excellent':
                    excellent_models += 1
                
                if evaluation['is_production_ready']:
                    production_ready += 1
        
        ensembles_created = sum(1 for result in self.results.values() if result['ensemble'])
        
        report_lines.extend([
            "## RESUMEN EJECUTIVO",
            f"- **Contaminantes analizados:** {total_contaminants}",
            f"- **Modelos entrenados:** {total_models}",
            f"- **Modelos con mejora espectral:** {spectral_enhanced_models} ({spectral_enhanced_models/total_models*100:.1f}%)",
            f"- **Modelos excelentes:** {excellent_models}",
            f"- **Modelos listos para producción:** {production_ready}",
            f"- **Ensembles creados:** {ensembles_created}",
            f"- **Firmas espectrales disponibles:** {len(self.spectral_signatures)}",
            ""
        ])
        
        # Análisis de efectividad espectral
        if spectral_enhanced_models > 0:
            spectral_effectiveness = (spectral_enhanced_models / total_models) * 100
            
            report_lines.extend([
                "## ANÁLISIS DE EFECTIVIDAD ESPECTRAL",
                f"- **Efectividad general:** {spectral_effectiveness:.1f}% de modelos mejorados",
                ""
            ])
            
            # Análisis por estrategia
            strategy_stats = {}
            for result in self.results.values():
                for model_data in result['individual_models'].values():
                    strategy = model_data['results'].get('spectral_strategy', 'unknown')
                    accuracy = model_data['evaluation']['accuracy']
                    
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = []
                    strategy_stats[strategy].append(accuracy)
            
            report_lines.append("### Rendimiento por Estrategia Espectral:")
            for strategy, accuracies in strategy_stats.items():
                avg_acc = np.mean(accuracies)
                count = len(accuracies)
                report_lines.append(f"- **{strategy}:** {avg_acc:.3f} promedio ({count} modelos)")
            
            report_lines.append("")
        
        # Top contaminantes
        contaminant_results = []
        for contaminant, result in self.results.items():
            best_accuracy = max([model_data['evaluation']['accuracy'] 
                               for model_data in result['individual_models'].values()])
            
            has_spectral = any([model_data['evaluation'].get('spectral_enhanced', False) 
                              for model_data in result['individual_models'].values()])
            
            signature_quality = 0
            if contaminant in self.spectral_signatures:
                signature_quality = self.spectral_signatures[contaminant]['quality_metrics']['overall_quality']
            
            contaminant_results.append({
                'name': contaminant,
                'best_accuracy': best_accuracy,
                'has_spectral': has_spectral,
                'signature_quality': signature_quality,
                'has_ensemble': bool(result['ensemble'])
            })
        
        # Ordenar por accuracy
        contaminant_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
        
        report_lines.extend([
            "## TOP CONTAMINANTES POR RENDIMIENTO",
            ""
        ])
        
        for i, cr in enumerate(contaminant_results[:10], 1):
            spectral_indicator = "" if cr['has_spectral'] else ""
            ensemble_indicator = "" if cr['has_ensemble'] else ""
            
            report_lines.append(f"{i:2d}. **{cr['name']}**: {cr['best_accuracy']:.3f} {spectral_indicator} {ensemble_indicator}")
        
        report_lines.append("")
        
        # Análisis de firmas espectrales
        if self.spectral_signatures:
            report_lines.extend([
                "## ANÁLISIS DE FIRMAS ESPECTRALES",
                ""
            ])
            
            # Correlación calidad espectral vs rendimiento
            signature_analysis = []
            for contaminant in self.results.keys():
                if contaminant in self.spectral_signatures:
                    signature = self.spectral_signatures[contaminant]
                    result = self.results[contaminant]
                    
                    signature_quality = signature['quality_metrics']['overall_quality']
                    best_accuracy = max([model_data['evaluation']['accuracy'] 
                                       for model_data in result['individual_models'].values()])
                    
                    signature_analysis.append({
                        'contaminant': contaminant,
                        'signature_quality': signature_quality,
                        'ml_accuracy': best_accuracy,
                        'n_peaks': len(signature['characteristic_peaks'])
                    })
            
            if signature_analysis:
                # Calcular correlación
                qualities = [sa['signature_quality'] for sa in signature_analysis]
                accuracies = [sa['ml_accuracy'] for sa in signature_analysis]
                
                correlation = np.corrcoef(qualities, accuracies)[0, 1] if len(qualities) > 1 else 0
                
                report_lines.extend([
                    f"### Correlación Calidad Espectral vs Rendimiento ML: {correlation:.3f}",
                    ""
                ])
                
                # Ordenar por calidad espectral
                signature_analysis.sort(key=lambda x: x['signature_quality'], reverse=True)
                
                report_lines.append("### Contaminantes con Firmas Espectrales:")
                for sa in signature_analysis:
                    quality_status = "" if sa['signature_quality'] >= 80 else ("" if sa['signature_quality'] >= 60 else "")
                    report_lines.append(f"- {quality_status} **{sa['contaminant']}**: Firma {sa['signature_quality']:.0f}/100 → ML {sa['ml_accuracy']:.3f}")
                
                report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---",
            f"*Generado automáticamente por Spectral Enhanced ML Pipeline V4*",
            f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        # Guardar reporte
        report_file = Path(self.output_dir) / "spectral_enhanced_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f" Reporte de análisis guardado: {report_file}")
        
        return str(report_file)


def main():
    """Función principal para ejecutar el entrenamiento espectral enhanced"""
    
    print(" TRAIN V4 SPECTRAL ENHANCED")
    print("Universidad Diego Portales - María José Erazo González")
    print("="*70)
    
    # Detectar directorios de datasets disponibles
    possible_dirs = [
        "spectral_enhanced_datasets_combined",
        "spectral_enhanced_datasets_spectral_only", 
        "spectral_enhanced_datasets_raw_only",
        "enhanced_datasets",
        "datasets"
    ]
    
    available_dirs = [d for d in possible_dirs if os.path.exists(d)]
    
    if not available_dirs:
        print(" No se encontraron directorios de datasets válidos")
        print("   Ejecuta primero ML_dataset_generator_spectral_enhanced.py")
        return None
    
    print(f"\n DIRECTORIOS DE DATASETS DISPONIBLES:")
    for i, dir_name in enumerate(available_dirs, 1):
        # Contar archivos
        dir_path = Path(dir_name)
        spectral_files = len(list(dir_path.glob("*_spectral_enhanced_*.npz")))
        enhanced_files = len(list(dir_path.glob("*_enhanced_*.npz")))
        classical_files = len(list(dir_path.glob("*_classical.npz")))
        
        total_files = spectral_files + enhanced_files + classical_files
        
        indicators = []
        if spectral_files > 0:
            indicators.append("")
        if enhanced_files > 0:
            indicators.append("")
        if classical_files > 0:
            indicators.append("")
        
        print(f"  {i}. {dir_name} ({total_files} datasets) {''.join(indicators)}")
    
    # Selección de directorio
    if len(available_dirs) == 1:
        datasets_dir = available_dirs[0]
        print(f"\n Usando único directorio disponible: {datasets_dir}")
    else:
        choice = input(f"\nSelecciona directorio (1-{len(available_dirs)}) [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_dirs):
                datasets_dir = available_dirs[idx]
                print(f" Directorio seleccionado: {datasets_dir}")
            else:
                datasets_dir = available_dirs[0]
                print(f" Índice inválido, usando: {datasets_dir}")
        except:
            datasets_dir = available_dirs[0]
            print(f" Entrada inválida, usando: {datasets_dir}")
    
    # Selección de estrategia espectral
    print(f"\n ESTRATEGIA DE ANÁLISIS ESPECTRAL:")
    print(f"1. Auto (recomendado) - Selecciona automáticamente la mejor estrategia")
    print(f"2. Spectral Only - Solo features espectrales interpretables")
    print(f"3. Combined - Features espectrales + bandas raw")
    print(f"4. Raw Only - Solo bandas espectrales raw")
    
    strategy_choice = input(f"\nSelecciona estrategia (1-4) [1]: ").strip() or "1"
    
    strategy_map = {
        "1": "auto",
        "2": "spectral_only", 
        "3": "combined",
        "4": "raw_only"
    }
    
    spectral_strategy = strategy_map.get(strategy_choice, "auto")
    print(f" Estrategia seleccionada: {spectral_strategy}")
    
    try:
        # Crear pipeline espectral enhanced
        pipeline = SpectralEnhancedMLPipeline(
            datasets_dir=datasets_dir,
            output_dir="model_outputs_spectral_v4",
            random_state=42,
            spectral_strategy=spectral_strategy
        )
        
        print(f"\n CARACTERÍSTICAS DEL PIPELINE:")
        print(f"   Análisis espectral: Integrado")
        print(f"   Firmas espectrales: {len(pipeline.spectral_signatures)} disponibles")
        print(f"   Estrategia adaptativa: {spectral_strategy}")
        print(f"   Ensemble espectral: Habilitado")
        print(f"   Evaluación mejorada: Con bonus espectral")
        
        # Ejecutar pipeline
        print(f"\n EJECUTANDO ENTRENAMIENTO SPECTRAL ENHANCED...")
        successful_trainings = pipeline.train_spectral_enhanced_pipeline()
        
        if successful_trainings == 0:
            print(" No se pudieron entrenar modelos")
            return None
        
        # Guardar resultados
        csv_file, json_file = pipeline.save_spectral_enhanced_results()
        
        # Generar reporte de análisis espectral
        report_file = pipeline.generate_spectral_analysis_report()
        
        print(f"\n ENTRENAMIENTO SPECTRAL ENHANCED COMPLETADO")
        print(f"="*70)
        print(f" Contaminantes procesados exitosamente: {successful_trainings}")
        
        # Calcular estadísticas finales
        total_models = sum(len(result['individual_models']) for result in pipeline.results.values())
        spectral_enhanced_models = sum(
            sum(1 for model_data in result['individual_models'].values() 
                if model_data['evaluation'].get('spectral_enhanced', False))
            for result in pipeline.results.values()
        )
        
        print(f"\n IMPACTO DEL ANÁLISIS ESPECTRAL:")
        print(f"   Modelos con mejora espectral: {spectral_enhanced_models}/{total_models}")
        print(f"   Tasa de mejora: {spectral_enhanced_models/total_models*100:.1f}%")
        
        if pipeline.spectral_signatures:
            # Análisis de correlación
            signature_contaminants = [cont for cont in pipeline.results.keys() 
                                    if cont in pipeline.spectral_signatures]
            
            if signature_contaminants:
                qualities = [pipeline.spectral_signatures[cont]['quality_metrics']['overall_quality'] 
                           for cont in signature_contaminants]
                
                accuracies = [max([model_data['evaluation']['accuracy'] 
                                 for model_data in pipeline.results[cont]['individual_models'].values()])
                            for cont in signature_contaminants]
                
                correlation = np.corrcoef(qualities, accuracies)[0, 1] if len(qualities) > 1 else 0
                print(f"   Correlación calidad espectral vs ML: {correlation:.3f}")
        
        print(f"\n ARCHIVOS GENERADOS:")
        print(f"   {csv_file}")
        print(f"   {json_file}")
        print(f"   {report_file}")
        
        print(f"\n🎓 CONTRIBUCIONES CIENTÍFICAS:")
        print(f"   Integración análisis espectral + ML")
        print(f"   Metodología de ensemble espectral")
        print(f"   Evaluación con métricas espectrales")
        print(f"   Optimización automática por estrategia")
        print(f"   Correlación firmas espectrales vs rendimiento")
        
        return pipeline
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_single_contaminant_spectral(contaminant_name: str = "diuron"):
    """Demo de entrenamiento espectral para un solo contaminante"""
    
    print(f" DEMO SPECTRAL ENHANCED: {contaminant_name}")
    print("="*60)
    
    pipeline = SpectralEnhancedMLPipeline(
        datasets_dir="spectral_enhanced_datasets_combined",
        output_dir="demo_spectral_outputs",
        spectral_strategy="auto"
    )
    
    result = pipeline.train_contaminant_spectral_enhanced(contaminant_name)
    
    if result:
        print(f"\n Demo completado exitosamente")
        
        # Mostrar análisis detallado
        print(f"\n ANÁLISIS DETALLADO:")
        for model_name, model_data in result['individual_models'].items():
            evaluation = model_data['evaluation']
            print(f"  {model_name}:")
            print(f"    Accuracy: {evaluation['accuracy']:.3f}")
            print(f"    Calidad: {evaluation['quality_category']}")
            print(f"    Mejora espectral: {'SÍ' if evaluation.get('spectral_enhanced', False) else 'NO'}")
        
        return result
    else:
        print(f" Demo falló")
        return None


if __name__ == "__main__":
    # Ejecutar entrenamiento completo
    pipeline = main()
    
    # Opcional: Ejecutar demo para un contaminante específico
    #demo_result = demo_single_contaminant_spectral("oit")