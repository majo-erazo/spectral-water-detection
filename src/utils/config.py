"""
Módulo de configuración central del sistema.

Este módulo maneja todas las configuraciones, parámetros y constantes
utilizadas en el sistema de detección de contaminantes.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigManager:
    """Gestor central de configuraciones del sistema."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Inicializa el gestor de configuraciones.
        
        Args:
            config_dir: Directorio de configuraciones (opcional)
        """
        if config_dir is None:
            # Buscar directorio de configuración desde la raíz del proyecto
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Carga todas las configuraciones desde archivos YAML."""
        config_files = {
            'models': 'model_configs.yaml',
            'data': 'data_configs.yaml',
            'experiments': 'experiment_configs.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                self._configs[config_name] = self._load_yaml(config_path)
            else:
                print(f"Advertencia: Archivo de configuración no encontrado: {config_path}")
                self._configs[config_name] = {}
    
    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Carga un archivo YAML."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error cargando {filepath}: {e}")
            return {}
    
    def get_model_config(self, algorithm: str) -> Dict[str, Any]:
        """
        Obtiene configuración para un algoritmo específico.
        
        Args:
            algorithm: Nombre del algoritmo ('svm', 'xgboost', 'lstm', 'ensemble')
            
        Returns:
            Diccionario con configuración del algoritmo
        """
        models_config = self._configs.get('models', {})
        algorithm_config = models_config.get('algorithms', {}).get(algorithm, {})
        
        # Combinar con configuración global
        global_config = models_config.get('global', {})
        
        # Crear configuración completa
        config = {**global_config, **algorithm_config}
        return config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Obtiene configuración de datos."""
        return self._configs.get('data', {})
    
    def get_experiment_config(self, experiment_name: str = 'default') -> Dict[str, Any]:
        """
        Obtiene configuración de experimento.
        
        Args:
            experiment_name: Nombre del experimento
            
        Returns:
            Configuración del experimento
        """
        experiments_config = self._configs.get('experiments', {})
        return experiments_config.get(experiment_name, {})
    
    def get_contaminants_list(self) -> List[str]:
        """Obtiene lista de contaminantes soportados."""
        data_config = self.get_data_config()
        return data_config.get('contaminants', DEFAULT_CONTAMINANTS)
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de preprocesamiento."""
        data_config = self.get_data_config()
        return data_config.get('preprocessing', DEFAULT_PREPROCESSING_CONFIG)
    
    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """
        Actualiza configuración en memoria.
        
        Args:
            config_type: Tipo de configuración ('models', 'data', 'experiments')
            updates: Actualizaciones a aplicar
        """
        if config_type in self._configs:
            self._configs[config_type].update(updates)
        else:
            self._configs[config_type] = updates


# Configuraciones por defecto
DEFAULT_CONTAMINANTS = [
    "6ppd-quinone", "4-&5-methylbenzotriazole", "13-diphenylguanidine", 
    "24-d", "acesulfame", "benzotriazole", "caffeine", "candesartan",
    "carbendazim", "citalopram", "cyclamate", "deet", "diclofenac", 
    "diuron", "hmmm", "hydrochlorthiazide", "mcpa", "mecoprop", 
    "oit", "triclosan"
]

DEFAULT_PREPROCESSING_CONFIG = {
    'normalization': 'snv',
    'smoothing': True,
    'baseline_correction': True,
    'noise_reduction': True,
    'window_size': 15,
    'poly_order': 3,
    'pca_components': 0.95
}

DEFAULT_MODEL_CONFIGS = {
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42,
        'optimize_hyperparameters': True
    },
    'xgboost': {
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 100,
        'random_state': 42,
        'optimize_hyperparameters': True
    },
    'lstm': {
        'lstm_units': 32,
        'dropout_rate': 0.3,
        'recurrent_dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 100,
        'patience': 10,
        'sequence_length': 10
    },
    'ensemble': {
        'voting_method': 'soft',
        'models': ['svm', 'xgboost'],
        'optimize_weights': True
    }
}

# Rutas del proyecto
def get_project_root() -> Path:
    """Obtiene la ruta raíz del proyecto."""
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    """Obtiene directorio de datos."""
    return get_project_root() / "data"

def get_models_dir() -> Path:
    """Obtiene directorio de modelos."""
    return get_project_root() / "models"

def get_results_dir() -> Path:
    """Obtiene directorio de resultados."""
    return get_project_root() / "results"

def get_config_dir() -> Path:
    """Obtiene directorio de configuración."""
    return get_project_root() / "config"

# Paths específicos
class Paths:
    """Clase con rutas estáticas del proyecto."""
    
    PROJECT_ROOT = get_project_root()
    DATA_DIR = get_data_dir()
    MODELS_DIR = get_models_dir()
    RESULTS_DIR = get_results_dir()
    CONFIG_DIR = get_config_dir()
    
    # Subdirectorios de datos
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    
    # Subdirectorios de datos procesados
    FEATURES_DIR = PROCESSED_DATA_DIR / "features"
    SIGNATURES_DIR = PROCESSED_DATA_DIR / "signatures"
    DATASETS_DIR = PROCESSED_DATA_DIR / "datasets"
    
    # Subdirectorios de modelos
    SVM_MODELS_DIR = MODELS_DIR / "svm"
    XGBOOST_MODELS_DIR = MODELS_DIR / "xgboost"
    LSTM_MODELS_DIR = MODELS_DIR / "lstm"
    ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"
    METADATA_DIR = MODELS_DIR / "metadata"
    
    # Subdirectorios de resultados
    EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
    VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
    REPORTS_DIR = RESULTS_DIR / "reports"
    LOGS_DIR = RESULTS_DIR / "logs"
    
    @classmethod
    def create_directories(cls):
        """Crea todos los directorios necesarios."""
        dirs_to_create = [
            cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR,
            cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR,
            cls.FEATURES_DIR, cls.SIGNATURES_DIR, cls.DATASETS_DIR,
            cls.SVM_MODELS_DIR, cls.XGBOOST_MODELS_DIR, cls.LSTM_MODELS_DIR,
            cls.ENSEMBLE_MODELS_DIR, cls.METADATA_DIR,
            cls.EXPERIMENTS_DIR, cls.VISUALIZATIONS_DIR, cls.REPORTS_DIR, cls.LOGS_DIR
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Directorios del proyecto creados en: {cls.PROJECT_ROOT}")


# Variables de entorno y configuración del sistema
class SystemConfig:
    """Configuración del sistema y variables de entorno."""
    
    # Configuración de logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configuración de paralelización
    N_JOBS = int(os.getenv('N_JOBS', -1))  # -1 para usar todos los cores
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    
    # Configuración de memoria
    MAX_MEMORY_USAGE = os.getenv('MAX_MEMORY_USAGE', '8GB')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))
    
    # Configuración de GPU (si está disponible)
    USE_GPU = os.getenv('USE_GPU', 'auto').lower()  # 'true', 'false', 'auto'
    GPU_MEMORY_LIMIT = os.getenv('GPU_MEMORY_LIMIT', '4GB')
    
    # Configuración de datos
    DATA_BACKEND = os.getenv('DATA_BACKEND', 'pandas')  # 'pandas', 'dask', 'polars'
    CACHE_DATA = os.getenv('CACHE_DATA', 'true').lower() == 'true'
    
    # Configuración de modelos
    MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', 5))
    AUTO_SAVE_MODELS = os.getenv('AUTO_SAVE_MODELS', 'true').lower() == 'true'
    
    # Configuración de visualización
    PLOT_BACKEND = os.getenv('PLOT_BACKEND', 'matplotlib')  # 'matplotlib', 'plotly'
    FIGURE_DPI = int(os.getenv('FIGURE_DPI', 300))
    
    # Configuración de APIs externas (para futuro)
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))


# Configuración de métricas de evaluación
EVALUATION_METRICS = {
    'binary_classification': [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'roc_auc', 'average_precision', 'balanced_accuracy'
    ],
    'multiclass_classification': [
        'accuracy', 'precision_weighted', 'recall_weighted', 
        'f1_weighted', 'balanced_accuracy'
    ],
    'regression': [
        'mse', 'rmse', 'mae', 'r2_score', 'explained_variance'
    ]
}

# Configuración de validación cruzada
CV_CONFIG = {
    'default_cv_folds': 5,
    'min_samples_per_fold': 5,
    'stratified': True,
    'shuffle': True,
    'random_state': 42
}

# Configuración de optimización de hiperparámetros
HYPEROPT_CONFIG = {
    'max_evals': 50,
    'timeout': 3600,  # 1 hora máximo
    'early_stopping': True,
    'early_stopping_rounds': 10,
    'n_jobs': -1
}

# Configuración de visualización
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',
    'figsize': (10, 6),
    'dpi': 300,
    'color_palette': 'viridis',
    'save_format': 'png',
    'transparent': False
}

# Configuración de exportación de datos
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'csv_separator': ',',
    'decimal_places': 4,
    'include_index': False,
    'compress': False
}


class AdaptiveConfig:
    """Configuración adaptativa basada en las características del dataset."""
    
    @staticmethod
    def get_adaptive_model_config(algorithm: str, n_samples: int, n_features: int, n_classes: int) -> Dict[str, Any]:
        """
        Obtiene configuración adaptativa para un modelo basado en características del dataset.
        
        Args:
            algorithm: Tipo de algoritmo
            n_samples: Número de muestras
            n_features: Número de características
            n_classes: Número de clases
            
        Returns:
            Configuración adaptativa
        """
        config = DEFAULT_MODEL_CONFIGS.get(algorithm, {}).copy()
        
        # Adaptaciones basadas en tamaño del dataset
        if algorithm == 'svm':
            if n_samples < 50:
                config.update({
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                })
            elif n_samples < 200:
                config.update({
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear', 'poly']
                })
        
        elif algorithm == 'xgboost':
            if n_samples < 50:
                config.update({
                    'learning_rate': 0.01,
                    'max_depth': 2,
                    'n_estimators': 50
                })
            elif n_samples < 200:
                config.update({
                    'learning_rate': 0.03,
                    'max_depth': 3,
                    'n_estimators': 100
                })
            else:
                config.update({
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'n_estimators': 200
                })
        
        elif algorithm == 'lstm':
            # Adaptar tamaño de LSTM según complejidad
            if n_features < 20:
                config['lstm_units'] = 16
            elif n_features < 50:
                config['lstm_units'] = 32
            else:
                config['lstm_units'] = 64
            
            # Adaptar batch size según número de muestras
            if n_samples < 100:
                config['batch_size'] = 8
            elif n_samples < 500:
                config['batch_size'] = 16
            else:
                config['batch_size'] = 32
        
        # Configuración específica para problemas multiclase
        if n_classes > 2:
            if algorithm == 'xgboost':
                config.update({
                    'objective': 'multi:softprob',
                    'num_class': n_classes
                })
        
        return config
    
    @staticmethod
    def get_adaptive_cv_config(n_samples: int, n_classes: int) -> Dict[str, Any]:
        """
        Obtiene configuración adaptativa para validación cruzada.
        
        Args:
            n_samples: Número de muestras
            n_classes: Número de clases
            
        Returns:
            Configuración de CV adaptativa
        """
        config = CV_CONFIG.copy()
        
        # Adaptar número de folds basado en muestras por clase
        min_samples_per_class = n_samples // n_classes
        max_folds = min_samples_per_class // 2
        
        config['cv_folds'] = max(2, min(config['default_cv_folds'], max_folds))
        
        # Deshabilitar estratificación si hay muy pocas muestras
        if min_samples_per_class < 3:
            config['stratified'] = False
        
        return config


def load_config_from_env() -> Dict[str, Any]:
    """
    Carga configuración desde variables de entorno.
    
    Returns:
        Diccionario con configuración desde variables de entorno
    """
    config = {}
    
    # Mapear variables de entorno a configuración
    env_mappings = {
        'CONTAMINANT_DATA_DIR': 'data_dir',
        'CONTAMINANT_MODELS_DIR': 'models_dir',
        'CONTAMINANT_RESULTS_DIR': 'results_dir',
        'CONTAMINANT_LOG_LEVEL': 'log_level',
        'CONTAMINANT_N_JOBS': 'n_jobs',
        'CONTAMINANT_RANDOM_STATE': 'random_state',
        'CONTAMINANT_USE_GPU': 'use_gpu'
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Intentar convertir a tipo apropiado
            if config_key in ['n_jobs', 'random_state']:
                try:
                    value = int(value)
                except ValueError:
                    pass
            elif config_key == 'use_gpu':
                value = value.lower() in ['true', '1', 'yes']
            
            config[config_key] = value
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida que la configuración sea correcta.
    
    Args:
        config: Configuración a validar
        
    Returns:
        True si la configuración es válida
    """
    required_keys = ['data_dir', 'models_dir', 'results_dir']
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Clave requerida '{key}' no encontrada en configuración")
            return False
    
    # Validar que los directorios existan o se puedan crear
    for key in ['data_dir', 'models_dir', 'results_dir']:
        if key in config:
            try:
                Path(config[key]).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: No se puede crear directorio {config[key]}: {e}")
                return False
    
    return True


# Instancia global del gestor de configuraciones
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Obtiene la instancia global del gestor de configuraciones."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_model_config(algorithm: str) -> Dict[str, Any]:
    """Función de conveniencia para obtener configuración de modelo."""
    return get_config_manager().get_model_config(algorithm)

def get_contaminants_list() -> List[str]:
    """Función de conveniencia para obtener lista de contaminantes."""
    return get_config_manager().get_contaminants_list()

def get_preprocessing_config() -> Dict[str, Any]:
    """Función de conveniencia para obtener configuración de preprocesamiento."""
    return get_config_manager().get_preprocessing_config()