"""
Módulo de preprocesamiento de datos espectrales hiperespectrales.

Este módulo contiene todas las funciones necesarias para procesar datos espectrales
desde su formato raw hasta datos listos para machine learning.
"""

import numpy as np
import pandas as pd
from spectral import envi, imshow
import os
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA


class SpectralPreprocessor:
    """Clase principal para preprocesamiento de datos espectrales."""
    
    def __init__(self, config=None):
        """
        Inicializa el preprocesador con configuración específica.
        
        Args:
            config (dict): Configuración de preprocesamiento
        """
        self.config = config or {}
        self.normalization_method = self.config.get('normalization', 'snv')
        self.smoothing_enabled = self.config.get('smoothing', True)
        self.baseline_correction = self.config.get('baseline_correction', True)
        self.noise_reduction = self.config.get('noise_reduction', True)
    
    def create_mask_80_percentile_reflection(self, img: np.ndarray) -> np.ndarray:
        """
        Crea máscara de reflexión basada en el percentil 80 de valores de píxeles.
        
        Args:
            img (np.ndarray): Imagen 3D
        
        Returns:
            np.ndarray: Máscara 3D
        """
        assert len(img.shape) == 3, 'La entrada debe ser un array 3D de numpy'

        # Calcular intensidad media a través de la dimensión espectral
        mean_intensity = img.mean(axis=2)

        # Calcular percentil 80 de la intensidad media
        perc80 = np.percentile(mean_intensity, 80)

        # Crear máscara para píxeles por debajo del percentil 80
        mask = mean_intensity < perc80

        # Repetir la máscara a lo largo de la dimensión espectral
        mask_3d = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)

        return mask_3d.astype(int)

    def create_mask_20_percentile_reflection(self, img: np.ndarray) -> np.ndarray:
        """
        Crea máscara de reflexión basada en el percentil 20 de valores de píxeles.
        
        Args:
            img (np.ndarray): Imagen 3D
        
        Returns:
            np.ndarray: Máscara 3D
        """
        assert len(img.shape) == 3, 'La entrada debe ser un array 3D de numpy'

        # Calcular intensidad media a través de la dimensión espectral
        mean_intensity = img.mean(axis=2)

        # Calcular percentil 20 de la intensidad media
        perc20 = np.percentile(mean_intensity, 20)

        # Crear máscara para píxeles por encima del percentil 20
        mask = mean_intensity > perc20

        # Repetir la máscara a lo largo de la dimensión espectral
        mask_3d = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)

        return mask_3d.astype(int)

    def apply_combined_mask(self, datacube: np.ndarray) -> np.ndarray:
        """
        Aplica máscaras de reflexión combinadas a un datacube.
        
        Args:
            datacube (np.ndarray): Datacube 3D
        
        Returns:
            np.ndarray: Datacube con máscaras aplicadas
        """
        assert len(datacube.shape) == 3, 'La entrada debe ser un array 3D de numpy'

        # Calcular cada máscara
        mask_80 = self.create_mask_80_percentile_reflection(datacube)
        mask_20 = self.create_mask_20_percentile_reflection(datacube)

        # Combinar las máscaras
        combined_mask = mask_80 * mask_20
        
        # Verificar si la máscara tiene al menos algunos píxeles
        non_zero_pixels = np.sum(combined_mask[:,:,0])
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        percentage = (non_zero_pixels / total_pixels) * 100
        
        if non_zero_pixels == 0:
            print(f"⚠️ La máscara combinada elimina todos los píxeles. Usando máscara menos restrictiva.")
            combined_mask = mask_20
            
            non_zero_pixels = np.sum(combined_mask[:,:,0])
            percentage = (non_zero_pixels / total_pixels) * 100
            
            if non_zero_pixels == 0:
                print(f"⚠️ Todas las máscaras son demasiado restrictivas. Usando datacube sin máscara.")
                return datacube
        
        print(f"ℹ️ La máscara retiene {percentage:.2f}% de los píxeles ({non_zero_pixels}/{total_pixels})")
        
        return combined_mask * datacube

    def convert_to_reflectance(self, datacube: np.ndarray, whiteref: np.ndarray, darkref: np.ndarray) -> np.ndarray:
        """
        Convierte datos raw de datacube a reflectancia usando referencias blanca y oscura.
        
        Args:
            datacube (np.ndarray): Datacube hiperespectral raw
            whiteref (np.ndarray): Datacube de referencia blanca
            darkref (np.ndarray): Datacube de referencia oscura
        
        Returns:
            np.ndarray: Datacube convertido a reflectancia
        """
        # Verificar si las dimensiones son compatibles
        if datacube.shape != whiteref.shape or datacube.shape != darkref.shape:
            print(f"Ajustando referencias - Shapes: datacube={datacube.shape}, white={whiteref.shape}, dark={darkref.shape}")
            
            white_spectrum = np.mean(whiteref, axis=(0, 1))
            dark_spectrum = np.mean(darkref, axis=(0, 1))
            
            min_bands = min(datacube.shape[2], len(white_spectrum), len(dark_spectrum))
            
            white_adjusted = np.zeros_like(datacube)
            dark_adjusted = np.zeros_like(datacube)
            
            for i in range(min_bands):
                white_adjusted[:, :, i] = white_spectrum[i]
                dark_adjusted[:, :, i] = dark_spectrum[i]
            
            whiteref = white_adjusted
            darkref = dark_adjusted
        
        # Evitar división por cero
        epsilon = 1e-10
        denominator = whiteref - darkref
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        
        # Calcular reflectancia
        reflectance = (datacube - darkref) / denominator
        
        # Manejar valores infinitos o NaN
        reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Limitar al rango [0, 1]
        reflectance = np.clip(reflectance, 0, 1)
        
        return reflectance

    def crop_datacube(self, datacube: np.ndarray) -> np.ndarray:
        """
        Recorta el datacube a una región de interés específica.
        
        Args:
            datacube (np.ndarray): Datacube 3D
        
        Returns:
            np.ndarray: Datacube recortado
        """
        if datacube.shape[0] < 45 or datacube.shape[1] < 860:
            print(f"⚠️ Datacube demasiado pequeño para recorte estándar: {datacube.shape}")
            return datacube
        
        return datacube[0:45, 110:860, :]

    def savitzky_golay_filter(self, spectrum, window_size=15, poly_order=3):
        """
        Aplica filtro Savitzky-Golay para suavizar espectros.
        
        Args:
            spectrum: Array de datos espectrales
            window_size: Tamaño de la ventana (debe ser impar)
            poly_order: Orden del polinomio
            
        Returns:
            Array con el espectro suavizado
        """
        if window_size % 2 == 0:
            window_size += 1
            
        return signal.savgol_filter(spectrum, window_size, poly_order)

    def baseline_als(self, spectrum, lam=100, p=0.01, n_iter=10):
        """
        Corrección de línea base usando Asymmetric Least Squares.
        
        Args:
            spectrum: Array de datos espectrales
            lam: Factor de suavizado (mayor = más suave)
            p: Factor de asimetría (0.001-0.1)
            n_iter: Número de iteraciones
            
        Returns:
            Espectro corregido para línea base
        """
        L = len(spectrum)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        
        for i in range(n_iter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + D
            z = spsolve(Z, w*spectrum)
            w = p * (spectrum > z) + (1-p) * (spectrum <= z)
            
        return spectrum - z

    def pca_noise_reduction(self, spectra_matrix, n_components=0.95):
        """
        Reducción de ruido usando PCA.
        
        Args:
            spectra_matrix: Matriz de espectros (filas=espectros, columnas=longitudes de onda)
            n_components: Número de componentes a retener o fracción de varianza
            
        Returns:
            Matriz de espectros con ruido reducido
        """
        pca = PCA(n_components=n_components)
        spectra_reduced = pca.fit_transform(spectra_matrix)
        spectra_reconstructed = pca.inverse_transform(spectra_reduced)
        
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        print(f"PCA utilizó {pca.n_components_} componentes, explicando {explained_variance:.2f}% de la varianza")
        
        return spectra_reconstructed

    def normalize_spectra(self, spectra_matrix, method='snv'):
        """
        Normaliza espectros usando diferentes métodos.
        
        Args:
            spectra_matrix: Matriz de espectros
            method: Método de normalización ('minmax', 'area', 'snv', 'msc')
            
        Returns:
            Matriz de espectros normalizados
        """
        normalized = np.zeros_like(spectra_matrix)
        
        if method == 'minmax':
            for i in range(spectra_matrix.shape[0]):
                spectrum = spectra_matrix[i, :]
                min_val = np.min(spectrum)
                max_val = np.max(spectrum)
                if max_val > min_val:
                    normalized[i, :] = (spectrum - min_val) / (max_val - min_val)
                else:
                    normalized[i, :] = spectrum
                    
        elif method == 'area':
            for i in range(spectra_matrix.shape[0]):
                spectrum = spectra_matrix[i, :]
                area = np.trapz(spectrum)
                if area != 0:
                    normalized[i, :] = spectrum / area
                else:
                    normalized[i, :] = spectrum
                    
        elif method == 'snv':
            for i in range(spectra_matrix.shape[0]):
                spectrum = spectra_matrix[i, :]
                mean_val = np.mean(spectrum)
                std_val = np.std(spectrum)
                if std_val > 0:
                    normalized[i, :] = (spectrum - mean_val) / std_val
                else:
                    normalized[i, :] = spectrum
                    
        elif method == 'msc':
            mean_spectrum = np.mean(spectra_matrix, axis=0)
            
            for i in range(spectra_matrix.shape[0]):
                spectrum = spectra_matrix[i, :]
                coeffs = np.polyfit(mean_spectrum, spectrum, 1)
                normalized[i, :] = (spectrum - coeffs[1]) / coeffs[0]
                
        else:
            raise ValueError(f"Método de normalización desconocido: {method}")
            
        return normalized

    def process_hyperspectral_datacube(self, datacube_path, white_ref_path=None, dark_ref_path=None):
        """
        Procesa un datacube hiperespectral completo.
        
        Args:
            datacube_path: Ruta al datacube
            white_ref_path: Ruta a la referencia blanca
            dark_ref_path: Ruta a la referencia oscura
            
        Returns:
            np.ndarray: Datacube procesado
        """
        # Cargar datacube
        datacube = self.open_datacube(datacube_path)
        
        # Aplicar máscaras
        datacube = self.apply_combined_mask(datacube)
        
        # Convertir a reflectancia si hay referencias
        if white_ref_path and dark_ref_path:
            white_ref = self.open_datacube(white_ref_path)
            dark_ref = self.open_datacube(dark_ref_path)
            datacube = self.convert_to_reflectance(datacube, white_ref, dark_ref)
        
        # Recortar datacube
        datacube = self.crop_datacube(datacube)
        
        return datacube

    def open_datacube(self, folder: str, name: str = "raw", show: bool = False) -> np.ndarray:
        """
        Abre un datacube hiperespectral desde una carpeta específica.
        
        Args:
            folder (str): Carpeta que contiene los archivos del datacube
            name (str): Nombre base de los archivos del datacube
            show (bool): Si mostrar el datacube
        
        Returns:
            np.ndarray: Datacube cargado
        """
        hdr_file = os.path.join(folder, f"{name}.hdr")
        bin_file = os.path.join(folder, f"{name}.bin")
        
        if not os.path.exists(hdr_file):
            raise FileNotFoundError(f"Archivo .hdr no encontrado: {hdr_file}")
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Archivo .bin no encontrado: {bin_file}")
        
        try:
            datacube = envi.open(hdr_file, bin_file).load()
        except Exception as e:
            print(f"Primer método falló, intentando método alternativo: {str(e)}")
            datacube = envi.open(hdr_file, image=bin_file).load()
        
        if datacube.shape[2] > 200:
            datacube = datacube[:, :, :200]
        
        if show:
            imshow(datacube)
        
        return datacube

    def calculate_mean_std_image(self, img: np.ndarray) -> tuple:
        """
        Calcula el espectro de reflectancia medio y la desviación estándar de una imagen.
        
        Args:
            img (np.ndarray): Array 3D representando la imagen
        
        Returns:
            tuple: (espectro_medio, desviacion_estandar)
        """
        assert len(img.shape) == 3, 'La entrada debe ser un array 3D de numpy'

        sp_list = [img[i][j] for i in range(img.shape[0]) for j in range(img.shape[1]) if np.any(img[i][j] != 0)]

        if not sp_list:
            print("⚠️ No se encontraron píxeles no-cero después de aplicar la máscara")
            mean_spectrum = np.zeros(img.shape[2])
            standard_deviation = np.zeros(img.shape[2])
            return mean_spectrum, standard_deviation
        
        sp_list = np.array(sp_list)
        mean_spectrum = sp_list.mean(axis=0)
        standard_deviation = sp_list.std(axis=0)

        return mean_spectrum, standard_deviation

    def preprocess_complete_pipeline(self, spectra_matrix):
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            spectra_matrix: Matriz de espectros a procesar
            
        Returns:
            Matriz de espectros procesados
        """
        print("Iniciando pipeline completo de preprocesamiento...")
        
        # 1. Normalización
        if self.normalization_method:
            print(f"Aplicando normalización: {self.normalization_method}")
            spectra_matrix = self.normalize_spectra(spectra_matrix, method=self.normalization_method)
        
        # 2. Suavizado
        if self.smoothing_enabled:
            print("Aplicando filtro Savitzky-Golay...")
            smoothed_spectra = np.zeros_like(spectra_matrix)
            for i in range(spectra_matrix.shape[0]):
                smoothed_spectra[i, :] = self.savitzky_golay_filter(spectra_matrix[i, :])
            spectra_matrix = smoothed_spectra
        
        # 3. Corrección de línea base
        if self.baseline_correction:
            print("Aplicando corrección de línea base...")
            baseline_corrected = np.zeros_like(spectra_matrix)
            for i in range(spectra_matrix.shape[0]):
                baseline_corrected[i, :] = self.baseline_als(spectra_matrix[i, :])
            spectra_matrix = baseline_corrected
        
        # 4. Reducción de ruido
        if self.noise_reduction:
            print("Aplicando reducción de ruido con PCA...")
            spectra_matrix = self.pca_noise_reduction(spectra_matrix)
        
        print("Pipeline de preprocesamiento completado.")
        return spectra_matrix