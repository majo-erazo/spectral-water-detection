# dataset_anti_overfitting.py
# Versión mejorada que reduce significativamente el overfitting

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

class DatasetAntiOverfitting:
    """Generador de datasets más realistas para evitar overfitting"""
    
    def __init__(self, df_firma):
        self.df_firma = df_firma
        self.wavelengths = df_firma['wavelength'].values
        self.high_mean = df_firma['high_mean'].values
        self.low_mean = df_firma['low_mean'].values
        self.signature = df_firma['signature'].values
        
        # Calcular desviaciones estándar más realistas
        if 'high_std' in df_firma.columns:
            self.high_std = df_firma['high_std'].values
            self.low_std = df_firma['low_std'].values
        else:
            # Estimar STD más conservadora
            self.high_std = np.abs(self.high_mean) * 0.1  # 10% en lugar de 2%
            self.low_std = np.abs(self.low_mean) * 0.1
    
    def simular_variabilidad_instrumental(self, n_samples):
        """Simula variabilidad del instrumento"""
        # Deriva espectral por calibración
        deriva_calibracion = np.random.normal(0, 0.02, (n_samples, len(self.wavelengths)))
        
        # Ruido baseline variable
        baseline_drift = np.random.normal(0, 0.005, (n_samples, len(self.wavelengths)))
        
        # Variación de intensidad de lámpara
        lamp_intensity = np.random.uniform(0.95, 1.05, (n_samples, 1))
        
        return deriva_calibracion + baseline_drift, lamp_intensity
    
    def simular_condiciones_ambientales(self, n_samples):
        """Simula efectos de temperatura, humedad, etc."""
        # Efecto de temperatura (desplazamiento espectral pequeño)
        temp_effect = np.random.normal(0, 0.01, (n_samples, len(self.wavelengths)))
        
        # Efecto de humedad en absorción
        humidity_factor = np.random.uniform(0.98, 1.02, (n_samples, 1))
        
        # Presión atmosférica (efecto menor)
        pressure_effect = np.random.normal(0, 0.003, (n_samples, len(self.wavelengths)))
        
        return temp_effect + pressure_effect, humidity_factor
    
    def simular_interferencias_quimicas(self, n_samples):
        """Simula interferencias de otros compuestos"""
        # Interferencias espectrales aleatorias
        interferencias = []
        
        for _ in range(n_samples):
            # Número aleatorio de interferencias (0-3)
            n_interferencias = np.random.poisson(1)
            
            interferencia_total = np.zeros(len(self.wavelengths))
            
            for _ in range(n_interferencias):
                # Pico de interferencia en posición aleatoria
                centro = np.random.choice(len(self.wavelengths))
                ancho = np.random.randint(5, 20)
                intensidad = np.random.uniform(0.001, 0.01)
                
                # Gaussiana de interferencia
                x = np.arange(len(self.wavelengths))
                interferencia = intensidad * np.exp(-0.5 * ((x - centro) / ancho) ** 2)
                interferencia_total += interferencia
            
            interferencias.append(interferencia_total)
        
        return np.array(interferencias)
    
    def generar_dataset_realista(self, n_samples_por_clase=250, anti_overfitting=True):
        """
        Genera dataset sintético más realista para evitar overfitting
        
        Args:
            n_samples_por_clase: Número de muestras por clase
            anti_overfitting: Si aplicar técnicas anti-overfitting
        """
        print(f"🔬 Generando dataset anti-overfitting...")
        print(f"   📊 Muestras por clase: {n_samples_por_clase}")
        print(f"   🛡️ Modo anti-overfitting: {anti_overfitting}")
        
        total_samples = n_samples_por_clase * 2
        
        # Generar efectos realistas
        if anti_overfitting:
            deriva_instrumental, lamp_factor = self.simular_variabilidad_instrumental(total_samples)
            efectos_ambientales, humidity_factor = self.simular_condiciones_ambientales(total_samples)
            interferencias = self.simular_interferencias_quimicas(total_samples)
            
            # Factor de ruido aumentado
            ruido_factor = 0.08  # 8% en lugar de 2%
            print(f"   🔊 Factor de ruido: {ruido_factor} (realista)")
        else:
            # Modo original (más propenso a overfitting)
            deriva_instrumental = np.zeros((total_samples, len(self.wavelengths)))
            lamp_factor = np.ones((total_samples, 1))
            efectos_ambientales = np.zeros((total_samples, len(self.wavelengths)))
            humidity_factor = np.ones((total_samples, 1))
            interferencias = np.zeros((total_samples, len(self.wavelengths)))
            ruido_factor = 0.02
            print(f"   🔊 Factor de ruido: {ruido_factor} (original)")
        
        # Generar muestras de alta concentración
        X_high = []
        for i in range(n_samples_por_clase):
            # Base del espectro
            base_spectrum = np.random.normal(
                self.high_mean, 
                self.high_std * ruido_factor
            )
            
            # Aplicar efectos realistas
            if anti_overfitting:
                spectrum = (
                    base_spectrum * lamp_factor[i] * humidity_factor[i] + 
                    deriva_instrumental[i] + 
                    efectos_ambientales[i] + 
                    interferencias[i]
                )
            else:
                spectrum = base_spectrum
            
            # Asegurar valores positivos
            spectrum = np.maximum(spectrum, 0.001)
            X_high.append(spectrum)
        
        # Generar muestras de baja concentración  
        X_low = []
        for i in range(n_samples_por_clase, total_samples):
            # Base del espectro
            base_spectrum = np.random.normal(
                self.low_mean, 
                self.low_std * ruido_factor
            )
            
            # Aplicar efectos realistas
            if anti_overfitting:
                spectrum = (
                    base_spectrum * lamp_factor[i] * humidity_factor[i] + 
                    deriva_instrumental[i] + 
                    efectos_ambientales[i] + 
                    interferencias[i]
                )
            else:
                spectrum = base_spectrum
            
            # Asegurar valores positivos
            spectrum = np.maximum(spectrum, 0.001)
            X_low.append(spectrum)
        
        # Combinar datos
        X = np.vstack([X_low, X_high])
        y = np.hstack([np.zeros(n_samples_por_clase), np.ones(n_samples_por_clase)])
        
        print(f"   ✅ Dataset creado: {X.shape[0]} muestras x {X.shape[1]} features")
        print(f"   🎯 Distribución: {n_samples_por_clase} baja, {n_samples_por_clase} alta concentración")
        
        return X, y
    
    def validacion_anti_overfitting(self, X, y, modelo_params=None):
        """Validación rigurosa para detectar overfitting"""
        
        print(f"\n🔍 VALIDACIÓN ANTI-OVERFITTING")
        print(f"="*50)
        
        # Configuración de SVM más conservadora
        if modelo_params is None:
            modelo_params = {
                'C': 0.1,  # Regularización más fuerte
                'gamma': 'scale',
                'kernel': 'rbf',
                'random_state': 42
            }
        
        # División train/test más estricta
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # 30% para test
        )
        
        # Normalización
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        modelo = SVC(**modelo_params, probability=True)
        modelo.fit(X_train_scaled, y_train)
        
        # Evaluación en train
        y_train_pred = modelo.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Evaluación en test
        y_test_pred = modelo.predict(X_test_scaled)
        y_test_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Validación cruzada robusta
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_accuracy = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring='f1')
        
        # Análisis de overfitting
        accuracy_gap = train_accuracy - test_accuracy
        f1_gap = train_f1 - test_f1
        
        print(f"📊 RESULTADOS:")
        print(f"   🎯 Train Accuracy: {train_accuracy:.4f}")
        print(f"   🎯 Test Accuracy:  {test_accuracy:.4f}")
        print(f"   📈 Accuracy Gap:   {accuracy_gap:.4f}")
        print(f"")
        print(f"   🎯 Train F1:       {train_f1:.4f}")
        print(f"   🎯 Test F1:        {test_f1:.4f}")
        print(f"   📈 F1 Gap:         {f1_gap:.4f}")
        print(f"")
        print(f"   🎯 Test AUC:       {test_auc:.4f}")
        print(f"")
        print(f"   📊 CV Accuracy:    {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"   📊 CV F1:          {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        # Diagnóstico de overfitting
        print(f"\n🩺 DIAGNÓSTICO DE OVERFITTING:")
        
        overfitting_score = 0
        
        # Test 1: Gap entre train y test
        if accuracy_gap > 0.1:
            print(f"   🚩 Accuracy gap alto: {accuracy_gap:.4f} (> 0.1)")
            overfitting_score += 2
        elif accuracy_gap > 0.05:
            print(f"   ⚠️ Accuracy gap moderado: {accuracy_gap:.4f}")
            overfitting_score += 1
        else:
            print(f"   ✅ Accuracy gap aceptable: {accuracy_gap:.4f}")
        
        # Test 2: Resultados demasiado perfectos
        if test_accuracy > 0.98:
            print(f"   🚩 Test accuracy sospechosamente alto: {test_accuracy:.4f}")
            overfitting_score += 2
        elif test_accuracy > 0.95:
            print(f"   ⚠️ Test accuracy muy alto: {test_accuracy:.4f}")
            overfitting_score += 1
        else:
            print(f"   ✅ Test accuracy realista: {test_accuracy:.4f}")
        
        # Test 3: Variabilidad en CV
        if cv_accuracy.std() < 0.01:
            print(f"   🚩 CV std muy baja: {cv_accuracy.std():.4f} (poca variación)")
            overfitting_score += 1
        else:
            print(f"   ✅ CV std saludable: {cv_accuracy.std():.4f}")
        
        # Test 4: AUC válido
        if test_auc < 0.5 or test_auc > 0.99:
            print(f"   🚩 AUC inválido: {test_auc:.4f}")
            overfitting_score += 2
        else:
            print(f"   ✅ AUC válido: {test_auc:.4f}")
        
        # Diagnóstico final
        if overfitting_score >= 4:
            diagnostico = "🚨 OVERFITTING SEVERO"
        elif overfitting_score >= 2:
            diagnostico = "⚠️ POSIBLE OVERFITTING"
        else:
            diagnostico = "✅ MODELO SALUDABLE"
        
        print(f"\n🏁 DIAGNÓSTICO FINAL: {diagnostico}")
        print(f"   Puntuación overfitting: {overfitting_score}/8")
        
        resultados = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'accuracy_gap': accuracy_gap,
            'f1_gap': f1_gap,
            'overfitting_score': overfitting_score,
            'diagnostico': diagnostico,
            'modelo': modelo,
            'scaler': scaler
        }
        
        return resultados

# Función de uso fácil
def test_overfitting_contaminante(archivo_firma):
    """Función para probar overfitting en un contaminante específico"""
    
    print(f"🧪 PROBANDO OVERFITTING: {archivo_firma}")
    print(f"="*70)
    
    # Cargar firma espectral
    df_firma = pd.read_csv(archivo_firma)
    
    # Crear generador
    generador = DatasetAntiOverfitting(df_firma)
    
    # Probar ambos enfoques
    print(f"\n1️⃣ DATASET ORIGINAL (propenso a overfitting):")
    X_original, y_original = generador.generar_dataset_realista(
        n_samples_por_clase=100, 
        anti_overfitting=False
    )
    resultados_original = generador.validacion_anti_overfitting(X_original, y_original)
    
    print(f"\n2️⃣ DATASET ANTI-OVERFITTING (más realista):")
    X_realista, y_realista = generador.generar_dataset_realista(
        n_samples_por_clase=200, 
        anti_overfitting=True
    )
    resultados_realista = generador.validacion_anti_overfitting(X_realista, y_realista)
    
    # Comparación
    print(f"\n📊 COMPARACIÓN FINAL:")
    print(f"="*50)
    print(f"{'Métrica':<20} {'Original':<12} {'Anti-Overfitting':<15}")
    print(f"{'-'*50}")
    print(f"{'Test Accuracy':<20} {resultados_original['test_accuracy']:<12.4f} {resultados_realista['test_accuracy']:<15.4f}")
    print(f"{'Test F1':<20} {resultados_original['test_f1']:<12.4f} {resultados_realista['test_f1']:<15.4f}")
    print(f"{'Test AUC':<20} {resultados_original['test_auc']:<12.4f} {resultados_realista['test_auc']:<15.4f}")
    print(f"{'Accuracy Gap':<20} {resultados_original['accuracy_gap']:<12.4f} {resultados_realista['accuracy_gap']:<15.4f}")
    print(f"{'Overfitting Score':<20} {resultados_original['overfitting_score']:<12.0f} {resultados_realista['overfitting_score']:<15.0f}")
    
    return resultados_original, resultados_realista

if __name__ == "__main__":
    # Ejemplo de uso
    archivo_ejemplo = "todo/firmas_espectrales_csv/Turbidity_Ntu_datos_espectrales.csv"
    
    if os.path.exists(archivo_ejemplo):
        resultados_orig, resultados_nuevo = test_overfitting_contaminante(archivo_ejemplo)
    else:
        print("❌ Archivo de ejemplo no encontrado")
        print("📝 Para usar, ejecuta:")
        print("test_overfitting_contaminante('tu_archivo_firma.csv')")