# test_corregido_adaptativo.py
# Test corregido del sistema adaptativo

import sys
import os

# Agregar el path del archivo principal si es necesario
# sys.path.append('ruta_donde_esta_el_archivo_principal')

from entrenamiento_adptativo_contaminantes import EntrenadorAdaptativoContaminantes

def test_sistema_adaptativo_corregido():
    """
    Test del sistema adaptativo con las funciones de validación corregidas
    """
    print("🧬 TEST SISTEMA ADAPTATIVO CORREGIDO")
    print("="*55)
    print("🎯 Probando entrenamiento específico por contaminante")
    print()
    
    # Inicializar entrenador
    entrenador = EntrenadorAdaptativoContaminantes("todo/firmas_espectrales_csv")
    
    # Contaminantes de prueba (diferentes tipos químicos)
    contaminantes_test = {
        'Nh4_Mg_L': {
            'descripcion': 'Ion inorgánico simple (NH4+)',
            'expectativa': 'Dataset minimal, features UV específicos, validación LOO'
        },
        'Caffeine_Ng_L': {
            'descripcion': 'Compuesto orgánico (xantina)',
            'expectativa': 'Features de picos, validación CV estratificada'
        },
        'Turbidity_Ntu': {
            'descripcion': 'Parámetro físico-químico (dispersión)',
            'expectativa': 'Features de dispersión, mayor complejidad'
        }
    }
    
    resultados = {}
    
    for i, (contaminante, info) in enumerate(contaminantes_test.items(), 1):
        print(f"[{i}/3] 🔬 PROCESANDO: {contaminante}")
        print(f"      📋 {info['descripcion']}")
        print(f"      💭 Esperado: {info['expectativa']}")
        print()
        
        try:
            # Ejecutar entrenamiento adaptativo
            resultado = entrenador.entrenar_adaptativo(contaminante)
            
            if resultado:
                resultados[contaminante] = resultado
                
                # Análisis inmediato del resultado
                print(f"      ✅ COMPLETADO:")
                print(f"         🏷️ Tipo: {resultado['tipo_quimico']} - {resultado['categoria_quimica']}")
                print(f"         📊 F1 Score: {resultado['test_f1']:.4f}")
                print(f"         🔢 Features: {resultado['n_features']}")
                print(f"         🔍 Validación: {resultado['estrategia_validacion']}")
                print(f"         📈 Estado: {resultado['diagnostico_overfitting']}")
                
                # Verificar adaptación
                if resultado['n_features'] <= 3 and contaminante == 'Nh4_Mg_L':
                    print(f"         ✅ Adaptación correcta: Ion simple con pocos features")
                elif resultado['n_features'] >= 5 and 'Turbidity' in contaminante:
                    print(f"         ✅ Adaptación correcta: Parámetro complejo con más features")
                
                print()
            else:
                print(f"      ❌ ERROR: No se pudo entrenar")
                print()
                
        except Exception as e:
            print(f"      ❌ EXCEPCIÓN: {str(e)}")
            print()
    
    # Análisis comparativo final
    print("="*70)
    print("📊 ANÁLISIS COMPARATIVO FINAL")
    print("="*70)
    
    if resultados:
        print(f"✅ Resultados obtenidos: {len(resultados)}/3")
        print()
        
        # Tabla de resultados
        print(f"{'Contaminante':<15} | {'Tipo':<12} | {'F1':<6} | {'Features':<8} | {'Validación':<12}")
        print("-" * 70)
        
        f1_scores = []
        for cont, res in resultados.items():
            tipo = res['tipo_quimico'][:10]
            f1 = res['test_f1']
            features = res['n_features']
            validacion = res['estrategia_validacion'][:10]
            
            f1_scores.append(f1)
            
            # Determinar estado
            if f1 > 0.7:
                estado = "🟢"
            elif f1 > 0.5:
                estado = "🟡"
            else:
                estado = "🔴"
            
            print(f"{cont:<15} | {tipo:<12} | {f1:<6.3f} | {features:<8} | {validacion:<12} {estado}")
        
        print()
        
        # Verificar diversidad (señal de adaptación exitosa)
        f1_unique = len(set([round(f1, 2) for f1 in f1_scores]))
        
        if f1_unique > 1:
            print("✅ ADAPTACIÓN EXITOSA: Resultados diversos entre contaminantes")
            print("   - Cada contaminante muestra comportamiento específico")
            print("   - El sistema se adapta según propiedades químicas")
        else:
            print("⚠️ ADAPTACIÓN PARCIAL: Resultados similares")
            print("   - Puede indicar limitaciones en los datos")
            print("   - O necesidad de ajustar perfiles químicos")
        
        print()
        
        # Análisis por tipo químico
        tipos_encontrados = set(res['tipo_quimico'] for res in resultados.values())
        print(f"🧪 Tipos químicos procesados: {len(tipos_encontrados)}")
        for tipo in tipos_encontrados:
            contaminantes_tipo = [cont for cont, res in resultados.items() 
                                if res['tipo_quimico'] == tipo]
            f1_promedio = np.mean([resultados[cont]['test_f1'] for cont in contaminantes_tipo])
            print(f"   - {tipo}: {len(contaminantes_tipo)} contaminante(s), F1 promedio: {f1_promedio:.3f}")
        
        print()
        
        # Recomendaciones
        print("💡 RECOMENDACIONES:")
        
        # Análisis de overfitting por tipo
        overfitting_por_tipo = {}
        for cont, res in resultados.items():
            tipo = res['tipo_quimico']
            overfitting = res['diagnostico_overfitting']
            if tipo not in overfitting_por_tipo:
                overfitting_por_tipo[tipo] = []
            overfitting_por_tipo[tipo].append(overfitting)
        
        for tipo, diagnosticos in overfitting_por_tipo.items():
            if any('SEVERO' in d for d in diagnosticos):
                print(f"   🚨 {tipo}: Requiere mayor regularización")
            elif any('MODERADO' in d for d in diagnosticos):
                print(f"   ⚠️ {tipo}: Ajustar parámetros")
            else:
                print(f"   ✅ {tipo}: Configuración adecuada")
        
        print()
        print("🎯 PRÓXIMOS PASOS:")
        print("   1. Refinar perfiles de contaminantes según resultados")
        print("   2. Ajustar parámetros para tipos problemáticos")
        print("   3. Expandir a todos los contaminantes disponibles")
        print("   4. Validar resultados con conocimiento químico")
        
    else:
        print("❌ No se obtuvieron resultados válidos")
        print("   Verificar datos de entrada y configuración")
    
    return resultados

if __name__ == "__main__":
    # Importar numpy aquí si no está disponible globalmente
    import numpy as np
    
    # Ejecutar test
    resultados = test_sistema_adaptativo_corregido()
    
    print(f"\n🏁 Test completado: {len(resultados)} resultados obtenidos")