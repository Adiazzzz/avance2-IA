"""
neurona_and.py
Implementación de una neurona artificial simple para aprender la compuerta AND.
No se usan librerías externas, solo Python puro.
"""
import matplotlib.pyplot as plt          # ← el import sube aquí, al inicio
# ============================================================
# 1. Definir la función de activación (escalón)
# ============================================================
def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0
# ============================================================
# 2. Definir la neurona
# ============================================================
def neurona(x1, x2, w1, w2, b):
    """Calcula la salida de la neurona."""
    z = w1 * x1 + w2 * x2 + b
    return escalon(z)
# ============================================================
# 3. Datos de entrenamiento (compuerta AND)
# ============================================================
datos = [
    (0, 0, 0),  # (x1, x2, y_real)
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]
# ============================================================
# 4. Parámetros iniciales (los vamos a ajustar)
# ============================================================
w1, w2, b = 0.1, 0.1, 0.0
tasa_aprendizaje = 0.1
# ============================================================
# 5. Entrenamiento manual (varias épocas)
# ============================================================
print("=" * 60)
print("ENTRENAMIENTO DE UNA NEURONA PARA LA COMPUERTA AND")
print("=" * 60)
print(f"Parámetros iniciales: w1={w1}, w2={w2}, b={b}\n")

# Lista para guardar los errores por época
errores_por_epoca = []                   # ← ANTES del bucle

for epoca in range(10):
    error_total = 0
    print(f"--- Época {epoca + 1} ---")
    for x1, x2, y_real in datos:
        # Predicción
        y_pred = neurona(x1, x2, w1, w2, b)

        # Cálculo del error
        error = y_real - y_pred
        error_total += abs(error)

        # Ajuste de pesos y bias
        w1 += tasa_aprendizaje * error * x1
        w2 += tasa_aprendizaje * error * x2
        b += tasa_aprendizaje * error

        print(f" Entrada: ({x1}, {x2}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")

    errores_por_epoca.append(error_total)    # ← DENTRO del bucle, antes del break

    print(f" Error total: {error_total}")
    print(f" Parámetros actuales: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}\n")

    if error_total == 0:
        print(f"La neurona aprendió en {epoca + 1} épocas.\n")
        break
# ============================================================
# 6. Prueba final
# ============================================================
print("=" * 60)
print("PRUEBA FINAL")
print("=" * 60)
for x1, x2, y_real in datos:
    y_pred = neurona(x1, x2, w1, w2, b)
    estado = "OK" if y_pred == y_real else "X"
    print(f" {estado} Entrada: ({x1}, {x2}) | Esperado: {y_real} | Predicho: {y_pred}")

# ============================================================
# 7. Gráfico del error por época       ← TODO el gráfico al FINAL, sin sangría
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_por_epoca) + 1), errores_por_epoca, marker='o', color='red')
plt.title('Evolución del Error durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Error Total')
plt.grid(True)
plt.savefig('outputs/graficos/clase9/evolución_error.png', dpi=150)
plt.close()
print("Gráfico guardado: evolución_error.png")