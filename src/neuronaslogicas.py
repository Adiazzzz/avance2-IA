"""
neuronas_logicas.py
Actividad independiente - Clase 8
Neuronas artificiales que aprenden las compuertas lógicas OR y NOT.
La neurona está hecha en Python puro; solo el gráfico usa Seaborn.
Al final hay una prueba extra con XOR para responder la pregunta del README.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0


def neurona(x1, x2, w1, w2, b):
    """Neurona de dos entradas (se usa para OR y XOR)."""
    z = w1 * x1 + w2 * x2 + b
    return escalon(z)

def neurona_not(x1, w1, b):
    """Neurona de una sola entrada (se usa para NOT)."""
    z = w1 * x1 + b
    return escalon(z)


def entrenar_dos_entradas(nombre, datos, w1, w2, b, tasa_aprendizaje, max_epocas=10):
    """Entrena una neurona de dos entradas. Devuelve los errores por época."""
    print("=" * 60)
    print(f"ENTRENAMIENTO DE UNA NEURONA PARA LA COMPUERTA {nombre}")
    print("=" * 60)
    print(f"Parámetros iniciales: w1={w1}, w2={w2}, b={b}\n")

    errores_por_epoca = []
    epocas_necesarias = None

    for epoca in range(max_epocas):
        error_total = 0
        print(f"--- Época {epoca + 1} ---")
        for x1, x2, y_real in datos:
            y_pred = neurona(x1, x2, w1, w2, b)
            error = y_real - y_pred
            error_total += abs(error)

            w1 += tasa_aprendizaje * error * x1
            w2 += tasa_aprendizaje * error * x2
            b += tasa_aprendizaje * error

            print(f" Entrada: ({x1}, {x2}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")

        errores_por_epoca.append(error_total)
        print(f" Error total: {error_total}")
        print(f" Parámetros actuales: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}\n")

        if error_total == 0:
            epocas_necesarias = epoca + 1
            print(f"La neurona aprendió en {epocas_necesarias} épocas.\n")
            break

    if epocas_necesarias is None:
        print(f"La neurona NO aprendió en {max_epocas} épocas.\n")

    print(f"PRUEBA FINAL - {nombre}")
    for x1, x2, y_real in datos:
        y_pred = neurona(x1, x2, w1, w2, b)
        estado = "OK" if y_pred == y_real else "X"
        print(f" {estado} Entrada: ({x1}, {x2}) | Esperado: {y_real} | Predicho: {y_pred}")
    print()
    return errores_por_epoca, epocas_necesarias


def entrenar_una_entrada(nombre, datos, w1, b, tasa_aprendizaje, max_epocas=10):
    """Entrena una neurona de una entrada (NOT). Devuelve los errores por época."""
    print("=" * 60)
    print(f"ENTRENAMIENTO DE UNA NEURONA PARA LA COMPUERTA {nombre}")
    print("=" * 60)
    print(f"Parámetros iniciales: w1={w1}, b={b}\n")

    errores_por_epoca = []
    epocas_necesarias = None

    for epoca in range(max_epocas):
        error_total = 0
        print(f"--- Época {epoca + 1} ---")
        for x1, y_real in datos:
            y_pred = neurona_not(x1, w1, b)
            error = y_real - y_pred
            error_total += abs(error)

            w1 += tasa_aprendizaje * error * x1
            b += tasa_aprendizaje * error

            print(f" Entrada: ({x1}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")

        errores_por_epoca.append(error_total)
        print(f" Error total: {error_total}")
        print(f" Parámetros actuales: w1={w1:.3f}, b={b:.3f}\n")

        if error_total == 0:
            epocas_necesarias = epoca + 1
            print(f"La neurona aprendió en {epocas_necesarias} épocas.\n")
            break

    if epocas_necesarias is None:
        print(f"La neurona NO aprendió en {max_epocas} épocas.\n")

    print(f"PRUEBA FINAL - {nombre}")
    for x1, y_real in datos:
        y_pred = neurona_not(x1, w1, b)
        estado = "OK" if y_pred == y_real else "X"
        print(f" {estado} Entrada: ({x1}) | Esperado: {y_real} | Predicho: {y_pred}")
    print()
    return errores_por_epoca, epocas_necesarias


datos_or = [
    (0, 0, 0),  
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
]

datos_not = [
    (0, 1),     
    (1, 0),
]

datos_xor = [   
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

tasa_aprendizaje = 0.1


errores_or, epocas_or = entrenar_dos_entradas("OR", datos_or, 0.1, 0.1, 0.0, tasa_aprendizaje)
errores_not, epocas_not = entrenar_una_entrada("NOT", datos_not, 0.1, 0.0, tasa_aprendizaje)

# Prueba extra: XOR (no debería aprender con una sola neurona)
errores_xor, epocas_xor = entrenar_dos_entradas("XOR (prueba extra)", datos_xor, 0.1, 0.1, 0.0, tasa_aprendizaje)


print("=" * 60)
print("RESUMEN")
print("=" * 60)
print(f" OR  -> épocas necesarias: {epocas_or}")
print(f" NOT -> épocas necesarias: {epocas_not}")
print(f" XOR -> épocas necesarias: {epocas_xor}  (nunca aprendió)")


os.makedirs("outputs/graficos", exist_ok=True)
sns.set_theme(style="whitegrid")


filas = []
for nombre, errores in [("OR", errores_or), ("NOT", errores_not)]:
    for epoca, error in enumerate(errores, start=1):
        filas.append({"Compuerta": nombre, "Época": epoca, "Error total": error})
df_errores = pd.DataFrame(filas)

plt.figure(figsize=(8, 5))
sns.lineplot(data=df_errores, x="Época", y="Error total",
             hue="Compuerta", style="Compuerta", markers=True, dashes=False)
plt.title('Evolución del Error: compuertas OR y NOT')
plt.xticks(range(1, df_errores["Época"].max() + 1))
plt.ylim(-0.3, df_errores["Error total"].max() + 1)
plt.savefig('outputs/graficos/error_or_not.png', dpi=150)
plt.close()
print("Gráfico guardado: error_or_not.png")


df_xor = pd.DataFrame({
    "Época": range(1, len(errores_xor) + 1),
    "Error total": errores_xor,
})
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_xor, x="Época", y="Error total", marker="o", color="red")
plt.title('Evolución del Error: compuerta XOR (no converge)')
plt.xticks(range(1, len(errores_xor) + 1))
plt.ylim(-0.3, df_xor["Error total"].max() + 1)
plt.savefig('outputs/graficos/error_xor.png', dpi=150)
plt.close()
print("Gráfico guardado: error_xor.png")