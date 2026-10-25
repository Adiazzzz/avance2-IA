
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Sin pantalla (Docker/WSL): los gráficos solo se guardan en archivo
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


RAIZ = Path(__file__).resolve().parents[1]
RUTA_CSV = RAIZ / "data" / "proyecto_limpio.csv"
SALIDA = RAIZ / "outputs" / "avance2"

ENTRADAS = ["plazo_dias", "n_ejemplares"]
OBJETIVO = "con_retraso"

TASA_APRENDIZAJE = 0.1
EPOCAS = 1000
SEMILLA = 42

CLASE_0 = "A tiempo (0)"
CLASE_1 = "Con retraso (1)"
ETIQUETA_X1 = "Plazo del préstamo"
ETIQUETA_X2 = "Ejemplares por préstamo"


def sigmoide(z):
    """Convierte z en una probabilidad entre 0 y 1."""
    return 1.0 / (1.0 + np.exp(-z))


def predecir_probabilidad(X, w, b):
    """z = X·w + b, pasado por la sigmoide (X tiene una fila por préstamo)."""
    return sigmoide(X @ w + b)


def perdida_logaritmica(y, p):
    """Entropía cruzada binaria. Se recorta p para evitar log(0)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def exactitud(y, p, umbral=0.5):
    """Proporción de aciertos al clasificar con p >= 0.5 como retraso."""
    return np.mean((p >= umbral).astype(int) == y)


def entrenar(X, y, w, b, tasa, epocas):
    """Descenso del gradiente por lotes. Devuelve w, b y la pérdida por época."""
    n = len(y)
    historial = []
    for epoca in range(epocas):
        p = predecir_probabilidad(X, w, b)
        historial.append(perdida_logaritmica(y, p))

        error = p - y                    
        grad_w = X.T @ error / n         
        grad_b = error.mean()            

        w = w - tasa * grad_w
        b = b - tasa * grad_b

        if epoca == 0 or (epoca + 1) % 100 == 0:
            print(f" Época {epoca + 1:>4} | pérdida = {historial[-1]:.4f} "
                  f"| w1 = {w[0]:+.4f}, w2 = {w[1]:+.4f}, b = {b:+.4f}")
    return w, b, historial



def cargar_y_dividir():
    
    try:
        df = pd.read_csv(RUTA_CSV)
    except FileNotFoundError:
        raise SystemExit(f"No existe {RUTA_CSV}. Ejecuta primero analisis_biblioteca.py")

    X = df[ENTRADAS].to_numpy(dtype=float)
    y = df[OBJETIVO].to_numpy(dtype=int)

    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(len(y))
    corte = int(0.8 * len(y))
    idx_ent, idx_pru = orden[:corte], orden[corte:]
    return X[idx_ent], y[idx_ent], X[idx_pru], y[idx_pru]


def estandarizar(X_ent, X_pru):
    """z = (x - media) / desviación. Media y desviación salen SOLO del entrenamiento."""
    media = X_ent.mean(axis=0)
    desv = X_ent.std(axis=0)
    return (X_ent - media) / desv, (X_pru - media) / desv, media, desv


#
def grafico_curva(historial):
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=np.arange(1, len(historial) + 1), y=historial, color="crimson")
    plt.axhline(np.log(2), color="gray", linestyle="--", linewidth=1,
                label="Adivinar al azar (0.693)")
    plt.title("Curva de aprendizaje (retraso): pérdida logarítmica por época")
    plt.xlabel("Época")
    plt.ylabel("Pérdida logarítmica")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA / "curva_aprendizaje_retraso.png", dpi=150)
    plt.close()


def grafico_frontera(X, y, w, b, exactitud_ent, media, desv):
    """Muestra las clases observadas y las regiones que predice el modelo."""
    rng = np.random.default_rng(SEMILLA)
    X_original = X * desv + media
    rangos = X_original.max(axis=0) - X_original.min(axis=0)
    ruido = rng.normal(0, np.maximum(rangos * 0.008, [0.1, 0.03]), X.shape)
    datos = pd.DataFrame({
        "x1": X_original[:, 0] + ruido[:, 0],
        "x2": X_original[:, 1] + ruido[:, 1],
        "Resultado": np.where(y == 1, CLASE_1, CLASE_0),
    })

    margen = np.maximum(rangos * 0.08, [1, 0.2])
    x1 = np.linspace(X_original[:, 0].min() - margen[0],
                     X_original[:, 0].max() + margen[0], 250)
    x2 = np.linspace(X_original[:, 1].min() - margen[1],
                     X_original[:, 1].max() + margen[1], 250)
    xx, yy = np.meshgrid(x1, x2)
    rejilla = np.column_stack([
        ((xx.ravel() - media[0]) / desv[0]),
        ((yy.ravel() - media[1]) / desv[1]),
    ])
    probabilidades = predecir_probabilidad(rejilla, w, b).reshape(xx.shape)

    plt.figure(figsize=(8, 5.5))
    plt.contourf(xx, yy, probabilidades, levels=[0, 0.5, 1],
                 colors=["#e7f0f8", "#fbe9e9"], alpha=0.9)
    plt.contour(xx, yy, probabilidades, levels=[0.5], colors="black", linewidths=2)
    sns.scatterplot(data=datos, x="x1", y="x2", hue="Resultado",
                    palette={CLASE_1: "crimson", CLASE_0: "steelblue"},
                    alpha=0.45, s=35)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title(f"Frontera de decisión (retraso) - exactitud en entrenamiento: {exactitud_ent:.1%}")
    plt.xlabel(f"{ETIQUETA_X1} (días)")
    plt.ylabel(f"{ETIQUETA_X2} (cantidad)")
    leyenda = [
        Line2D([0], [0], marker="o", linestyle="none", color="steelblue",
               label=CLASE_0),
        Line2D([0], [0], marker="o", linestyle="none", color="crimson",
               label=CLASE_1),
        Patch(facecolor="#e7f0f8", label="El modelo predice a tiempo"),
        Patch(facecolor="#fbe9e9", label="El modelo predice retraso"),
        Line2D([0], [0], color="black", linewidth=2, label="Límite (p = 0.5)"),
    ]
    plt.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.17),
               ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(SALIDA / "frontera_decision_retraso.png", dpi=150)
    plt.close()



def main():
    print("=" * 70)
    print("PERCEPTRÓN 1 - ¿SE DEVOLVERÁ EL PRÉSTAMO CON RETRASO?")
    print("=" * 70)
    SALIDA.mkdir(parents=True, exist_ok=True)

    X_ent, y_ent, X_pru, y_pru = cargar_y_dividir()
    print(f"Entrenamiento: {len(y_ent)} préstamos | Prueba: {len(y_pru)} préstamos")
    print(f"Proporción de retrasos en entrenamiento: {y_ent.mean():.1%}")

    X_ent, X_pru, media, desv = estandarizar(X_ent, X_pru)
    print(f"\nEstandarización (media, desviación): "
          f"plazo_dias = ({media[0]:.2f}, {desv[0]:.2f}) | "
          f"n_ejemplares = ({media[1]:.2f}, {desv[1]:.2f})")

   
    w = np.array([0.1, 0.1])
    b = 0.0
    perdida_inicial = perdida_logaritmica(y_ent, predecir_probabilidad(X_ent, w, b))
    print(f"\nParámetros iniciales: w1={w[0]}, w2={w[1]}, b={b}")
    print(f"Tasa de aprendizaje: {TASA_APRENDIZAJE} | Épocas: {EPOCAS}\n")

    w, b, historial = entrenar(X_ent, y_ent, w, b, TASA_APRENDIZAJE, EPOCAS)

    p_ent = predecir_probabilidad(X_ent, w, b)
    p_pru = predecir_probabilidad(X_pru, w, b)
    ex_ent, ex_pru = exactitud(y_ent, p_ent), exactitud(y_pru, p_pru)
    base_ent = max(y_ent.mean(), 1 - y_ent.mean())
    base_pru = max(y_pru.mean(), 1 - y_pru.mean())

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"Pesos finales:  w1 (plazo_dias) = {w[0]:+.4f} | "
          f"w2 (n_ejemplares) = {w[1]:+.4f} | b = {b:+.4f}")
    print(f"Pérdida inicial: {perdida_inicial:.4f} | pérdida final: {historial[-1]:.4f}")
    print(f"Exactitud en entrenamiento: {ex_ent:.1%} | en prueba: {ex_pru:.1%}")
    print(f"Referencia (predecir siempre la clase más común): "
          f"{base_ent:.1%} en entrenamiento | {base_pru:.1%} en prueba")

    
    print("\nINTERPRETACIÓN")
    for nombre, peso, d in zip(ENTRADAS, w, desv):
        sentido = "aumenta" if peso > 0 else "disminuye"
        print(f" {nombre}: peso {peso:+.3f} -> subir 1 desviación (~{d:.1f} unidades) "
              f"{sentido} la probabilidad de retraso; multiplica las odds por {np.exp(peso):.2f}")

    
    plazo_critico = media[0] + desv[0] * (-b / w[0]) if abs(w[0]) > 1e-9 else float("nan")
    print(f" Con el número de ejemplares promedio, p = 0.5 cuando el plazo es "
          f"de unos {plazo_critico:.1f} días.")

    if ex_ent < 0.99:
        print(f" Los datos NO parecen linealmente separables: la mejor recta acierta "
              f"{ex_ent:.1%}, no 100 %.")

    grafico_curva(historial)
    grafico_frontera(X_ent, y_ent, w, b, ex_ent, media, desv)
    print(f"\nGráficos guardados en {SALIDA}: "
          "curva_aprendizaje_retraso.png y frontera_decision_retraso.png")


if __name__ == "__main__":
    main()