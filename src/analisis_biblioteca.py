"""
analisis_biblioteca.py
Actividad independiente - Clase 8 (Pandas + Seaborn).

EDA, limpieza y preparación de los datos del proyecto Biblioteca COTECNOVA.
El producto final es una tabla con un registro por préstamo, lista para un
modelo que prediga si un préstamo se devolverá con retraso.

Avance 2 (Clase 7 - preparación de datos): se agregó la revisión de duplicados,
categorías, tipos, valores imposibles y atípicos, la tabla antes / después y la
exportación del dataset limpio a data/proyecto_limpio.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency 

sns.set_theme(style="whitegrid", context="notebook")


RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
GRAFICOS = RAIZ / "outputs" / "graficos"
SALIDA_ML = RAIZ / "outputs" / "prestamos_preparados_ml.csv"
RUTA_LIMPIO = DATA / "proyecto_limpio.csv"          # dataset limpio pedido en el Avance 2
SALIDA_ANTES_DESPUES = RAIZ / "outputs" / "antes_despues.csv"
RUTA_LIBROS = DATA / "libros_demanda.csv"            # dataset por libro (segundo perceptrón)

TABLAS = [
    "libro", "ejemplar", "genero", "libro_genero", "editorial",
    "autor", "autor_libro", "usuario", "prestamo", "prestamo_ejemplar",
]

DETALLADAS = ["prestamo", "ejemplar", "usuario"]


COLUMNAS_USUARIO = ["id_usuario", "rol", "limite_prestamos", "fecha_registro"]

# Llave primaria de cada tabla (en las tablas de relación la llave son dos columnas)
LLAVES = {
    "libro": ["id_libro"], "ejemplar": ["id_ejemplar"], "genero": ["id_genero"],
    "editorial": ["id_editorial"], "autor": ["id_autor"], "usuario": ["id_usuario"],
    "prestamo": ["id_prestamo"], "libro_genero": ["id_libro", "id_genero"],
    "autor_libro": ["id_autor", "id_libro"],
    "prestamo_ejemplar": ["id_prestamo", "id_ejemplar"],
}

# Columnas de texto con pocas categorías que se usan en el proyecto
CATEGORICAS = [("usuario", "rol"), ("prestamo_ejemplar", "estado"),
               ("ejemplar", "tipo_material")]



# 1. CARGA Y EXPLORACIÓN

def cargar_tablas():
    """Carga cada CSV de data/ en un diccionario de DataFrames."""
    tablas = {}
    for nombre in TABLAS:
        ruta = DATA / f"{nombre}.csv"
        try:
            tablas[nombre] = pd.read_csv(ruta)
            f, c = tablas[nombre].shape
            print(f"[OK] {nombre:<18} {f:>5} filas x {c:>2} columnas")
        except FileNotFoundError:
            print(f"[AVISO] No se encontró {ruta}")
    return tablas


def explorar_tabla(nombre, df, detallada):
    """head(), info(), describe() e isnull().sum() de una tabla."""
    print("\n" + "=" * 70)
    print(f"TABLA: {nombre}  ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print("=" * 70)
    print(df.head(3).to_string(max_colwidth=30))
    if detallada:
        print("\n-- info() --")
        df.info()
        print("\n-- describe() --")
        print(df.describe().round(2).to_string())
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    print("\n-- nulos por columna --")
    if nulos.empty:
        print("Sin valores nulos.")
    else:
        resumen = pd.DataFrame({"nulos": nulos,
                                "porcentaje": (nulos / len(df) * 100).round(1)})
        print(resumen.to_string())


def explorar_todo(tablas):
    for nombre, df in tablas.items():
        explorar_tabla(nombre, df, detallada=nombre in DETALLADAS)


def revisar_duplicados(tablas):
    """Duplicados por fila completa y por llave primaria en cada tabla."""
    print("\n" + "=" * 70)
    print("REVISIÓN DE DUPLICADOS")
    print("=" * 70)
    total = 0
    for nombre, df in tablas.items():
        filas = int(df.duplicated().sum())
        llave = int(df.duplicated(subset=LLAVES[nombre]).sum())
        total += filas + llave
        print(f"{nombre:<18} filas repetidas: {filas:>2} | llave repetida "
              f"({', '.join(LLAVES[nombre])}): {llave:>2}")
    if total == 0:
        print("-> No hay duplicados: no se elimina ninguna fila.")
    else:
        print(f"-> {total} duplicados encontrados: revisar antes de continuar.")


def revisar_categorias(tablas):
    """Busca variantes de una misma categoría (espacios, mayúsculas/minúsculas)."""
    print("\n" + "=" * 70)
    print("REVISIÓN DE CATEGORÍAS INCONSISTENTES")
    print("=" * 70)
    for nombre, col in CATEGORICAS:
        s = tablas[nombre][col].dropna().astype(str)
        con_espacios = int((s != s.str.strip()).sum())
        variantes = s.nunique() - s.str.strip().str.lower().nunique()
        print(f"{nombre}.{col}: {s.nunique()} valores -> {s.value_counts().to_dict()}")
        if con_espacios == 0 and variantes == 0:
            print("   Sin espacios sobrantes ni variantes de mayúsculas: no se corrige nada.")
        else:
            print(f"   {con_espacios} con espacios y {variantes} variantes: unificar.")
    print("Nota: el estado 'Danado' se escribe sin tilde en todo el archivo; es un "
          "único valor consistente, se deja igual.")



# 2. LIMPIEZA

def limpiar_tablas(tablas):
    """
    Maneja los nulos según su significado:
    - segundo_nombre / observaciones: campos OPCIONALES. No son un dato
      perdido, así que no se imputan con media o mediana.
    - fecha_devolucion: nulo = préstamo todavía sin cerrar. Inventar una
      fecha falsearía el análisis, por eso se conserva el nulo y se marca.
    """
    t = {k: v.copy() for k, v in tablas.items()}

    for nombre in ("autor", "usuario"):
        if nombre in t and "segundo_nombre" in t[nombre]:
            n = int(t[nombre]["segundo_nombre"].isnull().sum())
            t[nombre]["segundo_nombre"] = t[nombre]["segundo_nombre"].fillna("")
            print(f"{nombre}.segundo_nombre: {n} nulos -> '' (campo opcional)")

    p = t["prestamo"]
    for col in ("fecha_prestamo", "fecha_devolucion", "fecha_limite"):
        p[col] = pd.to_datetime(p[col], errors="coerce")
    print(f"prestamo.fecha_devolucion: {int(p['fecha_devolucion'].isnull().sum())} "
          "nulos conservados (préstamos sin cerrar)")

    u = t["usuario"]
    u["fecha_registro"] = pd.to_datetime(u["fecha_registro"], errors="coerce")
    print("Tipos de dato: prestamo.fecha_prestamo, fecha_devolucion, fecha_limite y "
          "usuario.fecha_registro estaban como texto -> convertidas a datetime")
    return t



# 3. DATASET POR PRÉSTAMO

def construir_prestamos(t):
    """Un registro por préstamo, con columnas derivadas útiles al proyecto."""
    p = t["prestamo"].copy()

    pe = t["prestamo_ejemplar"].assign(
        perdido=lambda d: (d["estado"] == "Perdido").astype(int),
        danado=lambda d: (d["estado"] == "Danado").astype(int),
    )
    agregados = pe.groupby("id_prestamo").agg(
        n_ejemplares=("id_ejemplar", "size"),
        n_perdidos=("perdido", "sum"),
        n_danados=("danado", "sum"),
    ).reset_index()

    df = (p.merge(agregados, on="id_prestamo", how="left")
           .merge(t["usuario"][COLUMNAS_USUARIO], on="id_usuario", how="left"))

    corte = df[["fecha_prestamo", "fecha_devolucion", "fecha_limite"]].max().max()

    # --- columnas derivadas ---
    df["plazo_dias"] = (df["fecha_limite"] - df["fecha_prestamo"]).dt.days
    df["devuelto"] = df["fecha_devolucion"].notna()
    df["dias_prestamo"] = (df["fecha_devolucion"] - df["fecha_prestamo"]).dt.days
    df["dias_retraso"] = (df["fecha_devolucion"] - df["fecha_limite"]).dt.days
    df["con_retraso"] = (df["dias_retraso"] > 0).astype("Int64").where(df["devuelto"])
    df["tiene_observacion"] = df["observaciones"].notna().astype(int)
    df["dias_sin_cerrar"] = (corte - df["fecha_prestamo"]).dt.days.where(~df["devuelto"])
    df["prestamo_antes_registro"] = df["fecha_prestamo"] < df["fecha_registro"]
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["dia_semana"] = df["fecha_prestamo"].dt.dayofweek
    df.attrs["fecha_corte"] = corte
    return df


def revisar_imposibles_y_atipicos(df):
    """Valores imposibles y atípicos del dataset por préstamo, con la decisión tomada."""
    print("\n" + "=" * 70)
    print("VALORES IMPOSIBLES Y ATÍPICOS")
    print("=" * 70)
    imposibles = {
        "plazo_dias <= 0": int((df["plazo_dias"] <= 0).sum()),
        "devolución anterior al préstamo": int((df["fecha_devolucion"] < df["fecha_prestamo"]).sum()),
        "multa negativa": int((df["multa"] < 0).sum()),
    }
    for regla, n in imposibles.items():
        print(f"{regla:<34} {n:>4}")
    if sum(imposibles.values()) == 0:
        print("-> Ningún valor imposible: no se elimina nada.")

    n_antes = int(df["prestamo_antes_registro"].sum())
    print(f"\nPréstamos anteriores al registro del usuario: {n_antes} ({n_antes / len(df):.1%}).")
    print("   Decisión: se conservan y quedan marcados con la columna "
          "'prestamo_antes_registro' (en el dataset de trabajo; no pasa al CSV final porque "
          "no es entrada del modelo). Ninguna variable de entrada del modelo depende de "
          "fecha_registro, y borrarlos quitaría más de una cuarta parte de los datos.")

    n_exceden = int((df["n_ejemplares"] > df["limite_prestamos"]).sum())
    print(f"Préstamos con más ejemplares que el límite del usuario: {n_exceden}. "
          "Se conservan: no afectan al objetivo.")

    r = df["dias_retraso"].dropna()
    q1, q3 = r.quantile([0.25, 0.75])
    iqr = q3 - q1
    bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_atip = int(((r < bajo) | (r > alto)).sum())
    print(f"\nAtípicos en dias_retraso (regla 1.5 x IQR): rango válido [{bajo:.1f}, {alto:.1f}], "
          f"datos de {r.min():.0f} a {r.max():.0f} -> {n_atip} atípicos. No se elimina nada.")


def tabla_antes_despues(antes, despues):
    """Compara filas, columnas y nulos de prestamo.csv (antes) y el dataset final (después)."""
    tabla = pd.DataFrame({
        "antes (prestamo.csv)": [antes.shape[0], antes.shape[1], int(antes.isnull().sum().sum())],
        "despues (proyecto_limpio.csv)": [despues.shape[0], despues.shape[1],
                                          int(despues.isnull().sum().sum())],
    }, index=["filas", "columnas", "nulos"])
    print("\n" + "=" * 70)
    print("TABLA ANTES / DESPUÉS")
    print("=" * 70)
    print(tabla.to_string())
    SALIDA_ANTES_DESPUES.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA_ANTES_DESPUES)
    print(f"\nTabla guardada en {SALIDA_ANTES_DESPUES}")
    return tabla


def construir_demanda_libros(t):
    """Por libro: cuántos ejemplares tiene y cuántas veces se ha prestado."""
    ej = t["ejemplar"][["id_ejemplar", "id_libro"]]
    veces = (t["prestamo_ejemplar"].merge(ej, on="id_ejemplar")
             .groupby("id_libro").size().rename("prestamos"))
    libros = (ej.groupby("id_libro").size().rename("ejemplares").reset_index()
              .merge(veces, on="id_libro", how="left").fillna({"prestamos": 0}))
    return libros


def exportar_libros_demanda(t, libros):
    """
    Dataset con un registro por libro para el segundo perceptrón (demanda del libro).
    - prestado: variable objetivo 0/1 (1 = el libro se ha prestado al menos una vez).
    - prestamos: cuántas veces se prestó. Se guarda solo como referencia: NO es entrada
      porque contiene la respuesta (prestado = prestamos > 0).
    - ejemplares: copias que tiene la biblioteca del libro (entrada 1).
    - anio_edicion: año de la edición MÁS RECIENTE entre sus ejemplares (entrada 2).
      Los ejemplares de un mismo libro traen años distintos, por eso se resume en un valor.
    """
    anio = (t["ejemplar"].groupby("id_libro")["anio_edicion"].max()
            .rename("anio_edicion").reset_index())
    out = libros.merge(anio, on="id_libro", how="left")
    out["prestamos"] = out["prestamos"].astype(int)
    out["prestado"] = (out["prestamos"] > 0).astype(int)
    out = out[["id_libro", "ejemplares", "anio_edicion", "prestamos", "prestado"]]
    out.to_csv(RUTA_LIBROS, index=False)
    print(f"\nDataset por libro: {out.shape[0]} filas x {out.shape[1]} columnas -> {RUTA_LIBROS}")
    print(f"Libros prestados al menos una vez: {out['prestado'].mean():.1%} "
          f"({int(out['prestado'].sum())} de {len(out)})")
    return out


def construir_demanda_generos(t):
    """Préstamos (a nivel ejemplar) por género."""
    x = (t["prestamo_ejemplar"]
         .merge(t["ejemplar"][["id_ejemplar", "id_libro"]], on="id_ejemplar")
         .merge(t["libro_genero"], on="id_libro")
         .merge(t["genero"].rename(columns={"nombre": "genero"}), on="id_genero"))
    return x["genero"].value_counts().rename_axis("genero").reset_index(name="prestamos")



# 4. HALLAZGOS

def imprimir_hallazgos(t, df, libros, generos):
    dev = df[df["devuelto"]]
    tarde = dev[dev["dias_retraso"] > 0]
    sin_cerrar = df[~df["devuelto"]]

    print("\n" + "=" * 70)
    print("HALLAZGOS")
    print("=" * 70)

    print(f"\n[1] Estado de los préstamos: {len(df)} en total, "
          f"{len(dev)} devueltos y {len(sin_cerrar)} sin devolución "
          f"({len(sin_cerrar) / len(df):.1%}).")
    print(f"    Fecha de corte (última fecha de los datos): "
          f"{df.attrs['fecha_corte'].date()}")
    mas_1a = int((sin_cerrar["dias_sin_cerrar"] > 365).sum())
    print(f"    Sin cerrar hace más de un año: {mas_1a} "
          f"(mediana: {sin_cerrar['dias_sin_cerrar'].median():.0f} días).")

    print(f"\n[2] Devoluciones tardías: {len(tarde)} de {len(dev)} "
          f"({len(tarde) / len(dev):.1%}).")
    tarifa = (tarde["multa"] / tarde["dias_retraso"]).round(2).unique()
    print(f"    Tarifa de multa por día de retraso (valores distintos): {[float(x) for x in tarifa]}")
    print(f"    Multa total registrada: ${df['multa'].sum():,.0f}; "
          f"multa en préstamos sin cerrar: ${sin_cerrar['multa'].sum():,.0f}")

    print("\n[3] Devoluciones tardías según el plazo del préstamo:")
    por_plazo = dev.groupby("plazo_dias").agg(
        prestamos=("con_retraso", "size"),
        pct_retraso=("con_retraso", lambda s: s.astype(float).mean() * 100),
        dias_promedio=("dias_prestamo", "mean")).round(1)
    print(por_plazo.to_string())
    corr = dev[["plazo_dias", "dias_prestamo"]].astype(float).corr().iloc[0, 1]
    print(f"    Correlación plazo vs días que realmente se queda el libro: {corr:.3f}")
    p_plazo = chi2_contingency(pd.crosstab(dev["plazo_dias"], dev["con_retraso"]))[1]
    print(f"    Prueba chi-cuadrado (plazo vs retraso): p = {p_plazo:.2e}")

    print("\n[4] Devoluciones tardías según el rol del usuario:")
    por_rol = dev.groupby("rol").agg(
        prestamos=("con_retraso", "size"),
        pct_retraso=("con_retraso", lambda s: s.astype(float).mean() * 100)).round(1)
    print(por_rol.sort_values("pct_retraso", ascending=False).to_string())
    p_rol = chi2_contingency(pd.crosstab(dev["rol"], dev["con_retraso"]))[1]
    print(f"    Prueba chi-cuadrado (rol vs retraso): p = {p_rol:.3f}")

    n_antes = int(df["prestamo_antes_registro"].sum())
    print(f"\n[5] Calidad de datos: {n_antes} préstamos ({n_antes / len(df):.1%}) "
          "tienen fecha anterior al registro del usuario.")

    top5 = generos.head(5)
    print(f"\n[6] Los 5 géneros más prestados concentran "
          f"{top5['prestamos'].sum() / generos['prestamos'].sum():.1%} de los préstamos:")
    print(top5.to_string(index=False))

    nunca = (libros["prestamos"] == 0).mean()
    ej_total = len(t["ejemplar"])
    ej_usados = t["prestamo_ejemplar"]["id_ejemplar"].nunique()
    print(f"\n[7] Demanda: {nunca:.1%} de los libros nunca se ha prestado; "
          f"{ej_usados} de {ej_total} ejemplares ({ej_usados / ej_total:.1%}) "
          "se han prestado al menos una vez.")
    print(f"    Correlación ejemplares vs préstamos por libro: "
          f"{libros['ejemplares'].corr(libros['prestamos']):.3f}")

    print(f"\n[8] Columna constante: tipo_material tiene "
          f"{t['ejemplar']['tipo_material'].nunique()} valor(es) "
          f"({t['ejemplar']['tipo_material'].unique()[0]}); no aporta información.")



# 5. VISUALIZACIONES

def guardar(nombre):
    plt.tight_layout()
    plt.savefig(GRAFICOS / nombre, dpi=150)
    plt.close()


def visualizar(df, libros, generos):
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    dev = df[df["devuelto"]].copy()

    # 1. Histograma + densidad: cuántos días se queda el libro
    plt.figure(figsize=(8, 5))
    sns.histplot(data=dev, x="dias_prestamo", bins=20, kde=True, color="steelblue")
    plt.axvline(dev["dias_prestamo"].median(), color="crimson", linestyle="--",
                label=f"Mediana: {dev['dias_prestamo'].median():.0f} días")
    plt.title("Duración real de los préstamos devueltos")
    plt.xlabel("Días entre préstamo y devolución")
    plt.ylabel("Número de préstamos")
    plt.legend()
    guardar("biblioteca_hist_dias_prestamo.png")

    # 2. Barras: % de devoluciones tardías según el plazo concedido
    resumen = (dev.groupby("plazo_dias")["con_retraso"]
               .apply(lambda s: s.astype(float).mean() * 100)
               .rename("pct_retraso").reset_index())
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=resumen, x="plazo_dias", y="pct_retraso",
                     hue="plazo_dias", palette="viridis", legend=False)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f%%", padding=3)
    ax.margins(y=0.12)
    plt.title("Devoluciones tardías según el plazo del préstamo")
    plt.xlabel("Plazo concedido (días)")
    plt.ylabel("% devuelto con retraso")
    guardar("biblioteca_barras_retraso_por_plazo.png")

    # 3. Boxplot: días de retraso por rol de usuario
    orden = dev.groupby("rol")["dias_retraso"].median().sort_values().index
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(data=dev, x="rol", y="dias_retraso", order=orden,
                hue="rol", hue_order=orden, palette="Set2", legend=False)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Días de retraso por rol (negativo = devuelto antes del límite)")
    plt.xlabel("Rol del usuario")
    plt.ylabel("Días de retraso")
    guardar("biblioteca_boxplot_retraso_por_rol.png")

    # 4. Regresión: ejemplares por libro vs veces prestado
    plt.figure(figsize=(8, 5.5))
    sns.regplot(data=libros, x="ejemplares", y="prestamos",
                x_jitter=0.15, y_jitter=0.15,
                scatter_kws={"alpha": 0.3, "s": 20}, line_kws={"color": "crimson"})
    plt.title("Ejemplares disponibles vs veces que se presta un libro")
    plt.xlabel("Número de ejemplares del libro")
    plt.ylabel("Número de préstamos")
    guardar("biblioteca_regresion_ejemplares_prestamos.png")

    # 5. Barras horizontales: géneros más prestados
    top = generos.head(10)
    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(data=top, y="genero", x="prestamos", hue="genero",
                     palette="mako", legend=False)
    for c in ax.containers:
        ax.bar_label(c, padding=3)
    ax.margins(x=0.1)
    plt.title("Los 10 géneros más prestados")
    plt.xlabel("Número de préstamos")
    plt.ylabel("")
    guardar("biblioteca_barras_generos.png")

    print(f"\nGráficos guardados en {GRAFICOS}")



# 6. PREPARACIÓN PARA MACHINE LEARNING

def preparar_ml(df):
    """
    Dataset para predecir si un préstamo se devolverá tarde.
    - Solo préstamos devueltos: en los demás no se conoce el resultado.
    - One-Hot para 'rol' (nominal, sin orden).
    - Se excluyen 'multa' y 'dias_prestamo': contienen la respuesta
      (multa = tarifa x días de retraso; retraso = días - plazo).
    - Se excluye 'tiene_observacion': podría escribirse al cerrar el
      préstamo y filtrar información del resultado.
    """
    dev = df[df["devuelto"]].copy()
    columnas = ["plazo_dias", "n_ejemplares", "limite_prestamos",
                "mes_prestamo", "dia_semana", "rol",
                "dias_retraso", "con_retraso"]
    ml = pd.get_dummies(dev[columnas], columns=["rol"], prefix="rol", dtype=int)
    ml[["dias_retraso", "con_retraso"]] = ml[["dias_retraso", "con_retraso"]].astype(int)

    SALIDA_ML.parent.mkdir(parents=True, exist_ok=True)
    ml.to_csv(SALIDA_ML, index=False)
    ml.to_csv(RUTA_LIMPIO, index=False)
    print(f"\nDataset ML: {ml.shape[0]} filas x {ml.shape[1]} columnas -> {SALIDA_ML}")
    print(f"Dataset limpio (Avance 2) exportado también a {RUTA_LIMPIO}")
    print(ml.head().to_string())
    return ml


def main():
    print("=" * 70)
    print("PREPARACIÓN Y ANÁLISIS DE DATOS - BIBLIOTECA COTECNOVA")
    print("=" * 70)
    tablas = cargar_tablas()
    explorar_todo(tablas)
    revisar_duplicados(tablas)
    revisar_categorias(tablas)
    prestamo_original = tablas["prestamo"].copy()   # para la tabla antes / después

    print("\n" + "=" * 70)
    print("LIMPIEZA")
    print("=" * 70)
    tablas = limpiar_tablas(tablas)

    df = construir_prestamos(tablas)
    revisar_imposibles_y_atipicos(df)
    libros = construir_demanda_libros(tablas)
    generos = construir_demanda_generos(tablas)

    imprimir_hallazgos(tablas, df, libros, generos)
    visualizar(df, libros, generos)
    ml = preparar_ml(df)
    exportar_libros_demanda(tablas, libros)
    tabla_antes_despues(prestamo_original, ml)
    print("\nProceso completado exitosamente.")


if __name__ == "__main__":
    main()