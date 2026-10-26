# Chatbot de IA para la gestión de inventario de la biblioteca de Cotecnova

## 1. Definición del proyecto

### 1.1 Planteamiento del problema

La biblioteca de Cotecnova, en Cartago (Valle del Cauca), cuenta con una mala gestión de su inventario, lo que lleva a que no se sepa con certeza qué ejemplares están disponibles, en qué zona se encuentran ni cuáles se están perdiendo o quedando sin devolver. Esto afecta a los dos lados:

- **Los usuarios** (estudiantes, docentes y externos) no pueden saber por su cuenta si un libro está disponible, dónde está ni qué libros hay para su carrera, y terminan dependiendo de preguntarle al bibliotecario.
- **El bibliotecario** no tiene datos para administrar la ejemplares: no sabe cuáles son los libros más solicitados, cuáles se quedaron sin copias, cuáles nunca se prestan ni qué ejemplares llevan mucho tiempo sin devolverse. Por eso decide qué comprar sin ningún respaldo.

El problema de fondo es que la base de datos no guarda la disponibilidad de los ejemplares. Solo registra los préstamos, y el estado de cada copia hay que deducirlo del historial. Por eso se necesita una herramienta que consulte ese historial y responda estas preguntas de forma inmediata: un chatbot de IA.

### 1.2 Objetivos

**Objetivo general.** Desarrollar un chatbot de inteligencia artificial que permita consultar el inventario de la biblioteca. El bibliotecario podrá obtener información para administrar la ejemplares, y los demás usuarios podrán consultar si un libro está disponible y qué libros se relacionan con su carrera.

**Objetivos específicos de este avance.**

- Cargar las tablas de la biblioteca en Python y estructurarlas como listas de diccionarios.
- Determinar el estado y la ubicación de cada uno de los 2 641 ejemplares a partir de su historial de préstamos.
- Analizar con NumPy y Matplotlib la relación entre el stock de cada libro y el número de préstamos.
- Identificar hallazgos que sirvan de base para las respuestas del chatbot y para decisiones de adquisición.

El chatbot funcionará como sistema de consulta y de recomendación. Su construcción corresponde a los siguientes cortes.

### 1.3 Datos

El proyecto de grado se desarrolla desde tercer semestre, por lo que el grupo ya contaba con el modelo entidad-relación de la biblioteca. Como no se tiene acceso a la información real de la biblioteca de la universidad, los archivos CSV fueron elaborados por el grupo a partir de ese modelo, con datos similares a los que la biblioteca maneja normalmente. Por lo tanto, **los datos son simulados y no corresponden a registros reales**. Los resultados sirven para validar el método y las funciones, y no permiten sacar conclusiones sobre la biblioteca de Cotecnova.

Se utilizan 11 archivos CSV ubicados en la carpeta `data/`:

| Tabla | Filas | Columnas | Contenido |
|---|---:|---:|---|
| `libro` | 1 200 | 7 | Título, ISBN, editorial y stock |
| `ejemplar` | 2 641 | 18 | Copias físicas: libro, ubicación, valor, código de barras |
| `ubicacion` | 60 | 2 | Zona, estante y nivel (6 zonas de 10 ubicaciones) |
| `prestamo` | 900 | 7 | Usuario, fecha de préstamo, fecha límite, fecha de devolución y multa |
| `prestamo_ejemplar` | 1 258 | 3 | Ejemplares de cada préstamo y su estado (`Devuelto`, `Prestado`, `Perdido`, `Danado`) |
| `usuario` | 400 | 12 | Usuarios y su rol (Estudiante, Docente, Bibliotecario, Externo, Administrativo) |
| `genero` | 30 | 2 | Géneros y áreas académicas |
| `libro_genero` | 1 257 | 2 | Relación entre libros y géneros |
| `autor` | 350 | 7 | Autores |
| `autor_libro` | 1 644 | 2 | Relación entre autores y libros |
| `editorial` | 40 | 2 | Editoriales |

Los préstamos abarcan del 2 de enero de 2022 al 28 de diciembre de 2025.

## 2. Estructura de datos

Los archivos se leen con el módulo `csv` de Python, sin pandas. Cada tabla se representa como una lista de diccionarios, con un diccionario por fila, y la función `cargar_todo()` reúne todas las tablas en un diccionario general. Por ejemplo, `datos["libro"][0]["titulo"]` devuelve el título del primer libro.

Todos los valores se leen como texto. Los identificadores se manejan como cadenas, y el stock y la multa se convierten a número cuando se requiere. Las fechas están en formato `AAAA-MM-DD`, por lo que al compararlas como texto se obtiene el orden cronológico.

### Funciones del proyecto

El código está en `src/`, en cuatro archivos. `cargar_datos.py` lee los CSV, `analisis_chatbot.py` calcula la disponibilidad de los libros, `eda_proyecto.py` hace el análisis exploratorio y `main.py` ejecuta el primer análisis. El análisis exploratorio se ejecuta por separado.

#### `cargar_datos.py`: lee los archivos

| Función | Qué hace |
|---|---|
| `leer_datos(nombre_archivo)` | Lee un archivo CSV y devuelve sus filas como una lista de diccionarios. Si el archivo no existe, muestra un aviso y el programa sigue funcionando. |
| `cargar_todo()` | Lee los 11 archivos con `leer_datos` y los junta en un solo diccionario. Por ejemplo, `datos["libro"]` es la tabla de libros. |
| `mostrar_resumen(datos)` | Muestra cuántos registros y qué columnas tiene cada tabla. |

#### `analisis_chatbot.py`: disponibilidad y ubicación

La base de datos no dice si un ejemplar está disponible. Para saberlo, este archivo revisa el último préstamo de cada ejemplar.

| Función | Qué hace |
|---|---|
| `estado_actual_ejemplares` | Busca el último préstamo de cada ejemplar y devuelve su estado (`Devuelto`, `Prestado`, `Perdido` o `Danado`). |
| `estado_de_cada_ejemplar` | Asigna un estado a los 2 641 ejemplares. Si nunca se prestó o ya se devolvió, queda como `Disponible`. |
| `contar_estados` | Cuenta cuántos ejemplares hay en cada estado. |
| `libros_sin_disponibilidad` | Devuelve la lista de libros que no tienen ninguna copia disponible. |
| `no_disponibles_por_zona` | Cuenta cuántos ejemplares no disponibles hay en cada zona de la biblioteca. |
| `top_libros_prestados` | Devuelve los libros más prestados (por defecto, los 10 primeros). |
| `grafico_estados`, `grafico_zonas`, `grafico_top_libros` | Dibujan gráficos de barras y los guardan en `outputs/`. |
| `main` | Ejecuta las funciones anteriores e imprime el informe. |

**Cómo funciona `estado_actual_ejemplares`.** Es la función más importante del proyecto.

1. Recorre los préstamos de cada ejemplar.
2. Se queda con el más reciente. Si dos préstamos tienen la misma fecha, gana el que tiene el id mayor.
3. El estado de ese préstamo es el estado actual del ejemplar.

Por ejemplo, si un ejemplar se prestó en enero y se devolvió, y en marzo se prestó otra vez sin devolverse, su estado actual es `Prestado`. Las fechas tienen el formato `AAAA-MM-DD`, por eso se pueden comparar como texto y el orden sale correcto.

**Cómo funciona `estado_de_cada_ejemplar`.** Toma el resultado anterior y lo completa para todos los ejemplares. Los que no aparecen (nunca se han prestado) quedan como `Disponible`, igual que los `Devuelto`. Los demás estados se dejan como `Prestado`, `Perdido` y `Dañado`.

**Cómo funciona `no_disponibles_por_zona`.** La ubicación de cada ejemplar es un texto como `"Zona A - Estante 1 - Nivel 1"`. La función se queda con la primera parte (`"Zona A"`) y cuenta los ejemplares que no están disponibles en cada zona.

#### `eda_proyecto.py`: análisis exploratorio

| Función | Qué hace |
|---|---|
| `construir_stock_prestamos` | Arma una tabla con una fila por libro: su título, cuántos ejemplares tiene (stock) y cuántas veces se ha prestado. |
| `estadisticas_numpy` | Calcula el promedio, el máximo, el mínimo y la desviación estándar de una lista de números con NumPy. |
| `libros_alta_demanda_bajo_stock` | Busca los libros con 3 o más préstamos y 3 o menos ejemplares, que son los que convendría comprar más. Esos dos números se pueden cambiar. |
| `grafico_distribucion` | Dibuja cuántos libros tienen 0, 1, 2 o más préstamos. |
| `grafico_stock_vs_prestamos` | Dibuja el promedio de préstamos según el stock del libro y devuelve esos promedios. |
| `main` | Ejecuta el análisis: imprime las estadísticas y los hallazgos, genera los gráficos y muestra los 5 primeros candidatos a comprar. |

#### `main.py`

| Función | Qué hace |
|---|---|
| `main` | Carga los datos, muestra el resumen de las tablas y ejecuta el análisis de disponibilidad de `analisis_chatbot.py`. |

Este archivo no ejecuta `eda_proyecto.py`. Además, `analisis_chatbot.main()` vuelve a cargar los CSV, por lo que se leen dos veces. Con estos datos no afecta, pero es una mejora pendiente.

## 3. Análisis exploratorio con NumPy

Estadísticas calculadas sobre los 1 200 libros:

| Variable | Promedio | Máximo | Mínimo | Desviación estándar |
|---|---:|---:|---:|---:|
| Stock (ejemplares por libro) | 2,20 | 5 | 1 | 1,19 |
| Préstamos por libro | 1,05 | 7 | 0 | 1,16 |

Estado actual de los 2 641 ejemplares: 2 430 disponibles (92,0 %), 181 prestados (6,9 %), 15 perdidos (0,6 %) y 15 dañados (0,6 %).

## 4. Visualización con Matplotlib

Los gráficos se guardan como imágenes PNG en `outputs/` (los del análisis exploratorio, en `outputs/graficos/`).

![Estado de los ejemplares](outputs/estado_ejemplares.png)

*Figura 1. Estado actual de los ejemplares. La mayor parte del inventario está disponible.*

![Ejemplares no disponibles por zona](outputs/no_disponibles_por_zona.png)

*Figura 2. Ejemplares no disponibles por zona. La Zona F registra 53 y la Zona B, 21.*

![Stock vs préstamos](outputs/graficos/stock_vs_prestamos.png)

*Figura 3. Promedio de préstamos según el stock del libro: 0,45 con un ejemplar y 2,29 con cinco.*

Además se generan `top_libros_prestados.png`, con los 10 libros más prestados, y `graficos/distribucion_prestamos.png`, con la distribución de préstamos por libro.

## 5. Hallazgos

1. **Una parte del catálogo no se presta.** 479 de los 1 200 libros (39,9 %) no registran ningún préstamo, mientras que los más solicitados llegan a 6 o 7 préstamos. El libro con más préstamos es *Realidad Aumentada: teoria y practica* (7).
2. **El stock se relaciona con los préstamos.** La correlación entre ambas variables es 0,49. Esto se explica en buena parte porque un libro con más ejemplares tiene más oportunidades de ser prestado: al dividir el promedio de préstamos entre el stock, el resultado ronda 0,5 en todos los niveles, es decir, cada ejemplar se presta con una frecuencia similar sin importar cuántas copias tenga su título.
3. **42 libros no tienen ningún ejemplar disponible.** De ellos, 31 cuentan con un solo ejemplar, por lo que su falta de disponibilidad se debe más al poco stock que a una demanda alta.
4. **Los ejemplares no disponibles se distribuyen de forma desigual entre zonas.** La Zona F concentra 53 y la Zona B, 21.
5. **Hay 65 libros candidatos a adquirir más ejemplares**, al tener 3 o más préstamos y 3 o menos ejemplares (`libros_alta_demanda_bajo_stock`).

### Consideraciones

- Los datos son simulados, por lo que estos hallazgos validan el funcionamiento de las funciones y no describen la biblioteca real.
- La disponibilidad se deduce del último préstamo de cada ejemplar. En los datos, 189 préstamos (21 %) no tienen fecha de devolución y son los que dejan un ejemplar como `Prestado`. Con datos reales sería necesario verificar con la biblioteca cómo se registran las devoluciones, las pérdidas y los daños.
- El estado `Danado` aparece sin tilde en los archivos CSV y el código lo muestra como `Dañado`.
- La desviación estándar es la poblacional (`np.std` con `ddof=0`).

## 6. Relación con el chatbot

El chatbot aún no está implementado. La siguiente tabla muestra las consultas que atenderá y la función del proyecto que aporta la información:

| Usuario | Consulta | Función |
|---|---|---|
| Estudiante, docente o externo | ¿Está disponible este libro? | `estado_de_cada_ejemplar` |
| Estudiante, docente o externo | ¿Por qué no se encuentra? | `libros_sin_disponibilidad` |
| Estudiante, docente o externo | ¿Qué libros hay para mi carrera? | Pendiente (usa `libro_genero`; ver próximos pasos) |
| Bibliotecario | ¿Cuáles son los libros más prestados? | `top_libros_prestados` |
| Bibliotecario | ¿Qué zona tiene más ejemplares fuera de la estantería? | `no_disponibles_por_zona` |
| Bibliotecario | ¿Qué libros conviene adquirir? | `libros_alta_demanda_bajo_stock` |

## 7. Ejecución

Con Docker:

```bash
docker compose up -d --build
docker compose exec python python src/main.py
docker compose exec python python src/eda_proyecto.py
```

Sin Docker:

```bash
pip install -r requirements.txt
python src/main.py
python src/eda_proyecto.py
```

`main.py` muestra el resumen de las tablas y el análisis de disponibilidad. `eda_proyecto.py` se ejecuta por separado y muestra las estadísticas y los hallazgos.

### Estructura del repositorio

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/            # 11 archivos CSV
├── notebooks/       # cuadernos Jupyter
├── outputs/         # gráficos generados
└── src/
    ├── main.py
    ├── cargar_datos.py
    ├── analisis_chatbot.py
    └── eda_proyecto.py
```

Herramientas utilizadas: Python 3.12, NumPy, Matplotlib y Docker.

## 8. Desarrollo del proyecto

1. Se configuró el entorno con la estructura propuesta por el profesor: Docker, `requirements.txt` y las carpetas `data/`, `src/`, `notebooks/` y `outputs/`.
2. Se migró el proyecto a Visual Studio Code y se desarrolló `cargar_datos.py` para leer los 11 CSV.
3. Se desarrolló `analisis_chatbot.py` con las funciones de disponibilidad y ubicación.
4. Se desarrolló `eda_proyecto.py` con el análisis exploratorio.
5. Se creó `main.py` como punto de entrada.

## 9. Próximos pasos

Para los siguientes cortes se plantea:

1. Guardar el informe en `outputs/resumen.json` para que el chatbot lo consulte sin recalcular el análisis. Tecnología: módulo `json`.
2. Convertir `buscar_libro` en una función del proyecto que entregue disponibilidad y ubicación, sin distinguir tildes ni mayúsculas. Tecnología: `unicodedata`.
3. Medir la demanda como préstamos por ejemplar, limitada a los últimos 12 meses. Tecnología: `datetime` y `pandas`.
4. Analizar los préstamos sin cerrar y las multas: lista de ejemplares pendientes hace más de un año y recordatorios de devolución. Tecnología: `datetime` y `APScheduler`.
5. Recomendar libros por género, autor o carrera con las tablas `libro_genero` y `autor_libro`. Tecnología: `scikit-learn`.
6. Elaborar un notebook en `notebooks/` para los análisis en que `pandas` resulta útil, como `groupby` y series de tiempo. Tecnología: `Jupyter`.
7. Conectar las funciones con el chatbot y validar con la biblioteca el registro de devoluciones, pérdidas y daños. Tecnología: `FastAPI` y un modelo de lenguaje con llamadas a herramientas.


# Preparación de Datos



En este script se hace un análisis de datos con los csv trabajados previamente, imprime el EDA de las tablas, construye el un tbla con un registro por préstamo, se generan 5 gráficos y guarda el dataset generado.


### Valores nulos encontrados y cómo se manejaron

| Tabla.columna | Nulos | Qué significa | Decisión |
|---|---|---|---|
| `prestamo.fecha_devolucion` | 189 de 900 (21.0 %) | El préstamo todavía no se ha cerrado | Inventar una fecha de devolución falsearía todos los cálculos de retraso. Se crea la bandera `devuelto` y las columnas de retraso quedan vacías en esas filas. Para el dataset de Machine Learning se **eliminan esas filas** (se usan los 711 préstamos devueltos), porque en los otros no se conoce el resultado. |
| `prestamo.observaciones` | 627 de 900 (69.7 %) | El bibliotecario no escribió ninguna nota | Es texto libre y opcional. Se convierte en la columna binaria `tiene_observacion`. |
| `autor.segundo_nombre` | 123 de 350 (35.1 %) | El autor no tiene segundo nombre | Campo opcional: se rellena con cadena vacía. |
| `usuario.segundo_nombre` | 143 de 400 (35.8 %) | Igual que en autor | Se rellena con cadena vacía. |

Las demás tablas no tienen nulos. 

### Variables categóricas codificadas

- **`rol`** (Administrativo, Bibliotecario, Docente, Estudiante, Externo) se codificó con `pd.get_dummies()` en 5 columnas `rol_*` con valores 0/1. Es una variable nominal (no hay un rol "mayor" que otro), por eso se usa One-Hot y no números del 1 al 5.
- **Columnas descartadas por no servir al modelo:** los identificadores, las fechas originales (sus partes útiles pasan a `mes_prestamo` y `dia_semana`), `tipo_material` (todos los ejemplares son `Fisico`, así que no aporta información) y los datos personales del usuario (`correo`, `telefono`, `direccion`).
- **`clave` no se usa y no debería estar en los CSV:** la tabla `usuario` guarda las contraseñas en texto plano. Aunque sean datos de prueba, en el sistema real deben guardarse con hash.
- **Columnas excluidas para evitar fuga de datos:** `multa` y `dias_prestamo` contienen la respuesta (la multa es exactamente $500 por cada día de retraso, y el retraso es los días reales menos el plazo). `tiene_observacion` también se deja fuera porque la nota podría escribirse al cerrar el préstamo.

El dataset resultante tiene **711 filas y 12 columnas**: 5 variables de entrada numéricas (`plazo_dias`, `n_ejemplares`, `limite_prestamos`, `mes_prestamo`, `dia_semana`), las 5 columnas de `rol` y 2 variables objetivo (`con_retraso` y `dias_retraso`).

### Columnas derivadas y por qué

| Columna | Cómo se calcula | Para qué sirve |
|---|---|---|
| **`dias_retraso`** (principal) | `fecha_devolucion − fecha_limite` | Mide cuánto se pasó el usuario del plazo. Es la base de la gestión de multas y de la futura predicción. Negativo significa que devolvió antes. |
| `con_retraso` | 1 si `dias_retraso > 0`, 0 si no | Variable objetivo para clasificar. |
| `dias_prestamo` | `fecha_devolucion − fecha_prestamo` | Cuánto tiempo se queda realmente el libro. |
| `plazo_dias` | `fecha_limite − fecha_prestamo` | Plazo concedido (7, 14, 21 o 30 días). |
| `devuelto` | `fecha_devolucion` no nula | Separa préstamos cerrados de los pendientes. |
| `dias_sin_cerrar` | Fecha de corte (26/01/2026, la última de los datos) menos `fecha_prestamo`, solo en préstamos sin devolver | Detecta préstamos abandonados. |
| `tiene_observacion` | 1 si hay nota | Convierte el texto libre en una variable utilizable. |
| `n_ejemplares` | Ejemplares incluidos en el préstamo | Tamaño del préstamo (1, 2 o 3). |
| `prestamo_antes_registro` | `fecha_prestamo < fecha_registro` del usuario | Bandera de calidad de datos. |

### Hallazgos del análisis exploratorio


El **plazo del préstamo es uno de los factores que más influye en los retrasos**. Los préstamos de 7 días son los que más se devuelven tarde: el 80.1 % de estos préstamos presentan retraso. En los préstamos de 14 días, el 60.5 % llega tarde; en los de 21 días, el 41.1 %; y en los de 30 días, el 22.1 %.

Esto pasa porque, aunque el plazo sea diferente, las personas normalmente se quedan con el libro casi el mismo tiempo: entre 19 y 21 días en promedio. La correlación entre el plazo y los días que realmente tuvieron el libro es de **−0.025**, lo que significa que prácticamente no hay relación entre estas dos cosas. La prueba realizada también confirma que esta diferencia no se debe simplemente al azar. Por eso, para la biblioteca, un préstamo de 7 días termina en multa aproximadamente **8 de cada 10 veces**.

En general, el **51.6 % de las devoluciones fueron hechas tarde**, es decir, 367 de 711 préstamos. La multa se calcula de una manera sencilla: se cobran **$500 por cada día de retraso**. En total, aparecen registradas multas por **$2.488.000**.

Sin embargo, hay un problema: existen **189 préstamos que todavía aparecen como no devueltos**, pero tienen una multa de $0, aunque ya están vencidos. Esto significa que mientras el libro siga sin devolverse, la multa no aumenta en los datos. Por eso, la cantidad real de dinero que se debería cobrar puede ser mayor a la que aparece registrada.

De esos 189 préstamos que todavía están abiertos, **135 tienen más de un año**. El tiempo medio de estos préstamos es de 758 días. Además, entre los libros de estos préstamos hay **25 marcados como "Perdido" y 20 como "Dañado"**. Estos casos deberían revisarse y posiblemente considerarse como libros perdidos o dañados, en lugar de tratarlos simplemente como préstamos que todavía están activos.

También se encontró un **problema con las fechas de los datos**. Hay 244 préstamos, que representan el **27.1 %**, donde la fecha del préstamo aparece antes de la fecha en que el usuario fue registrado. Es decir, según el sistema, una persona pidió un libro antes de estar registrada. En una base de datos real esto no debería pasar y sería necesario controlar las fechas cuando se registran los datos.

La mayor parte de los préstamos se concentra en **carreras académicas**. Los cinco grupos con más préstamos son:

- Administración de Empresas
- Contabilidad
- Ingeniería de Sistemas
- Diseño e Integración Multimedia
- Agropecuaria

Entre todos representan el **77.1 % de los préstamos**. En cambio, los géneros relacionados con literatura tienen muchos menos préstamos, con 21 o menos. Esto indica que el chatbot debería facilitar especialmente la búsqueda de libros por carrera.

También se encontró que **muchos libros casi no se utilizan**. El 39.9 % de los libros, es decir, 479 de 1.200, nunca ha sido prestado. Además, solo 1.018 de los 2.641 ejemplares han sido prestados alguna vez, lo que representa el 38.5 %.

En promedio, por cada ejemplar adicional se generan aproximadamente **0.48 préstamos**, un resultado muy parecido al promedio general de 0.476 préstamos por ejemplar. Esto indica que tener más ejemplares normalmente genera más préstamos, pero no hay un libro que sobresalga demasiado sobre los demás. El libro con más préstamos solo tiene 7.

Por último, **el tipo de usuario no parece ser un factor claro para explicar los retrasos**. Por ejemplo, el 61.0 % de los usuarios administrativos devuelve tarde, mientras que en los bibliotecarios es el 43.8 %. Aunque existe una diferencia, los resultados no son suficientes para decir que el tipo de usuario sea la causa de los retrasos.

También se encontró que los días de préstamo están distribuidos de manera muy pareja: hay préstamos de entre 1 y 40 días y no existe una duración que sea mucho más común que las demás.

Finalmente, hay que tener en cuenta que los datos parecen haber sido **creados para el proyecto y no necesariamente representan el comportamiento real de una biblioteca**, ya que algunas observaciones parecen ser datos de relleno. Por eso, estos resultados sirven para encontrar problemas y tendencias, pero sería importante compararlos después con datos reales de la biblioteca.
### Visualizaciones

| Gráfico | Tipo Seaborn | Qué muestra |
|---|---|---|
| ![Duración de los préstamos](outputs/graficos/biblioteca_hist_dias_prestamo.png) | `histplot` con densidad | Distribución plana de 1 a 40 días; mediana de 19 días. |
| ![Retraso por plazo](outputs/graficos/biblioteca_barras_retraso_por_plazo.png) | `barplot` | % de devoluciones tardías según el plazo (80 % → 22 %). |
| ![Retraso por rol](outputs/graficos/biblioteca_boxplot_retraso_por_rol.png) | `boxplot` | Días de retraso por rol; cajas muy superpuestas. |
| ![Ejemplares vs préstamos](outputs/graficos/biblioteca_regresion_ejemplares_prestamos.png) | `regplot` | Relación positiva (r = 0.49) entre ejemplares y préstamos por libro. |
| ![Géneros más prestados](outputs/graficos/biblioteca_barras_generos.png) | `barplot` horizontal | Los 10 géneros con más préstamos. |

## Neuronas para las compuertas OR y NOT

### ¿Qué se hizo?

Con la misma neurona que se usó en clase para la compuerta AND, se entrenaron dos compuertas más en el archivo.

- **OR:** da 1 si al menos una de las dos entradas es 1.
- **NOT:** tiene una sola entrada y da el valor contrario (0 → 1 y 1 → 0).

En ambos casos se usó la tasa de aprendizaje de 0.1 y se empezó con los mismos valores que en la clase (`w = 0.1` y `b = 0.0`). El código recorre las filas de la tabla, calcula el error de cada una (`error = y_real - y_pred`) y corrige los pesos y el bias, igual que en el ejercicio de clase. Al final se verificó cada fila y se guardó el gráfico del error en `outputs/graficos/error_or_not.png`.

Además se hizo una **prueba extra con XOR** (`outputs/graficos/error_xor.png`) para poder responder la última pregunta con un resultado real y no solo con teoría.

### Resultados

| Compuerta | Épocas para aprender | Error total por época | Parámetros finales |
|-----------|----------------------|-----------------------|--------------------|
| AND (clase) | 3 | 3, 3, 0 | w₁ = 0.2, w₂ = 0.1, b = -0.2 |
| OR | 2 | 1, 0 | w₁ = 0.1, w₂ = 0.1, b = -0.1 |
| NOT | 4 | 1, 2, 1, 0 | w₁ = -0.1, b = 0.0 |
| XOR (extra) | no aprende | 2, 3, 2, 3, 4, 4, 4, 4, 4, 4 | se queda dando vueltas |

Las pruebas finales de OR y NOT dieron todas las filas correctas.

Una cosa que llama la atención en NOT es que el peso termina **negativo** (w₁ = -0.1). Tiene sentido: en NOT, que la entrada valga 1 debe *apagar* la neurona, y para eso el peso tiene que restar en vez de sumar. Además, el error de NOT sube a 2 en la época 2 antes de bajar: mientras el bias y el peso se ajustan, la neurona se equivoca en las dos filas, hasta que se acomodan.

### Respuestas a las preguntas

**1. ¿Cuántas épocas necesitó cada compuerta para aprender?**

OR necesitó **2 épocas** y NOT necesitó **4 épocas**. Como referencia, AND (con los mismos valores iniciales) necesitó 3.

**2. ¿Fue necesario ajustar la tasa de aprendizaje? ¿Por qué?**

No. Se usó 0.1 en las dos compuertas y ambas llegaron a error 0 en pocas épocas, así que no hubo motivo para cambiarla. En la clase ya se había visto qué pasa en los extremos: con 0.001 la neurona de AND tardó 34 épocas (muy lenta) y con 1.0 aprendió en 6 pero con parámetros enormes (b = -3.0). El valor de 0.1 queda en un punto intermedio que funciona bien para estos problemas tan pequeños.

**3. ¿Qué diferencias se encontraron entre la compuerta AND y la OR?**

Las dos empezaron con los mismos valores, pero terminaron distinto:

- Con `z = 0` la neurona predice 1, así que al arrancar dice "sí" en todas las filas. En OR solo una fila debe dar 0, la (0, 0), y por eso el error de la primera época fue 1. En AND tres filas deben dar 0, y el error fue 3.
- En OR basta con bajar un poco el bias (de 0.0 a -0.1) y los pesos se quedan en 0.1. En AND hubo que bajar más el bias (a -0.2) y además subir `w₁` a 0.2, para que la neurona solo se active cuando las dos entradas valen 1.
- Por eso OR aprendió más rápido (2 épocas contra 3): "activarse con una entrada" exige un umbral más bajo que "activarse solo con las dos".

**4. ¿Por qué la compuerta XOR no puede ser aprendida por una sola neurona?**

La neurona decide con la cuenta `z = w₁·x₁ + w₂·x₂ + b`, y eso equivale a trazar **una sola línea recta** que separa las entradas que dan 1 de las que dan 0. Con AND y OR sí se puede trazar esa recta. Con XOR no: las entradas (0, 1) y (1, 0) deben dar 1, mientras que (0, 0) y (1, 1) deben dar 0, y esos dos grupos quedan cruzados, así que ninguna recta los separa.

Se comprobó en la prueba extra: el error de XOR no baja, sube hasta 4 y se queda ahí. A partir de la época 5 los parámetros dan una vuelta completa y regresan siempre al mismo punto, y la prueba final falla 2 de las 4 filas. Esta limitación del perceptrón simple la señalaron Minsky y Papert en 1969, y se resuelve con **más de una neurona organizadas en capas**, algo que se verá más adelante.


# Avance 2


**Contenido:**  [B. Preparación de datos](#b-preparación-de-datos-con-pandas) · [C. Visualización](#c-visualización-con-seaborn) · [D. Perceptrón](#d-primer-modelo-perceptrón-desde-cero) · [E. Plan final](#e-plan-hacia-el-proyecto-final) · 
## Resumen

| Elemento | Resultado |
|---|---|
| Pregunta que responde el modelo | Dado un préstamo, ¿se devolverá con retraso? |
| Tipo de problema | Clasificación binaria (retraso = 1, a tiempo = 0) |
| Dataset limpio | `data/proyecto_limpio.csv` (711 filas, 15 columnas, 0 nulos) |
| Gráficos de Seaborn | 5, en `outputs/avance2/` |
| Modelo | Perceptrón con sigmoide y pérdida logarítmica (NumPy) |
| Exactitud | 69.5 % en entrenamiento y 68.5 % en prueba. Adivinar siempre la clase más común daría 50.5 % |

**Idea central.** Los préstamos de 7 días terminan con retraso 8 de cada 10 veces, y los de 30 días, 2 de cada 10. El perceptrón aprende esa relación y la usa para estimar la probabilidad de retraso de un préstamo nuevo.

> **Los datos son simulados:** se elaboraron a partir del modelo entidad-relación de la biblioteca. Los resultados validan el método, no describen la biblioteca real de Cotecnova.

---


**Cambios técnicos**

| Qué cambió | Por qué |
|---|---|
| El análisis pasó de listas de diccionarios y NumPy a **Pandas y Seaborn** | Pandas facilita unir tablas |
| Se definió una **variable objetivo** (`con_retraso`) | En el avance 1 el análisis solo describía los datos; para entrenar un modelo hay que decirle qué debe predecir |
| Se construyó una tabla con **un registro por préstamo** | El préstamo es la unidad sobre la que se predice |
| Se descartaron datos personales del usuario (`correo`, `telefono`, `direccion`, `clave`) | No ayudan al modelo y no deben circular |

**El alcance y la fuente de datos no cambiaron:** sigue siendo un chatbot de consulta para la biblioteca, con CSV simulados. Este modelo aporta al chatbot porque, si se sabe qué préstamos tienen riesgo de retraso, el bibliotecario puede avisar a tiempo y reducir los ejemplares sin devolver (hallazgo del primer corte).

---

## B. Preparación de datos con Pandas

Script: `src/preparacion_proyecto.py`. Entrega un dataset limpio, con un registro por préstamo cerrado, listo para el modelo.

### Carga e inspección
Se cargan `prestamo`, `usuario` y `prestamo_ejemplar` con `pd.read_csv` (si falta un archivo, el programa avisa y se detiene). De cada tabla se imprime `shape`, `info()`, nulos por columna y duplicados, para conocer el estado de los datos antes de tocarlos.

### Tabla antes / después

| | Antes (`prestamo.csv`) | Después (`proyecto_limpio.csv`) |
|---|---:|---:|
| Filas | 900 | 711 |
| Columnas | 7 | 15 |
| Valores nulos | 816 | 0 |

- **Filas (900 → 711):** se excluyen los 189 préstamos que todavía no se han devuelto.
- **Columnas (7 → 15):** se unen datos de `usuario` y `prestamo_ejemplar` y se crean columnas nuevas (ver más abajo).
- **Nulos (816 → 0):** eran 189 de `fecha_devolucion` y 627 de `observaciones`; ninguno pasa al dataset final.

### Tratamiento de cada problema

| Problema | Qué se encontró | Decisión y por qué |
|---|---|---|
| Nulos | `fecha_devolucion`: 189 (el préstamo sigue abierto) | **No se inventa una fecha**, porque falsearía el retraso. Se excluyen esas filas: no se conoce el resultado, así que no sirven para enseñarle al modelo. |
| Nulos | `observaciones`: 627 (nota opcional del bibliotecario) | No se usa como entrada: puede escribirse al cerrar el préstamo y daría pistas de la respuesta. |
| Duplicados | 0, revisados por fila completa y por `id_prestamo` | No hay nada que eliminar. |
| Tipos incorrectos | 4 columnas de fecha llegaron como texto | Se convierten a `datetime` con `pd.to_datetime`, necesario para restar fechas. |
| Valores imposibles | Ningún plazo ≤ 0, ninguna devolución anterior al préstamo, ninguna multa negativa | No hay nada que eliminar. |
| Valores imposibles | 244 préstamos son anteriores a la fecha de registro del usuario y 21 superan su límite de ejemplares | Se **marcan con una bandera** (`prestamo_antes_registro`, `excede_limite`) y se conservan: ninguna entrada del modelo depende de ellos y borrarlos eliminaría el 27 % de los datos. |
| Atípicos | `dias_retraso` va de −29 a 33 y ningún valor queda fuera de 1.5 veces el rango intercuartílico | No se elimina nada: los valores son razonables. |
| Categorías inconsistentes | `rol` tiene 5 valores sin variantes de escritura | Se limpian espacios y se codifica (ver abajo). |

### Codificación de la variable categórica
`rol` (Administrativo, Bibliotecario, Docente, Estudiante, Externo) es texto, y el modelo solo entiende números. Se convierte con `pd.get_dummies` en 5 columnas `rol_*` con valores 0 o 1 (una columna por rol, con un 1 en el rol del usuario). Se usa esta técnica y no los números del 1 al 5 porque ningún rol es "mayor" que otro, y un 5 le haría creer al modelo que lo es.

```python
df = pd.get_dummies(df, columns=["rol"], prefix="rol", dtype=int)
```

### Columnas creadas

| Columna | Cómo se calcula |
|---|---|
| `plazo_dias` | `fecha_limite − fecha_prestamo` (7, 14, 21 o 30 días) |
| `dias_retraso` | `fecha_devolucion − fecha_limite` (negativo = devuelto antes de tiempo) |
| `n_ejemplares` | Cuántos libros incluye el préstamo (1, 2 o 3) |
| `mes_prestamo`, `dia_semana` | Partes de la fecha del préstamo |
| `con_retraso` | 1 si `dias_retraso > 0`, 0 si no. Es la **variable objetivo** |

**Archivos generados:** `data/proyecto_limpio.csv` (dataset limpio) y `outputs/avance2/antes_despues.csv` (tabla anterior).

---

## C. Visualización con Seaborn

Los gráficos los genera el mismo script `src/preparacion_proyecto.py`. Cuentan una historia en tres pasos: el plazo se relaciona con el retraso, es la única variable que lo hace, y aun así los datos no se separan limpiamente.

### Gráfico 1. Retraso según el plazo concedido (entrada contra objetivo)
![Retraso por plazo](outputs/avance2/grafico1_retraso_por_plazo.png)

Cada barra es el porcentaje de préstamos devueltos tarde para un plazo: 80 % con 7 días, 61 % con 14, 41 % con 21 y 22 % con 30. Cuanto más corto es el plazo, más retrasos hay. **Decisión que apoya:** usar `plazo_dias` como entrada principal del modelo.

### Gráfico 2. Correlaciones entre las variables de entrada y `con_retraso`
![Correlaciones](outputs/avance2/grafico2_correlaciones.png)

Un número cercano a 1 o −1 indica una relación fuerte, y uno cercano a 0, que no hay relación. Solo `plazo_dias` se relaciona con el retraso (−0.43, relación inversa: más plazo, menos retraso). `n_ejemplares`, `limite_prestamos`, `mes_prestamo` y `dia_semana` están cerca de 0. **Decisión que apoya:** el perceptrón usa `plazo_dias` y `n_ejemplares` como entradas, y se espera que casi toda la información venga del plazo.

### Gráfico 3. Días de retraso según el plazo, con el umbral del objetivo
![Días de retraso por plazo](outputs/avance2/grafico3_dias_retraso_por_plazo.png)

Cada caja resume los días de retraso de un plazo. La línea punteada en 0 es el umbral: por encima, retraso; por debajo, a tiempo. La mediana pasa de 14 días de retraso (plazo de 7) a 10 días de adelanto (plazo de 30), pero las cajas se superponen mucho. **Decisión que apoya:** fijar el umbral de la variable objetivo en 0 días, y anticipar que una sola recta no separará bien las clases.

---

## D. Primer modelo: perceptrón desde cero

### Variable objetivo
`con_retraso` vale 1 si `dias_retraso > 0` y 0 si no. El umbral es 0 días: devolver después de la fecha límite cuenta como retraso. Se creó a partir de la columna numérica `dias_retraso`, que **no se usa como entrada** porque contiene la respuesta: dársela al modelo sería como entregarle el examen resuelto.

### Entradas
Dos variables numéricas: `plazo_dias` (7 a 30) y `n_ejemplares` (1 a 3). Se **estandarizan** con `(x − media) / desviación`, para que ambas queden con media 0 y desviación 1. Sin esto, la variable con números más grandes dominaría el aprendizaje.

Los datos se dividen en **80 % para entrenar** (568 préstamos) y **20 % para probar** (143). La media y la desviación se calculan solo con los de entrenamiento y se aplican también a los de prueba. Así se mide el modelo con préstamos que no vio mientras aprendía.

### Cómo funciona el modelo
Sigue los mismos pasos de la neurona de la Clase 8 (AND), con una mejora: la sigmoide en lugar del escalón.

1. **Suma ponderada:** `z = w1·x1 + w2·x2 + b`. Cada entrada se multiplica por su peso (su importancia) y se suma el sesgo. Da un número cualquiera, sin significado directo.
2. **Sigmoide:** traduce ese número a una **probabilidad entre 0 y 1**. Un `z` muy alto da casi 1 (retraso casi seguro), `z = 0` da 0.5 (no sabe) y un `z` muy bajo da casi 0. Reemplaza al escalón de la Clase 8 por dos razones: dice **qué tan seguro** está el modelo (80 % no es lo mismo que 51 %), y es una curva suave, lo que permite aprender con el descenso del gradiente.
3. **Pérdida logarítmica:** mide qué tan mal van las probabilidades. Castiga más al modelo cuando se equivoca con mucha seguridad. Si dijera 0.5 para todo, la pérdida sería 0.693; esa es la referencia para saber si mejora.
4. **Descenso del gradiente:** en cada época se calcula hacia dónde mover `w1`, `w2` y `b` para que la pérdida baje, usando todos los préstamos a la vez, y se actualizan los pesos. Es la misma idea de la Clase 8 (`w += α · error · x`), ahora con el error medido sobre probabilidades. Se usa una tasa de aprendizaje de 0.1 y 1000 épocas.

```python
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def predecir_probabilidad(X, w, b):
    return sigmoide(X @ w + b)      # calcula z de todos los préstamos en una línea

error = p - y                        # qué tan lejos quedó cada probabilidad de la realidad
grad_w = X.T @ error / n             # promedio sobre todos los préstamos
w = w - tasa * grad_w                # se mueve cada peso en la dirección que baja la pérdida
```

El modelo parte de los mismos valores iniciales de la Clase 8: `w1 = 0.1`, `w2 = 0.1` y `b = 0`. Para clasificar, una probabilidad de 0.5 o más se toma como "con retraso".

### Resultados

| Resultado | Valor |
|---|---|
| Pesos finales `w1` (plazo) / `w2` (ejemplares) / `b` | −0.9859 / −0.0231 / +0.0150 |
| Pérdida inicial → final | 0.7185 → 0.5923 |
| Exactitud en entrenamiento / prueba | 69.5 % / 68.5 % |
| Referencia: predecir siempre la clase más común | 50.5 % |

Dos lecturas rápidas: la pérdida **bajó**, así que el modelo aprendió, y la exactitud en prueba es casi igual a la de entrenamiento, así que no está memorizando los datos que vio.

![Curva de aprendizaje]

**Curva de aprendizaje.** Muestra la pérdida en cada época. Baja rápido y se estabiliza hacia la época 200, siempre por debajo de la línea de adivinar (0.693). No hacen falta muchas más épocas.

![Frontera de decisión]

**Frontera de decisión.** Cada punto es un préstamo (rojo = con retraso, azul = a tiempo). La línea negra es donde el modelo da 50 % de probabilidad: a su izquierda predice retraso y a su derecha, a tiempo. Los puntos forman columnas porque el plazo solo toma 4 valores.

### Interpretación

- **Signo de `w1` (plazo): negativo.** A mayor plazo, menor probabilidad de retraso. Cada 8.5 días más de plazo (una desviación) dejan las probabilidades de retraso en alrededor de un tercio (se multiplican por 0.37). En la práctica, los préstamos de 7 días son los de más riesgo.
- **Signo de `w2` (ejemplares): casi cero.** Cuántos libros se lleve el usuario no cambia la predicción.
- **Frontera.** Con el número de ejemplares promedio, la probabilidad es 50 % cuando el plazo es de unos 18 días. La línea es casi vertical porque solo el plazo pesa.
- **¿Los datos son linealmente separables? No.** En cada plazo hay préstamos de ambas clases mezclados y ninguna recta los separa del todo. El modelo acierta cerca del 69 % porque casi toda la información está en una sola variable.

### Limitaciones

- **Pocas combinaciones de entrada:** solo hay 4 plazos y 3 tamaños de préstamo, así que la frontera tiene poco que separar.
- **Préstamos abiertos excluidos:** los 189 préstamos sin devolver quedaron fuera del entrenamiento. Según el avance 1, 135 llevan más de un año abiertos, así que probablemente son retrasos y la tasa real se subestima.
- **Una sola neurona** solo puede trazar una recta; no captura relaciones más complejas.
- **Datos simulados:** en ellos, los días que el usuario se queda con el libro no dependen del plazo (correlación −0.025). De ahí sale el patrón de 80 % a 22 %, que podría ser distinto con datos reales.

---

## E. Plan hacia el proyecto final

- **Tipo de problema:** clasificación. Se predice si un préstamo se devolverá con retraso o a tiempo.
- **Modelos a comparar en la Clase 12 (9 de noviembre), con Scikit-Learn:** regresión logística (la versión profesional de lo que hizo el perceptrón), árbol de decisión y bosque aleatorio.
- **Métrica para elegir el mejor:** el **recall** de la clase "con retraso", acompañado del F1. Es peor no avisar de un retraso que avisar de más, y el recall mide cuántos retrasos reales se detectan. El F1 evita ignorar la precisión (que no se avise de retrasos que no ocurren). La exactitud sola no basta.
- **Demostración final:** predecir la probabilidad de retraso de préstamos nuevos, variando el plazo y el rol del usuario, y mostrar cómo el bibliotecario usaría ese aviso a través del chatbot.

**Cronograma hasta la sustentación final**

Las clases son los lunes. Los lunes 2 y 16 de noviembre son festivos, por lo que esas semanas se trabaja en grupo, sin clase.

| Semana | Fechas | Actividad | Resultado esperado |
|---|---|---|---|
| 1 | 26 oct – 1 nov | Clase del lunes 26. Revisar la retroalimentación del Avance 2 y corregir el README. Estudiar herramientas del Scikit-Learn `train_test_split`, `LogisticRegression` y las métricas (recall, F1, matriz de confusión) | Lista de ajustes y pendientes del grupo |
| 2 | 2 – 8 nov (lunes 2 festivo) | Dividir los datos en entrenamiento y prueba con Scikit-Learn y entrenar la regresión logística como primer modelo. Calcular recall, F1 y matriz de confusión | Script `modelos_sklearn.py` con la regresión logística funcionando |
| 3 | 9 – 15 nov | **Clase 12 (lunes 9):** entrenar el árbol de decisión y el bosque aleatorio y compararlos con la regresión logística. Elegir el mejor por recall, acompañado del F1 | Tabla comparativa de los 3 modelos y el modelo elegido, justificado |
| 4 | 16 – 22 nov (lunes 16 festivo) | Mejorar el modelo elegido (más variables como el rol, ajuste del umbral). Definir los préstamos nuevos de la demostración y crear la función que predice la probabilidad de retraso | Función `predecir_retraso()` y lista de casos de demostración |
| 5 | 23 – 29 nov | Conectar la predicción con el chatbot (aviso de riesgo de retraso). Ensayar la demostración con cronómetro. Redactar el informe final (normas APA) | Chatbot con la predicción integrada, informe final en borrador |
| 6 | 30 nov | **Sustentación final**  | Demostración y documento final entregados |



**Estructura de los archivos del avance**

```
├── data/
│   └── proyecto_limpio.csv            # dataset limpio (salida de la preparación)
├── outputs/avance2/                   # gráficos y tabla antes/después
└── src/
    ├── preparacion_proyecto.py        # requisitos B y C
    └── perceptron_psaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasssaroyecto.py         # requisito D
```
