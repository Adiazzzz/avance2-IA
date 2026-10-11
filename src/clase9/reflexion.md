# Reflexión – Clase 8: ¿Cómo aprende una IA?

## 1. ¿qué se hizo?

En esta clase se construyó una **neurona artificial** (un perceptrón) en Python puro, sin librerías, para que **aprendiera la compuerta lógica AND**. La compuerta AND solo da 1 cuando las dos entradas son 1:

| x₁ | x₂ | Salida esperada |
|----|----|-----------------|
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 |

**Cómo funciona la neurona.** Recibe dos entradas (x₁ y x₂), cada una con su peso (w₁ y w₂), y un bias (b). Hace esta cuenta:

```
z = w₁·x₁ + w₂·x₂ + b
```

Después pasa `z` por la función escalón: si `z >= 0` la salida es 1, y si `z < 0` la salida es 0.

**Cómo aprende.** Los pesos y el bias empiezan con valores que no sirven (`w₁ = 0.1`, `w₂ = 0.1`, `b = 0.0`) y se van corrigiendo con ejemplos. Para cada fila de la tabla se hace lo siguiente:

1. La neurona predice una salida.
2. Se calcula el error: `error = y_real - y_pred`.
3. Se corrigen los parámetros con la tasa de aprendizaje (0.1):
   - `w += tasa · error · x`
   - `b += tasa · error`

Cuando se recorren las 4 filas una vez, eso es **una época**. El entrenamiento se repite hasta que el error total de una época es 0, o hasta llegar a 10 épocas. Al final se guardó el error total de cada época y se hizo un gráfico con la evolución.

---

## 2. Resultados obtenidos

La neurona **aprendió en 3 épocas**:

| Época | Error total | w₁ | w₂ | b |
|-------|-------------|-----|-----|-------|
| 1 | 3 | 0.2 | 0.1 | -0.1 |
| 2 | 3 | 0.2 | 0.1 | -0.2 |
| 3 | 0 | 0.2 | 0.1 | -0.2 |

Y la prueba final dio las 4 filas correctas (OK).

**Qué dice el gráfico** (`evolución_error.png`): el error empieza en 3, sigue en 3 en la segunda época y cae a 0 en la tercera.

**Por qué pasó así.** Al principio el bias es 0, así que la fila (0, 0) da `z = 0`, y como el escalón da 1 cuando `z >= 0`, la neurona se activa cuando no debería. Está "demasiado dispuesta" a decir que sí. En la época 1 se equivoca en 3 filas, y las correcciones van bajando el bias (de 0.0 a -0.1). Eso no alcanza, y en la época 2 vuelve a fallar en 3 filas, aunque otras. El bias llega a -0.2 y los pesos quedan en 0.2 y 0.1. Con esos valores ya solo se activa cuando las dos entradas valen 1: 0.2 + 0.1 − 0.2 = 0.1, que es mayor que 0.

**Por qué la línea cae de golpe.** Como el escalón solo da 0 o 1, cada predicción está bien o está mal, sin términos medios. Por eso el error es un conteo de fallas (3, 3, 0) y no baja poco a poco. La línea recta entre la época 2 y la 3 solo une dos puntos; no existe una "época 2.5".

**Un detalle.** Con los parámetros finales, la fila (1, 0) da `z = 0.2 − 0.2 = 0`, justo en el límite. Con el escalón, `z = 0` daría 1, que sería un error. Si se imprime el valor exacto, sale `-0.0000000000000000278`: una diferencia mínima por cómo el computador guarda los decimales. Así que la neurona acierta, pero **en el borde** de la decisión. Un cambio pequeño en los parámetros bastaría para que esa fila fallara.

---

## 3. Preguntas de reflexión de la guía (6.6)

Para responderlas se corrió el mismo código cambiando solo los valores iniciales y la tasa de aprendizaje, y se registró qué pasaba en cada caso.

| Caso | Épocas para aprender | Error total por época |
|------|----------------------|-----------------------|
| Base (w=0.1, b=0.0, tasa 0.1) | 3 | 3, 3, 0 |
| Todo en 0 (w₁=w₂=b=0) | 4 | 2, 3, 3, 0 |
| Tasa 1.0 | 6 | 2, 3, 3, 2, 1, 0 |
| Tasa 0.001 | 34 | 3, 2, 2, 2, 2, 2, ... (se queda en 2 por muchas épocas) |
| Sin bias (b = 0 siempre) | Nunca  | no llega a 0 |

### 1. ¿Qué pasa si se inicializa w₁ = 0, w₂ = 0, b = 0? ¿La neurona aprende? ¿Por qué?

**Sí aprende, en 4 épocas** (los errores son 2, 3, 3 y 0). Se podría pensar que, si todo empieza en cero, no hay de dónde corregir, pero no es así. Con todo en cero, `z = 0` en todas las filas y la neurona predice 1 siempre. Eso genera errores, y cada error mueve los parámetros. El bias se corrige en cada error, y cada peso solo se corrige cuando su entrada vale 1. Por eso los pesos terminan con valores distintos entre sí (w₁ = 0.2 y w₂ = 0.1) aunque hayan empezado iguales.

### 2. ¿Qué pasa si la tasa de aprendizaje es muy alta (1.0)?

En esta prueba **aprendió, en 6 épocas**, pero con saltos grandes: terminó en w₁ = 2.1, w₂ = 1.1 y b = -3.0, contra 0.2, 0.1 y -0.2 con la tasa de 0.1. No se desestabilizó, y creo que es porque el escalón solo mira si `z` es positivo o negativo, y la compuerta AND se puede separar bien. Los valores crecen mucho, pero la decisión sigue siendo la correcta. Aun así, una tasa alta no es buena idea en general: los parámetros pasan de largo y se pierde precisión. Con funciones más suaves y problemas más difíciles, que se verán en próximas clases, ese problema sí puede aparecer.

### 3. ¿Qué pasa si la tasa de aprendizaje es muy baja (0.001)? ¿Cuántas épocas necesita?

Necesitó **34 épocas**, más de diez veces las del caso base. Cada corrección mueve los parámetros una cantidad minúscula, así que el error se queda en 2 durante muchas épocas seguidas hasta que el acumulado de pasitos por fin cambia la predicción de una fila. Una tasa baja es más segura pero lenta.

### 4. ¿Por qué el bias es necesario? ¿Qué pasaría si b = 0 siempre?

Si el bias se deja siempre en 0, **la neurona nunca aprende la compuerta AND**. La razón está en la fila (0, 0): ahí `z = w₁·0 + w₂·0 + b = b`. Los pesos se multiplican por 0 y desaparecen, así que solo el bias decide la salida. Sin bias, `z` es siempre 0, el escalón da 1, y se espera 0. Ningún cambio en los pesos puede arreglarlo. 

### 5. ¿Cómo cambiaría el código si se quisiera aprender la compuerta OR en lugar de AND?

Solo cambia la lista de datos de entrenamiento, que pasa a:

```python
datos = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
]
```

La neurona, el escalón, el cálculo del error y el ajuste de pesos y bias son iguales. Con los mismos valores iniciales (0.1, 0.1, 0.0) y tasa 0.1, la neurona aprendió OR en 2 épocas (errores 1 y 0).

---

