# ================================================================
# APP FERIA ESTADÍSTICA - DISEÑO FACTORIAL 2x3
# Proyecto: Etiqueta nutricional y precio vs. elección de azúcar
# Ejecutar con:  streamlit run app_feria_estadistica.py
# ================================================================

import io
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, f
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

plt.switch_backend("Agg")
sns.set(style="whitegrid")

# ================================================================
# 0. UTILIDADES PARA DESCARGAS
# ================================================================

def fig_to_png_bytes(fig):
    """Convierte una figura de Matplotlib en bytes PNG para descarga."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def generar_pdf(texto):
    """Genera un PDF sencillo con el texto del informe."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 72
    y = height - 72

    for linea in texto.split("\n"):
        # envolver texto para que quepa en el ancho de la página
        for sublinea in textwrap.wrap(linea, 95):
            if y < 72:  # salto de página
                c.showPage()
                y = height - 72
            c.drawString(x_margin, y, sublinea)
            y -= 14

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ================================================================
# 1. CARGA Y LIMPIEZA DEL EXCEL
# ================================================================

def cargar_datos_desde_excel(file):
    """
    Lee el Excel usando header=1 (segunda fila como encabezado),
    renombra la primera columna como 'Tratamiento', rellena hacia abajo
    y elimina filas sin respuestas.
    """
    df = pd.read_excel(file, sheet_name=0, header=1)

    # Nos quedamos con las primeras 5 columnas: Tratamiento, Edad, Sexo, Bebida, Snack
    df = df.iloc[:, :5]

    # Renombrar primera columna (RESPUESTAS ESTADÍSTICA) a 'Tratamiento'
    primera_col = df.columns[0]
    df = df.rename(columns={primera_col: "Tratamiento"})

    # Limpiar espacios en columnas de texto
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # Rellenar tratamientos hacia abajo
    df["Tratamiento"] = df["Tratamiento"].ffill()

    # Quitar filas sin respuestas completas
    df = df.dropna(subset=["Edad", "Sexo", "Bebida Elegida", "Snack Elegido"], how="any")

    # Convertir Edad a numérico
    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    df = df.dropna(subset=["Edad"])

    return df

# ================================================================
# 2. LIMPIEZA NOMBRES PRODUCTOS Y CÁLCULO DE AZÚCAR
# ================================================================

def limpiar_nombre(producto):
    if pd.isna(producto):
        return None

    # Quitar parte del precio → nombre antes de "—" o "-"
    p = str(producto).split("—")[0].split("-")[0].strip().lower()

    reemplazos = {
        "gaseosa": "Gaseosa",
        "jugo": "Jugo Hit",
        "té": "Té",
        "te ": "Té",
        "agua": "Agua",
        "ponqué": "Ponqué",
        "ponque": "Ponqué",
        "galletas": "Galletas",
        "barra": "Barra de cereal",
        "manzana": "Manzana"
    }

    for key, val in reemplazos.items():
        if key in p:
            return val

    return producto  # por si aparece algo extraño

AZUCAR_PRODUCTOS = {
    "Gaseosa": 35,
    "Jugo en caja": 18,
    "Té sin Azúcar": 0,
    "Agua": 0,
    "Ponqué": 24,
    "Galletas": 16,
    "Barra de cereal": 8,
    "Manzana": 10
}

def calcular_azucar(df, umbral_saludable=15):
    df["Bebida"] = df["Bebida Elegida"].apply(limpiar_nombre)
    df["Snack"]  = df["Snack Elegido"].apply(limpiar_nombre)

    df["AzucarBebida"] = df["Bebida"].map(AZUCAR_PRODUCTOS).fillna(0)
    df["AzucarSnack"]  = df["Snack"].map(AZUCAR_PRODUCTOS).fillna(0)

    df["AzucarTotal"] = df["AzucarBebida"] + df["AzucarSnack"]

    df["Saludable"]   = (df["AzucarTotal"] <= umbral_saludable).astype(int)

    return df

# ================================================================
# 3. ASIGNAR FACTORES A (ETIQUETA) Y B (PRECIO)
# ================================================================

def asignar_factores(df):
    """
    A: Etiqueta
       - SIN etiqueta → Tratamientos 1,2,3
       - CON etiqueta → Tratamientos 4,5,6
    B: Precio
       - Igual      → Tratamientos 1 y 4
       - Descuento  → Tratamientos 2 y 5
       - Recargo    → Tratamientos 3 y 6
    """
    df["TratamientoNum"] = df["Tratamiento"].astype(str).str.extract(r"(\d+)").astype(int)

    # Factor A: Etiqueta
    df["Etiqueta"] = df["TratamientoNum"].apply(
        lambda x: "Sin etiqueta" if x in [1, 2, 3] else "Con etiqueta"
    )

    # Factor B: Precio
    precio_map = {
        1: "Igual",
        2: "Descuento",
        3: "Recargo",
        4: "Igual",
        5: "Descuento",
        6: "Recargo",
    }
    df["Precio"] = df["TratamientoNum"].map(precio_map)

    return df

# ================================================================
# 4. ESTIMACIONES E INTERVALOS DE CONFIANZA
# ================================================================

def ic_media(x, alpha=0.05):
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    media = x.mean()
    s = x.std(ddof=1)
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    margen = t_crit * s / np.sqrt(n)
    return media, media - margen, media + margen


def ic_proporcion(p_hat, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan, np.nan
    z_crit = norm.ppf(1 - alpha/2)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    margen = z_crit * se
    return p_hat, p_hat - margen, p_hat + margen


# ================================================================
# 5. PRUEBA DE PROPORCIONES Y ANOVA FACTORIAL
# ================================================================

def prueba_proporciones(p1, p2, n1, n2):
    p_pool = (p1*n1 + p2*n2) / (n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (p1 - p2)/se
    p_value = 2*(1 - norm.cdf(abs(z)))
    return z, p_value

def anova_factorial(df, factorA, factorB, y):
    A = df[factorA].unique()
    B = df[factorB].unique()

    n = df.groupby([factorA, factorB]).size().iloc[0]
    N = len(df)
    media_global = df[y].mean()

    # Sumas de cuadrados
    SSA = sum([
        n * (df[df[factorA] == a][y].mean() - media_global) ** 2
        for a in A
    ])

    SSB = sum([
        n * (df[df[factorB] == b][y].mean() - media_global) ** 2
        for b in B
    ])

    SSAB = 0
    for a in A:
        for b in B:
            media_ab = df[(df[factorA] == a) & (df[factorB] == b)][y].mean()
            media_a = df[df[factorA] == a][y].mean()
            media_b = df[df[factorB] == b][y].mean()
            SSAB += n * (media_ab - media_a - media_b + media_global) ** 2

    SSE = sum((df[y] - df.groupby([factorA, factorB])[y].transform("mean")) ** 2)

    # Grados de libertad
    a = len(A)
    b = len(B)
    dfA = a - 1
    dfB = b - 1
    dfAB = dfA * dfB
    dfE = N - (a * b)

    # Cuadrados medios
    MSA = SSA / dfA
    MSB = SSB / dfB
    MSAB = SSAB / dfAB
    MSE = SSE / dfE

    # Estadísticos F y p-values
    FA = MSA / MSE
    FB = MSB / MSE
    FAB = MSAB / MSE

    pA = 1 - f.cdf(FA, dfA, dfE)
    pB = 1 - f.cdf(FB, dfB, dfE)
    pAB = 1 - f.cdf(FAB, dfAB, dfE)

    # Tabla ANOVA limpia (sin valores vacíos)
    tabla = pd.DataFrame({
        "Fuente": ["Etiqueta (A)", "Precio (B)", "Interacción AB", "Error"],
        "SS": [SSA, SSB, SSAB, SSE],
        "df": [dfA, dfB, dfAB, dfE],
        "MS": [MSA, MSB, MSAB, MSE],
        "F": [FA, FB, FAB, None],
        "p-value": [pA, pB, pAB, None]
    })

    return tabla, (FA, FB, FAB, dfA, dfB, dfAB, dfE)



# ================================================================
# 6. GRÁFICOS (MEJORADOS + INTERPRETACIÓN AUTOMÁTICA)
# ================================================================

def interpretar_proporcion_saludable(df):
    prop = df.groupby("Tratamiento")["Saludable"].mean()
    mayor = prop.idxmax()
    menor = prop.idxmin()

    texto = (
        f"La mayor proporción de elecciones saludables se observa en el **Tratamiento {mayor}** "
        f"({prop[mayor]:.2f}). Esto indica que bajo esa combinación de etiqueta y precio, "
        f"los estudiantes tendieron a elegir opciones más bajas en azúcar.\n\n"
        f"La menor proporción de elecciones saludables se presenta en el **Tratamiento {menor}** "
        f"({prop[menor]:.2f}), lo que sugiere que en esa condición predominan elecciones "
        f"con mayor contenido de azúcar.\n\n"
        "Comparar visualmente las alturas de las barras permite identificar qué tratamientos "
        "son más efectivos para promover decisiones saludables."
    )
    return texto

def grafico_bar_saludable(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    prop = df.groupby("Tratamiento")["Saludable"].mean().reset_index()

    sns.barplot(data=prop, x="Tratamiento", y="Saludable", ax=ax, palette="viridis")

    ax.set_ylim(0, 1)
    ax.set_title("Proporción de elecciones saludables por tratamiento", fontsize=14)
    ax.set_ylabel("Proporción (0 a 1)", fontsize=12)
    ax.set_xlabel("Tratamiento", fontsize=12)

    for i, row in prop.iterrows():
        ax.text(i, row["Saludable"] + 0.03, f"{row['Saludable']:.2f}",
                ha="center", fontsize=10)

    texto = interpretar_proporcion_saludable(df)
    return fig, texto

def interpretar_azucar_promedio(df):
    media = df.groupby("Tratamiento")["AzucarTotal"].mean()
    mayor = media.idxmax()
    menor = media.idxmin()
    dif = media[mayor] - media[menor]

    texto = (
        f"El mayor consumo promedio de azúcar se presenta en el **Tratamiento {mayor}** "
        f"con {media[mayor]:.1f} g, mientras que el menor se observa en el "
        f"**Tratamiento {menor}** con {media[menor]:.1f} g.\n\n"
        f"La diferencia entre ambos tratamientos es de aproximadamente **{dif:.1f} g**. "
        "Esto evidencia que las distintas combinaciones de etiqueta y precio sí cambian "
        "la cantidad de azúcar que terminan eligiendo los estudiantes.\n\n"
        "En general, los tratamientos con barras más bajas representan configuraciones "
        "más favorables desde el punto de vista de la reducción del consumo de azúcar."
    )
    return texto

def grafico_bar_azucar(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    media = df.groupby("Tratamiento")["AzucarTotal"].mean().reset_index()

    sns.barplot(data=media, x="Tratamiento", y="AzucarTotal", ax=ax, palette="rocket")

    ax.set_title("Azúcar promedio por tratamiento", fontsize=14)
    ax.set_ylabel("Azúcar total (g)", fontsize=12)
    ax.set_xlabel("Tratamiento", fontsize=12)

    for i, row in media.iterrows():
        ax.text(i, row["AzucarTotal"] + 1, f"{row['AzucarTotal']:.1f} g",
                ha="center", fontsize=10)

    texto = interpretar_azucar_promedio(df)
    return fig, texto

def interpretar_boxplot(df):
    grupos = df.groupby("Tratamiento")["AzucarTotal"]
    textos = []
    for t, serie in grupos:
        mediana = serie.median()
        iqr = serie.quantile(0.75) - serie.quantile(0.25)
        textos.append(
            f"- Tratamiento {t}: mediana ≈ {mediana:.1f} g, IQR ≈ {iqr:.1f} g."
        )

    texto = (
        "El diagrama de cajas permite analizar la variabilidad del azúcar consumido en cada tratamiento.\n\n"
        + "\n".join(textos) +
        "\n\nLos tratamientos con IQR más grande muestran mayor dispersión en las elecciones, "
        "mientras que los tratamientos con IQR más pequeño indican decisiones más homogéneas."
    )
    return texto

def grafico_box_azucar(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Tratamiento", y="AzucarTotal", ax=ax, palette="Set2")

    ax.set_title("Distribución de azúcar por tratamiento", fontsize=14)
    ax.set_ylabel("Azúcar total (g)", fontsize=12)
    ax.set_xlabel("Tratamiento", fontsize=12)

    texto = interpretar_boxplot(df)
    return fig, texto

def interpretar_interaccion(df):
    tabla = df.groupby(["Etiqueta", "Precio"])["AzucarTotal"].mean().unstack()

    texto = "Interpretación de la interacción Etiqueta × Precio:\n\n"
    for col in tabla.columns:
        diff = tabla.loc["Con etiqueta", col] - tabla.loc["Sin etiqueta", col]
        texto += (
            f"- Para el precio **{col}**, la diferencia (Con etiqueta - Sin etiqueta) "
            f"es de {diff:.1f} g.\n"
        )

    texto += (
        "\nSi estas diferencias varían mucho entre los distintos niveles de precio, "
        "entonces la interacción es importante: el efecto del precio cambia según haya "
        "o no etiqueta. Si las diferencias son similares, la interacción es débil."
    )
    return texto

def grafico_interaccion(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(
        data=df,
        x="Precio",
        y="AzucarTotal",
        hue="Etiqueta",
        dodge=True,
        palette="tab10",
        ax=ax
    )

    ax.set_title("Interacción Etiqueta × Precio", fontsize=14)
    ax.set_ylabel("Azúcar total (g)", fontsize=12)
    ax.set_xlabel("Condición de precio", fontsize=12)

    texto = interpretar_interaccion(df)
    return fig, texto

# ================================================================
# 7. GENERACIÓN DEL INFORME ESCRITO
# ================================================================

def generar_informe_texto(media, li_m, ls_m,
                          p_est, li_p, ls_p,
                          umbral,
                          p1, p2, z, p_val_prop,
                          pA, pB, pAB,
                          alpha,
                          n_total):
    """
    Crea un informe largo en texto, siguiendo una estructura típica:
    Introducción, objetivos, metodología, resultados, conclusiones.
    """
    hoy = datetime.today().strftime("%d/%m/%Y")

    texto = f"""FERIA ESTADÍSTICA – INFORME DEL PROYECTO

Título provisional:
Influencia del etiquetado nutricional y del precio en la elección de productos con distinto contenido de azúcar en estudiantes de bachillerato

Fecha de generación del informe: {hoy}

1. PLANTEAMIENTO DEL PROBLEMA

En la actualidad, el consumo excesivo de azúcar en niños y adolescentes se ha asociado
con sobrepeso, obesidad y otras enfermedades metabólicas. Una de las estrategias de salud
pública ha sido introducir etiquetas frontales de advertencia y modificar los precios de los
productos para desincentivar las opciones menos saludables. Sin embargo, no está claro en qué
medida estas estrategias influyen realmente en las decisiones de consumo de los estudiantes.

Pregunta de investigación:
¿El etiquetado nutricional y las variaciones en el precio influyen en la cantidad de azúcar
que eligen los estudiantes cuando compran una bebida y un snack en la cafetería escolar?

2. OBJETIVOS

Objetivo general:
Analizar el efecto del etiquetado nutricional y del precio en la elección de productos con distinto
contenido de azúcar en estudiantes de grados 9.º a 11.º.

Objetivos específicos:
- Estimar la media y la variabilidad del azúcar total escogido por los estudiantes en una
  combinación bebida + snack.
- Estimar la proporción de elecciones consideradas “saludables” (≤ {umbral} g de azúcar).
- Comparar la proporción de elecciones saludables entre productos con etiqueta y sin etiqueta.
- Evaluar, mediante un diseño factorial 2×3, el efecto de la etiqueta, del precio y de su interacción
  sobre el azúcar total elegido.

3. METODOLOGÍA

Población:
Estudiantes de bachillerato (grados 9.º a 11.º) del colegio, con edades entre 14 y 18 años.

Muestra:
Se trabajó con una muestra de tamaño n = {n_total}, obtenida por conveniencia
a partir de los estudiantes que aceptaron responder la encuesta.

Diseño experimental:
Se implementó un diseño factorial completamente al azar de tipo 2×3 con dos factores:
- Factor A: Etiqueta (A1 = sin etiqueta, A2 = con etiqueta).
- Factor B: Precio (B1 = precio igual, B2 = con descuento, B3 = con recargo).
Cada combinación de niveles (tratamiento) corresponde a una forma distinta de presentar
los mismos productos de cafetería (bebida y snack), variando solo etiqueta y precio.

Variable de respuesta:
Azúcar total (en gramos) de la combinación bebida + snack escogida por cada estudiante.
Adicionalmente se definió una variable dicotómica “Saludable”, que vale 1 si el azúcar total
es menor o igual a {umbral} g y 0 en caso contrario.

4. RESULTADOS DESCRIPTIVOS

La media muestral del azúcar total consumido fue de aproximadamente {media:.2f} g.
El intervalo de confianza al {int((1-alpha)*100)} % para la media poblacional de azúcar total es:
({li_m:.2f} g ; {ls_m:.2f} g). Esto indica que, con alta confianza, el promedio real de azúcar
que escogería un estudiante al comprar bebida + snack se encuentra en ese rango.

En cuanto a la elección saludable, la proporción global de estudiantes que eligieron
una combinación con ≤ {umbral} g de azúcar fue p̂ = {p_est:.3f}. El intervalo de confianza
al {int((1-alpha)*100)} % para la proporción poblacional de elecciones saludables es:
({li_p:.3f} ; {ls_p:.3f}). Esto sugiere que, en la población de estudiantes similar a la muestra,
entre el {li_p*100:.1f}% y el {ls_p*100:.1f}% elegiría opciones relativamente bajas en azúcar.

5. PRUEBA DE HIPÓTESIS SOBRE LA ETIQUETA (PROPORCIONES)

Se comparó la proporción de elecciones saludables en dos grupos:
- Con etiqueta: p1 = {p1:.3f}
- Sin etiqueta: p2 = {p2:.3f}

Se plantearon las hipótesis:
H0: p1 = p2  (la etiqueta no cambia la proporción de elecciones saludables)
H1: p1 ≠ p2  (la etiqueta sí cambia la proporción de elecciones saludables)

El estadístico de prueba Z fue aproximadamente {z:.3f}, con un valor-p de {p_val_prop:.4f}."""

    texto += f"""

6. ANÁLISIS DE VARIANZA (ANOVA) FACTORIAL 2×3

Mediante ANOVA de dos vías se evaluó el efecto de:
- La etiqueta (Factor A).
- El precio (Factor B).
- La interacción A×B.

Los valores-p obtenidos fueron:
- Etiqueta (A): p-value = {pA:.4f}
- Precio (B):  p-value = {pB:.4f}
- Interacción A×B: p-value = {pAB:.4f}

Con un nivel de significancia α = {alpha:.2f}, se interpreta:

- Si el valor-p de la etiqueta es menor que α, se concluye que el hecho de mostrar o no
  la etiqueta nutricional sí modifica de forma significativa la cantidad de azúcar escogida.
- Si el valor-p del precio es menor que α, las estrategias de precio (igual, descuento o recargo)
  sí generan diferencias en el azúcar total elegido.
- Si el valor-p de la interacción es menor que α, el efecto del precio depende de la presencia
  de la etiqueta (o viceversa).

7. CONCLUSIONES GENERALES

A partir de los resultados obtenidos en esta muestra de estudiantes se concluye que:

- El consumo promedio de azúcar por compra de bebida + snack se sitúa alrededor de
  {media:.2f} g, lo que indica que, en general, la elección típica puede superar el
  umbral de {umbral} g definido como saludable.

- La proporción de estudiantes que elige combinaciones consideradas saludables es
  aproximadamente {p_est:.3f}, lo que sugiere que todavía una parte importante de los
  alumnos se inclina por opciones altas en azúcar.

- El contraste de proporciones entre productos con etiqueta y sin etiqueta permite evaluar
  si la sola presencia de la información nutricional logra cambiar la conducta de consumo.
  Según el valor-p obtenido, se decide si hay evidencia estadística suficiente para afirmar
  que la etiqueta sí modifica la proporción de elecciones saludables.

- El ANOVA factorial 2×3 muestra si la etiqueta, el precio y su interacción tienen efecto
  sobre el azúcar total escogido. En particular, la interacción A×B es clave para saber
  si las estrategias de precio funcionan de la misma forma cuando hay etiqueta y cuando no.

8. RECOMENDACIONES

- Repetir el estudio con muestras de mayor tamaño y en otros cursos o instituciones
  para generalizar los resultados.
- Explorar otros tipos de etiquetas (colores, advertencias más explícitas) y diferentes
  magnitudes de descuentos o recargos.
- Complementar el experimento con actividades pedagógicas sobre lectura de etiquetas
  nutricionales y riesgos del exceso de consumo de azúcar.

9. LIMITACIONES

- La muestra se obtuvo por conveniencia y se restringe a un solo colegio, lo cual limita
  la posibilidad de generalizar los resultados.
- Los datos provienen de un experimento simulado, en el que los estudiantes escogen
  opciones a partir de alternativas presentadas en una encuesta y no necesariamente
  de compras reales en la cafetería.
- El criterio de “saludable” se basa únicamente en el contenido de azúcar, y no considera
  otros nutrientes como grasas, sodio o fibra.

Este informe puede utilizarse como base para la redacción final del trabajo escrito,
ajustando el lenguaje y completando los apartados específicos que pida la rúbrica
de la Feria Estadística.
"""

    return texto

# ================================================================
# 8. INTERFAZ STREAMLIT
# ================================================================

def main():
    st.title("Feria Estadística – Diseño factorial 2×3")
    st.write("""
    **Proyecto:** Influencia del etiquetado nutricional y del precio en la elección
    de productos con distinto contenido de azúcar en estudiantes de 9.º a 11.º.
    """)

    uploaded_file = st.file_uploader("Sube el archivo de respuestas (.xlsx)", type=["xlsx"])

    umbral = st.sidebar.slider(
        "Umbral de azúcar para considerar 'saludable' (g)",
        min_value=0, max_value=50, value=15, step=1
    )

    alpha = st.sidebar.slider(
        "Nivel de significancia α",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )

    if uploaded_file is None:
        st.info("Por favor sube el archivo **RespuestasEncuestaEstadistica.xlsx**.")
        return

    # ----------------- Procesamiento de datos -----------------
    df = cargar_datos_desde_excel(uploaded_file)
    df = asignar_factores(df)
    df = calcular_azucar(df, umbral_saludable=umbral)

    n_total = len(df)

    st.subheader("Datos procesados (todas las filas)")
    st.dataframe(df)  # ⬅ ahora muestra TODO, no solo head(15)

    # ----------------- Descriptiva -----------------
    st.subheader("1. Análisis descriptivo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Media, mediana y cuartiles de azúcar total (g)**")
        st.write(df["AzucarTotal"].describe()[["mean","50%","25%","75%"]])
    with col2:
        st.markdown("**Proporción de elecciones saludables por tratamiento**")
        st.write(df.groupby("Tratamiento")["Saludable"].mean())

    # ----------------- Estimaciones e IC -----------------
    st.subheader("2. Estimaciones puntuales e intervalos de confianza")

    media, li_m, ls_m = ic_media(df["AzucarTotal"], alpha=alpha)
    st.markdown("**2.1. Media poblacional de azúcar total (g)**")
    st.write(f"Estimación puntual:  x̄ = {media:.2f} g")
    st.write(f"IC {int((1-alpha)*100)}%: ({li_m:.2f} g ; {ls_m:.2f} g)")
    st.markdown(
        f"_Interpretación_: Con un nivel de confianza del {int((1-alpha)*100)} %, "
        f"la **media real de azúcar total** consumida por los estudiantes se encuentra "
        f"entre **{li_m:.2f} g** y **{ls_m:.2f} g**."
    )

    p_hat = df["Saludable"].mean()
    p_est, li_p, ls_p = ic_proporcion(p_hat, n_total, alpha=alpha)
    st.markdown("**2.2. Proporción poblacional de elecciones saludables**")
    st.write(f"Estimación puntual:  p̂ = {p_est:.3f}")
    st.write(f"IC {int((1-alpha)*100)}%: ({li_p:.3f} ; {ls_p:.3f})")
    st.markdown(
        f"_Interpretación_: Con un nivel de confianza del {int((1-alpha)*100)} %, "
        f"la **proporción real de estudiantes que eligen una combinación saludable** "
        f"se encuentra entre **{li_p:.3f}** y **{ls_p:.3f}**."
    )

    # ----------------- Prueba de hipótesis (proporciones) -----------------
    st.subheader("3. Prueba de hipótesis: efecto de la etiqueta en la proporción saludable")

    p1 = df[df["Etiqueta"]=="Con etiqueta"]["Saludable"].mean()
    p2 = df[df["Etiqueta"]=="Sin etiqueta"]["Saludable"].mean()
    n1 = len(df[df["Etiqueta"]=="Con etiqueta"])
    n2 = len(df[df["Etiqueta"]=="Sin etiqueta"])
    z, p_value = prueba_proporciones(p1, p2, n1, n2)

    st.write(f"p₁ = proporción saludable **con etiqueta** = {p1:.3f} (n₁ = {n1})")
    st.write(f"p₂ = proporción saludable **sin etiqueta** = {p2:.3f} (n₂ = {n2})")
    st.write(f"Estadístico Z = {z:.3f}")
    st.write(f"Valor-p = {p_value:.4f}")

    st.markdown("""
    **Hipótesis planteadas**

    - H₀: p₁ = p₂ → la etiqueta **no cambia** la proporción de elecciones saludables.  
    - H₁: p₁ ≠ p₂ → la etiqueta **sí cambia** la proporción de elecciones saludables.
    """)

    if p_value < alpha:
        st.success(
            f"Como p-value = {p_value:.4f} < α = {alpha:.2f}, se **rechaza H₀**. "
            "Concluimos que el etiquetado nutricional tiene un efecto estadísticamente "
            "significativo en la proporción de elecciones saludables."
        )
    else:
        st.info(
            f"Como p-value = {p_value:.4f} ≥ α = {alpha:.2f}, **no se rechaza H₀**. "
            "Con los datos de esta muestra, no hay evidencia suficiente para afirmar que la etiqueta "
            "modifique la proporción de estudiantes que eligen opciones saludables."
        )

    # ----------------- ANOVA factorial 2×3 -----------------
    st.subheader("4. ANOVA factorial 2×3 (Etiqueta × Precio) sobre azúcar total")

    tabla_anova, anova_info = anova_factorial(df, "Etiqueta", "Precio", "AzucarTotal")
    st.dataframe(tabla_anova)

    FA, FB, FAB, dfA, dfB, dfAB, dfE = anova_info
    pA = tabla_anova.loc[0, "p-value"]
    pB = tabla_anova.loc[1, "p-value"]
    pAB = tabla_anova.loc[2, "p-value"]

    st.markdown("""
    **Hipótesis del ANOVA**

    - Para Etiqueta (A):  
      H₀: las medias de azúcar son iguales con y sin etiqueta.  
      H₁: al menos una de las medias difiere.

    - Para Precio (B):  
      H₀: las medias de azúcar son iguales entre las estrategias de precio (igual, descuento, recargo).  
      H₁: al menos una de las medias difiere.

    - Para la Interacción A×B:  
      H₀: no hay interacción entre etiqueta y precio.  
      H₁: sí hay interacción; el efecto de uno depende del otro.
    """)

    def interpreta_factor(nombre, p_val):
        if p_val < alpha:
            st.success(
                f"Para **{nombre}**: p-value = {p_val:.4f} < α = {alpha:.2f} ⇒ "
                "se **rechaza H₀**. Hay evidencia de que este factor tiene un "
                "efecto significativo en el azúcar total elegido."
            )
        else:
            st.info(
                f"Para **{nombre}**: p-value = {p_val:.4f} ≥ α = {alpha:.2f} ⇒ "
                "**no se rechaza H₀**. Con esta muestra no hay evidencia suficiente "
                "de que este factor afecte de manera importante el azúcar total elegido."
            )

    interpreta_factor("la ETIQUETA", pA)
    interpreta_factor("el PRECIO", pB)
    interpreta_factor("la INTERACCIÓN Etiqueta × Precio", pAB)

    st.markdown(
        "_En términos del problema_, el ANOVA indica si el etiquetado, las estrategias de precio "
        "o la combinación de ambos logran modificar de forma significativa la cantidad de azúcar "
        "que escogen los estudiantes."
    )

    # ----------------- Gráficos + descarga PNG -----------------
    st.subheader("5. Gráficos")

    # 5.1
    st.markdown("### 5.1 Proporción de elecciones saludables por tratamiento")
    fig, texto = grafico_bar_saludable(df)
    st.pyplot(fig)
    st.info(texto)
    st.download_button(
        "Descargar gráfico 5.1 (PNG)",
        data=fig_to_png_bytes(fig),
        file_name="grafico_5_1_proporcion_saludable.png",
        mime="image/png"
    )

    # 5.2
    st.markdown("### 5.2 Azúcar promedio por tratamiento")
    fig, texto = grafico_bar_azucar(df)
    st.pyplot(fig)
    st.info(texto)
    st.download_button(
        "Descargar gráfico 5.2 (PNG)",
        data=fig_to_png_bytes(fig),
        file_name="grafico_5_2_azucar_promedio.png",
        mime="image/png"
    )

    # 5.3
    st.markdown("### 5.3 Distribución de azúcar por tratamiento (boxplot)")
    fig, texto = grafico_box_azucar(df)
    st.pyplot(fig)
    st.info(texto)
    st.download_button(
        "Descargar gráfico 5.3 (PNG)",
        data=fig_to_png_bytes(fig),
        file_name="grafico_5_3_boxplot_azucar.png",
        mime="image/png"
    )

    # 5.4
    st.markdown("### 5.4 Interacción Etiqueta × Precio")
    fig, texto = grafico_interaccion(df)
    st.pyplot(fig)
    st.info(texto)
    st.download_button(
        "Descargar gráfico 5.4 (PNG)",
        data=fig_to_png_bytes(fig),
        file_name="grafico_5_4_interaccion.png",
        mime="image/png"
    )

    # ----------------- Informe y conclusiones -----------------
    st.subheader("6. Informe escrito y conclusiones")

    informe_texto = generar_informe_texto(
        media, li_m, ls_m,
        p_est, li_p, ls_p,
        umbral,
        p1, p2, z, p_value,
        pA, pB, pAB,
        alpha,
        n_total
    )

    st.markdown("#### 6.1 Informe automáticamente generado (puedes copiarlo y pulirlo)")
    st.text_area("Informe", informe_texto, height=400)

    pdf_bytes = generar_pdf(informe_texto)
    st.download_button(
        "Descargar informe completo en PDF",
        data=pdf_bytes,
        file_name="informe_feria_estadistica.pdf",
        mime="application/pdf"
    )

    st.markdown("#### 6.2 Conclusiones del proyecto (resumen corto, listo para pegar en el póster)")
    conclusiones_resumen = f"""
- El consumo promedio de azúcar por combinación bebida + snack fue de {media:.2f} g.
- Aproximadamente el {p_est*100:.1f}% de los estudiantes eligió opciones consideradas saludables (≤ {umbral} g de azúcar).
- La comparación de proporciones entre productos con y sin etiqueta arrojó un valor-p de {p_value:.4f}, \
lo que {'indica un efecto significativo del etiquetado.' if p_value < alpha else 'no muestra evidencia estadística suficiente de efecto del etiquetado.'}
- El ANOVA factorial 2×3 permitió evaluar simultáneamente los efectos de la etiqueta, del precio y de su interacción sobre \
el azúcar total elegido, identificando qué factores tienen impacto real en el comportamiento de compra simulado.
    """
    st.write(conclusiones_resumen.strip())

# ================================================================
# 9. MAIN
# ================================================================

if __name__ == "__main__":
    main()
