from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

R = Path(__file__).resolve().parent.parent
VERDE, ROJO, GRIS, AZUL = "#1DB954", "#e4572e", "#a3ada7", "#2e86de"
GENEROS = ["edm", "latin", "pop", "r&b", "rap", "rock"]
ESCALA = alt.Scale(domain=GENEROS, range=["#2e86de", "#f39c12", "#9b59b6", "#16a085", "#e4572e", "#34495e"])
TIPOS = ["acústica", "remix", "en vivo", "otra", "radio edit", "remaster"]
COLOR_TIPO = alt.Scale(domain=TIPOS, range=["#16a085", "#e4572e", "#9b59b6", GRIS, "#f39c12", AZUL])

st.set_page_config(page_title="PIA Minería de Datos", page_icon=":material/graphic_eq:", layout="wide")


@st.cache_data
def leer(semana, archivo, carpeta="resultados"):
    return pd.read_csv(R / f"Semana {semana}" / carpeta / archivo)


def grafica(semana, archivo):
    return str(R / f"Semana {semana}" / "graficas" / archivo)


@st.cache_data
def otras_del_artista():
    df = leer(5, "canciones.csv", "dataframes")
    g = df.groupby(df.track_artist.str.lower().str.strip()).track_popularity
    suma, n = g.transform("sum"), g.transform("count")
    d = df.assign(otras=(suma - df.track_popularity) / (n - 1))[n >= 2]
    return d.sample(3000, random_state=42)[["track_name", "track_artist", "track_popularity", "otras"]]


def tarjetas(datos):
    with st.container(horizontal=True):
        for etiqueta, valor, nota in datos:
            st.metric(etiqueta, valor, nota, delta_color="off", delta_arrow="off", border=True)


canciones = leer(5, "canciones.csv", "dataframes")
reg = leer(9, "resumen_multiple.csv").iloc[0]
coef = leer(9, "coeficientes_multiple.csv").set_index("termino")
knn = leer(10, "resumen.csv").iloc[0]
km = leer(12, "resumen.csv").iloc[0]
serie = leer(13, "serie_anual.csv")
pron = leer(13, "resumen.csv").iloc[0]
texto = leer(14, "resumen.csv").iloc[0]
marcas = leer(14, "marcadores.csv")
pop = marcas[marcas.variable == "track_popularity"].set_index("marcador")
versiones = leer(15, "versiones_resumen.csv").set_index("tipo")
artista = leer(15, "artista_vs_audio.csv")
viernes = leer(15, "viernes.csv").iloc[0]
s = serie.set_index("anio")


# INICIO
def inicio():
    st.title("¿Qué hace que una canción sea popular?")
    st.caption(f"{len(canciones):,} canciones de playlists de Spotify · {canciones.release_year.min()}-"
               f"{canciones.release_year.max()} · 6 géneros")
    tarjetas([("Canciones", f"{len(canciones):,}", None),
              ("Artistas", f"{canciones.track_artist.nunique():,}", None),
              ("Géneros", "6", "edm, latin, pop, r&b, rap, rock"),
              ("Técnicas avanzadas", "4", "clasificación, clustering, predicción, texto")])
    st.subheader("Hallazgos")
    a0 = artista.iloc[0]
    hallazgos = [
        (":material/graphic_eq:", "El audio casi no explica la popularidad", "R²", f"{reg.R2:.3f}"),
        (":material/person:", "El artista la predice mucho mejor", "veces mejor que el audio", f"{a0.veces_mejor:.1f}"),
        (":material/category:", "El género no se escucha", "acierto del KNN", f"{knn.exactitud_prueba:.1%}"),
        (":material/timer:", "Las canciones se acortan", "duración media en 2019", f"{s.duracion_min[2019]:.2f} min"),
        (":material/library_music:", "La versión original le gana a sus variantes", "de las veces",
         f"{versiones.loc['todas'].gana_principal:.1%}"),
        (":material/calendar_month:", "El viernes se volvió el día de estreno", "de los estrenos 2016-2019",
         f"{viernes.viernes_2016_2019:.1%}"),
    ]
    for fila in (hallazgos[:3], hallazgos[3:]):
        for col, (icono, titulo, etiqueta, valor) in zip(st.columns(3), fila):
            with col.container(border=True):
                st.markdown(f"**{icono} {titulo}**")
                st.metric(etiqueta, valor)


# AUDIO
def audio():
    st.title("El audio casi no explica la popularidad")
    st.caption("Regresión lineal múltiple con 11 variables de audio contra la popularidad (0 a 100)")
    a0 = artista.iloc[0]
    tarjetas([("R² del audio", f"{reg.R2:.3f}", f"{reg.R2_prueba:.3f} en canciones de prueba"),
              ("Prueba F", f"{reg.F:.1f}", "p < 0.001"),
              ("R² usando solo al artista", f"{a0.r2_artista_fuera_de_muestra:.3f}", "mismas canciones de prueba"),
              ("Veces mejor que el audio", f"{a0.veces_mejor:.1f}", f"entre {artista.veces_mejor.min():.1f} y "
               f"{artista.veces_mejor.max():.1f} según el filtro")])

    izq, der = st.columns(2)
    with izq.container(border=True):
        st.markdown("**Peso de cada variable** (beta estandarizada)")
        c = coef.drop("intercepto").reset_index()
        c["efecto"] = np.where(c.beta_estandarizado > 0, "sube", "baja")
        st.altair_chart(alt.Chart(c).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("beta_estandarizado:Q", title=None),
            y=alt.Y("termino:N", sort=alt.EncodingSortField("beta_estandarizado", order="ascending"), title=None),
            color=alt.Color("efecto:N", scale=alt.Scale(domain=["sube", "baja"], range=[VERDE, ROJO]),
                            legend=alt.Legend(title="popularidad", orient="bottom")),
            tooltip=["termino", alt.Tooltip("beta_estandarizado:Q", title="beta", format="+.3f")],
        ), width="stretch")
    with der.container(border=True):
        st.markdown("**Qué predice mejor la popularidad** (R² en canciones de prueba)")
        a = artista.melt(id_vars="minimo_canciones", value_vars=["r2_audio_prueba", "r2_artista_fuera_de_muestra"],
                         var_name="modelo", value_name="r2")
        a["modelo"] = a.modelo.map({"r2_audio_prueba": "audio", "r2_artista_fuera_de_muestra": "artista"})
        a["filtro"] = a.minimo_canciones.astype(str) + "+ canciones"
        st.altair_chart(alt.Chart(a).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("filtro:N", title="artistas con", sort=list(a.filtro.unique()), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("modelo:N", sort=["audio", "artista"]),
            y=alt.Y("r2:Q", title="R²"),
            color=alt.Color("modelo:N", scale=alt.Scale(domain=["audio", "artista"], range=[GRIS, VERDE]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["filtro", "modelo", alt.Tooltip("r2:Q", format=".3f")],
        ), width="stretch")

    izq, der = st.columns(2)
    with izq.container(border=True):
        st.markdown("**Una variable de audio contra la popularidad**")
        var = st.segmented_control("Variable", ["energy", "danceability", "loudness", "valence", "acousticness",
                                                "duracion_min", "tempo"], default="energy", required=True,
                                   label_visibility="collapsed")
        muestra = canciones.sample(3000, random_state=42)
        puntos = alt.Chart(muestra).mark_circle(size=12, opacity=0.25, color=GRIS).encode(
            x=alt.X(f"{var}:Q", scale=alt.Scale(zero=False)), y=alt.Y("track_popularity:Q", title="popularidad"),
            tooltip=["track_name", "track_artist", var, "track_popularity"])
        st.altair_chart(puntos + puntos.transform_regression(var, "track_popularity").mark_line(color=ROJO, size=3),
                        width="stretch")
        st.caption(f"r = {canciones[var].corr(canciones.track_popularity):+.3f}")
    with der.container(border=True):
        st.markdown("**Las otras canciones del mismo artista contra la popularidad**")
        o = otras_del_artista()
        puntos = alt.Chart(o).mark_circle(size=12, opacity=0.25, color=VERDE).encode(
            x=alt.X("otras:Q", title="popularidad media de sus otras canciones"),
            y=alt.Y("track_popularity:Q", title="popularidad"),
            tooltip=["track_name", "track_artist", "track_popularity", alt.Tooltip("otras:Q", format=".1f")])
        st.altair_chart(puntos + puntos.transform_regression("otras", "track_popularity").mark_line(color=ROJO, size=3),
                        width="stretch")
        st.caption(f"r = {artista.r_artista.iloc[0]:+.3f}")


# GENERO
def genero():
    st.title("El género no se escucha")
    st.caption("Clasificación KNN y clustering k-means sobre las mismas variables de audio")
    clases = leer(10, "metricas_por_clase.csv")
    clases = clases[clases.clase.isin(GENEROS)]
    tarjetas([("Acierto del KNN", f"{knn.exactitud_prueba:.1%}", f"el azar acierta {knn.exactitud_base:.1%}"),
              ("Mejor número de grupos", f"{int(km.k)}", f"silueta = {km.silueta:.3f}"),
              ("Pureza con 6 grupos", f"{km.pureza_con_k_igual_generos:.1%}", "contra los 6 géneros"),
              ("Rand ajustado", f"{km.rand_ajustado_con_k_igual_generos:.3f}", "0 = azar, 1 = coinciden")])

    izq, der = st.columns(2)
    with izq.container(border=True):
        st.markdown(f"**Clasificación KNN** · qué tan bien reconoce cada género (k = {int(knn.k)})")
        barras = alt.Chart(clases).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("f1-score:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("clase:N", sort="-x", title=None), color=alt.Color("clase:N", scale=ESCALA, legend=None),
            tooltip=["clase", alt.Tooltip("f1-score:Q", format=".3f")])
        azar = alt.Chart(pd.DataFrame({"x": [knn.exactitud_base]})).mark_rule(
            color=ROJO, strokeDash=[5, 4], size=2).encode(x="x:Q")
        st.altair_chart(barras + azar, width="stretch")
        conf = leer(10, "matriz_confusion_normalizada.csv").rename(columns={"Unnamed: 0": "real"})
        larga = conf.melt(id_vars="real", var_name="predicho", value_name="p")
        base = alt.Chart(larga).encode(
            x=alt.X("predicho:N", sort=GENEROS, title="el modelo dijo", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("real:N", sort=GENEROS, title="género real"))
        st.altair_chart(base.mark_rect(cornerRadius=3).encode(
            color=alt.Color("p:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["real", "predicho", alt.Tooltip("p:Q", format=".1%")]) + base.mark_text(fontSize=12).encode(
            text=alt.Text("p:Q", format=".0%"),
            color=alt.when(alt.datum.p > 0.45).then(alt.value("white")).otherwise(alt.value("#191414"))),
            width="stretch")

    with der.container(border=True):
        st.markdown("**Clustering k-means** · cuántos grupos hay en el audio")
        sel = leer(12, "seleccion_k.csv").melt(id_vars="k", value_vars=["silueta", "pureza_genero"],
                                                var_name="medida", value_name="valor")
        sel["medida"] = sel.medida.map({"silueta": "silueta", "pureza_genero": "pureza contra el género"})
        st.altair_chart(alt.Chart(sel).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("k:O", title="número de grupos", axis=alt.Axis(labelAngle=0)), y=alt.Y("valor:Q", title=None),
            color=alt.Color("medida:N", scale=alt.Scale(domain=["silueta", "pureza contra el género"],
                                                        range=[VERDE, GRIS]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["k", "medida", alt.Tooltip("valor:Q", format=".3f")]), width="stretch")
        mezcla = leer(12, "grupo_por_genero_normalizado.csv").melt(id_vars="grupo", var_name="genero",
                                                                    value_name="p")
        mezcla["grupo"] = "grupo " + mezcla.grupo.astype(str)
        st.markdown("**De qué género es cada grupo** (k = 6)")
        st.altair_chart(alt.Chart(mezcla).mark_bar().encode(
            x=alt.X("p:Q", stack="normalize", title=None, axis=alt.Axis(format="%")), y=alt.Y("grupo:N", title=None),
            color=alt.Color("genero:N", scale=ESCALA, legend=alt.Legend(title=None, orient="bottom")),
            order=alt.Order("genero:N"), tooltip=["grupo", "genero", alt.Tooltip("p:Q", format=".1%")]),
            width="stretch")


# TIEMPO
def tiempo():
    st.title("Las canciones se acortan")
    st.caption("Serie de tiempo de la duración media por año y fechas de estreno")
    tarjetas([("Duración media 2013", f"{s.duracion_min[2013]:.2f} min", None),
              ("Duración media 2019", f"{s.duracion_min[2019]:.2f} min",
               f"{s.duracion_min[2019] - s.duracion_min[2013]:+.2f} min"),
              ("Pronóstico 2024", f"{pron.pronostico_2024:.2f} min", f"MAPE {pron.mape_recientes:.1f}%"),
              ("Estrenos en viernes", f"{viernes.viernes_2016_2019:.1%}",
               f"{viernes.viernes_2010_2014:.1%} en 2010-2014")])

    p = leer(13, "pronostico.csv")
    errores = leer(13, "errores_modelo.csv").tail(5)
    izq, der = st.columns([3, 2])
    with izq.container(border=True):
        st.markdown("**Duración media por año y pronóstico**")
        eje = alt.X("anio:Q", title=None, axis=alt.Axis(format="d"), scale=alt.Scale(domain=[1980, 2024]))
        obs = alt.Chart(serie).mark_line(point=alt.OverlayMarkDef(color=VERDE), color=VERDE, strokeWidth=3).encode(
            x=eje, y=alt.Y("duracion_min:Q", title="minutos", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("anio:Q", title="año", format="d"), alt.Tooltip("duracion_min:Q", format=".2f")])
        banda = alt.Chart(p).mark_area(color=ROJO, opacity=0.15).encode(x="anio:Q", y="ic_95_inf:Q", y2="ic_95_sup:Q")
        futuro = alt.Chart(p).mark_line(point=alt.OverlayMarkDef(color=ROJO, shape="square", size=60), color=ROJO,
                                        strokeDash=[6, 4], strokeWidth=3).encode(
            x="anio:Q", y="pronostico:Q", tooltip=[alt.Tooltip("anio:Q", format="d"),
                                                   alt.Tooltip("pronostico:Q", format=".2f")])
        st.altair_chart(banda + obs + futuro, width="stretch")
        st.caption(f"El modelo sobreestimó {int((errores.error < 0).sum())} de los últimos 5 años: "
                   f"{pron.pronostico_2024:.2f} min es un techo.")
    with der.container(border=True):
        st.markdown("**El pronóstico según los años que mira el modelo**")
        ventanas = leer(13, "ventanas.csv")
        opciones = {v.replace("anios", "años"): v for v in ventanas.ventana}
        eleccion = st.segmented_control("Ventana", list(opciones), default="10 años", required=True,
                                        label_visibility="collapsed")
        fila = ventanas[ventanas.ventana == opciones[eleccion]].iloc[0]
        tramo = serie.tail(int(fila.n))
        m, b = np.polyfit(tramo.anio, tramo.duracion_min, 1)
        recta = pd.DataFrame({"anio": np.arange(tramo.anio.min(), 2025)})
        recta["y"] = m * recta.anio + b
        st.altair_chart(alt.Chart(serie).mark_line(color=GRIS).encode(
            x=alt.X("anio:Q", title=None, axis=alt.Axis(format="d")),
            y=alt.Y("duracion_min:Q", title="minutos", scale=alt.Scale(zero=False))) + alt.Chart(recta).mark_line(
            color=ROJO, strokeWidth=3).encode(x="anio:Q", y="y:Q"), width="stretch")
        st.metric("Pronóstico 2024 con esta ventana", f"{fila.pronostico_2024:.2f} min")

    with st.container(border=True):
        st.markdown("**El viernes se volvió el día de estreno** (% de los estrenos de cada año)")
        d = leer(15, "dia_de_estreno.csv")
        d["otros días"] = d[["lunes", "miercoles", "jueves", "sabado", "domingo"]].sum(axis=1)
        d = d.melt(id_vars="anio", value_vars=["viernes", "martes", "otros días"], var_name="dia", value_name="p")
        lineas = alt.Chart(d).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("anio:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("p:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("dia:N", scale=alt.Scale(domain=["viernes", "martes", "otros días"],
                                                     range=[VERDE, AZUL, GRIS]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["anio", "dia", alt.Tooltip("p:Q", format=".1%")])
        regla = alt.Chart(pd.DataFrame({"anio": [2015]})).mark_rule(color=ROJO, strokeDash=[4, 4], size=2).encode(
            x="anio:O")
        st.altair_chart(lineas + regla, width="stretch")
        st.caption(f"Chi cuadrada p < 0.001 · {int(viernes.fechas_usadas):,} fechas exactas · la popularidad casi no "
                   f"cambia por estrenar en viernes (d = {viernes.d_cohen:.2f})")


# TITULOS
def titulos():
    st.title("El título revela más que el audio")
    st.caption("Análisis de texto de los títulos y comparación de versiones de la misma canción")
    todas = versiones.loc["todas"]
    tarjetas([("Títulos con etiqueta", f"{texto.titulos_con_marcador:.1%}", "feat, remix, radio edit, live..."),
              ("Popularidad con feat", f"{pop.media_con.feat:.1f}", f"{pop.media_con.feat - pop.media_sin.feat:+.1f} "
               f"vs. sin feat"),
              ("Pares de la misma canción", f"{int(todas.pares):,}", "mismo artista, otra versión"),
              ("Gana la versión original", f"{todas.gana_principal:.1%}", f"mediana {todas.mediana_diferencia:+.0f} "
               f"puntos")])

    izq, der = st.columns([3, 2])
    with izq.container(border=True):
        st.markdown("**Nube de palabras de los títulos**")
        st.image(grafica(14, "01_nube.png"), width="stretch")
    with der.container(border=True):
        st.markdown("**Las 15 palabras más usadas**")
        top = leer(14, "frecuencias.csv").head(15)
        top["tipo"] = np.where(top.palabra.isin(marcas.marcador.unique()), "etiqueta de producción", "contenido")
        st.altair_chart(alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("frecuencia:Q", title=None), y=alt.Y("palabra:N", sort="-x", title=None),
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["etiqueta de producción", "contenido"],
                                                      range=[ROJO, VERDE]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["palabra", "frecuencia"]), width="stretch")

    izq, der = st.columns(2)
    with izq.container(border=True):
        st.markdown("**La original contra su propia versión alternativa** (% de veces que gana la original)")
        v = versiones.drop("todas").reset_index().replace({"tipo": {"acustica": "acústica"}})
        barras = alt.Chart(v).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("gana_principal:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tipo:N", sort="-x", title=None), color=alt.Color("tipo:N", scale=COLOR_TIPO, legend=None),
            opacity=alt.when(alt.datum.significativo).then(alt.value(1)).otherwise(alt.value(0.45)),
            tooltip=["tipo", "pares", alt.Tooltip("gana_principal:Q", format=".1%"), alt.Tooltip("p:Q", format=".3f")])
        mitad = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color="#191414", strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(barras + mitad, width="stretch")
        st.caption(f"Wilcoxon p < 0.001 · barras claras no significativas (remaster p = "
                   f"{versiones.loc['remaster'].p:.2f})")
    with der.container(border=True):
        st.markdown("**Cada punto es una canción** (arriba de la diagonal gana la original)")
        pares = leer(15, "versiones_pares.csv").replace({"tipo": {"acustica": "acústica"}})
        puntos = alt.Chart(pares).mark_circle(size=22, opacity=0.6).encode(
            x=alt.X("pop_variante:Q", title="versión alternativa"), y=alt.Y("pop_principal:Q", title="original"),
            color=alt.Color("tipo:N", scale=COLOR_TIPO, legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["artista", "titulo_principal", "titulo_variante", "pop_principal", "pop_variante"])
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]})).mark_line(color="#191414").encode(
            x="x:Q", y="y:Q")
        st.altair_chart(puntos + diagonal, width="stretch")


# CONCLUSIONES
def conclusiones():
    st.title("Qué haría con esto una disquera")
    st.caption("Recomendaciones a partir de los hallazgos")
    a0 = artista.iloc[0]
    recomendaciones = [
        (":material/person:", "Invertir en el artista, no en el sonido",
         f"R² del artista {a0.r2_artista_fuera_de_muestra:.3f} contra {a0.r2_audio_prueba:.3f} del audio"),
        (":material/groups:", "Apostar por colaboraciones",
         f"Con feat: {pop.media_con.feat:.1f} de popularidad contra {pop.media_sin.feat:.1f}"),
        (":material/star:", "Promocionar la versión original",
         f"Le gana a su remix el {versiones.loc['remix'].gana_principal:.0%} de las veces"),
        (":material/timer:", "Canciones de unos tres minutos",
         f"Pronóstico 2024: {pron.pronostico_2024:.2f} min como máximo"),
        (":material/category:", "No clasificar el género con el audio",
         f"El KNN se equivoca en {1 - knn.exactitud_prueba:.0%} de las canciones"),
        (":material/calendar_month:", "Estrenar en viernes casi no da ventaja",
         f"Es la norma ({viernes.viernes_2016_2019:.0%}), pero d = {viernes.d_cohen:.2f}"),
    ]
    for fila in (recomendaciones[:3], recomendaciones[3:]):
        for col, (icono, titulo, evidencia) in zip(st.columns(3), fila):
            with col.container(border=True):
                st.markdown(f"**{icono} {titulo}**")
                st.caption(evidencia)
    izq, der = st.columns([2, 3])
    with izq.container(border=True):
        st.markdown("**Limitaciones**")
        st.markdown("- Correlación, no causa\n- Playlists de 2020: más canciones recientes\n"
                    "- Popularidad de Spotify en un solo momento")
    with der.container(border=True):
        st.markdown("**Técnicas usadas**")
        st.dataframe(pd.DataFrame({
            "técnica": ["Regresión lineal múltiple", "Clasificación KNN", "Clustering k-means", "Serie de tiempo",
                        "Análisis de texto", "Pruebas de hipótesis"],
            "resultado clave": [f"R² = {reg.R2:.3f}", f"exactitud {knn.exactitud_prueba:.1%}",
                                f"Rand ajustado {km.rand_ajustado_con_k_igual_generos:.3f}",
                                f"{pron.pronostico_2024:.2f} min en 2024",
                                f"{texto.titulos_con_marcador:.1%} con etiqueta",
                                "Wilcoxon y chi cuadrada, p < 0.001"]}), hide_index=True, width="stretch")


pagina = st.navigation([
    st.Page(inicio, title="Inicio", icon=":material/home:", default=True),
    st.Page(audio, title="Audio y artista", icon=":material/graphic_eq:"),
    st.Page(genero, title="Género", icon=":material/category:"),
    st.Page(tiempo, title="Tiempo", icon=":material/timer:"),
    st.Page(titulos, title="Títulos", icon=":material/text_fields:"),
    st.Page(conclusiones, title="Conclusiones", icon=":material/flag:"),
], position="top")
pagina.run()
