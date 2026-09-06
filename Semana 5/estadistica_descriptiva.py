import os

import numpy as np
import pandas as pd

os.makedirs("dataframes", exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

df = pd.read_csv("../spotify_songs_limpio.csv", parse_dates=["track_album_release_date"])
raw = pd.read_csv("../spotify_songs.csv")
raw = raw[raw["track_id"].isin(df["track_id"])].reset_index(drop=True)

num = ["track_popularity", "num_playlists", "danceability", "energy", "loudness",
       "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
       "tempo", "duration_ms", "release_year"]

df["decada"] = (df["release_year"] // 10 * 10).astype("Int64")
df["duracion_min"] = df["duration_ms"] / 60000


print("\n1. ESTADISTICA DESCRIPTIVA\n")

resumen = pd.DataFrame({
    "conteo": df[num].count(),
    "sumatoria": df[num].sum(),
    "min": df[num].min(),
    "max": df[num].max(),
    "media": df[num].mean(),
    "mediana": df[num].median(),
    "moda": [df[c].mode()[0] for c in num],
    "varianza": df[num].var(),
    "desv_est": df[num].std(),
    "curtosis": df[num].kurt(),
    "sesgo": df[num].skew(),
})
print(resumen.round(3).to_string())

cat = ["track_artist", "playlist_genre", "playlist_subgenre", "key", "mode"]
categoricas = pd.DataFrame({
    "conteo": [df[c].count() for c in cat],
    "distintos": [df[c].nunique() for c in cat],
    "moda": [df[c].mode()[0] for c in cat],
}, index=cat)
print("\nCategoricas:")
print(categoricas.to_string())


print("\n\n2. ALGEBRA RELACIONAL\n")

transpuesta = df[num].head(5).T
print("Transposicion:")
print(transpuesta.round(3).to_string())

seleccion = df[(df["track_popularity"] >= 80) & (df["playlist_genre"] == "latin")]
print(f"\nSeleccion (pop>=80 y genero=latin): {len(seleccion)} de {len(df)}")
print(seleccion[["track_name", "track_artist", "track_popularity"]].head().to_string(index=False))

proyeccion = df[["track_name", "track_artist", "playlist_genre", "track_popularity"]]
print("\nProyeccion:")
print(proyeccion.head().to_string(index=False))

top = df[df["track_popularity"] >= 95]
flop = df[df["track_popularity"] <= 5]
union = pd.concat([top, flop]).drop_duplicates("track_id")
print(f"\nUnion: top={len(top)} flop={len(flop)} union={len(union)}")

e = set(df.loc[df["energy"] > 0.8, "track_id"])
b = set(df.loc[df["danceability"] > 0.8, "track_id"])
print(f"Interseccion: {len(e & b)}   Diferencia: {len(e - b)}")

artista = df[["track_artist"]].drop_duplicates().sort_values("track_artist").reset_index(drop=True)
artista["artista_id"] = "AR" + (artista.index + 1).astype(str).str.zfill(5)

genero = df[["playlist_genre"]].drop_duplicates().sort_values("playlist_genre").reset_index(drop=True)
genero["genero_id"] = "G" + (genero.index + 1).astype(str).str.zfill(2)

subgenero = df[["playlist_subgenre", "playlist_genre"]].drop_duplicates("playlist_subgenre").sort_values("playlist_subgenre").reset_index(drop=True)
subgenero["subgenero_id"] = "SG" + (subgenero.index + 1).astype(str).str.zfill(2)
subgenero = subgenero.merge(genero, on="playlist_genre").drop(columns=["playlist_genre"])

playlist = raw[["playlist_id", "playlist_name"]].drop_duplicates("playlist_id").reset_index(drop=True)
playlist["playlist_subgenre"] = playlist["playlist_id"].map(
    raw.groupby("playlist_id")["playlist_subgenre"].agg(lambda s: s.mode()[0]))
playlist = playlist.merge(subgenero, on="playlist_subgenre").drop(columns=["playlist_subgenre", "genero_id"])

cancion = df[["track_id", "track_name", "track_popularity", "track_artist"]].merge(artista, on="track_artist").drop(columns=["track_artist"])

cancion_playlist = raw[["track_id", "playlist_id"]].drop_duplicates().reset_index(drop=True)

join = (cancion_playlist
        .merge(cancion, on="track_id")
        .merge(playlist, on="playlist_id")
        .merge(artista, on="artista_id")
        .merge(subgenero, on="subgenero_id")
        .merge(genero, on="genero_id"))
print(f"\nJoin de 5 tablas: {len(join)} filas x {len(join.columns)} columnas")
print(join[["track_name", "track_artist", "playlist_name", "playlist_genre"]].head().to_string(index=False))

por_genero = df.groupby("playlist_genre").agg(
    conteo=("track_id", "count"),
    suma_min=("duracion_min", "sum"),
    pop_min=("track_popularity", "min"),
    pop_max=("track_popularity", "max"),
    pop_media=("track_popularity", "mean"),
    pop_moda=("track_popularity", lambda s: s.mode()[0]),
    pop_var=("track_popularity", "var"),
    pop_desv=("track_popularity", "std"),
    pop_curtosis=("track_popularity", lambda s: s.kurt()),
    energy_media=("energy", "mean"),
    dance_media=("danceability", "mean"),
    tempo_media=("tempo", "mean"),
).round(3)
print("\nAgrupacion por genero:")
print(por_genero.to_string())

por_subgenero = df.groupby(["playlist_genre", "playlist_subgenre"]).agg(
    conteo=("track_id", "count"),
    pop_media=("track_popularity", "mean"),
    energy_media=("energy", "mean"),
    dance_media=("danceability", "mean"),
).round(3)

por_decada = df.groupby("decada").agg(
    conteo=("track_id", "count"),
    pop_media=("track_popularity", "mean"),
    energy_media=("energy", "mean"),
    dance_media=("danceability", "mean"),
    duracion_media=("duracion_min", "mean"),
).round(3)
print("\nAgrupacion por decada:")
print(por_decada.to_string())

genero_decada = df.pivot_table(index="playlist_genre", columns="decada",
                               values="track_popularity", aggfunc="mean").round(1)
print("\nGenero x decada (media de popularidad):")
print(genero_decada.to_string())

top_artistas = df.groupby("track_artist").agg(
    canciones=("track_id", "count"),
    pop_media=("track_popularity", "mean"),
).sort_values("canciones", ascending=False).head(20).round(2)

correlacion = df[num].corr().round(3)


print("\n\n3. DATOS AGRUPADOS\n")


def frecuencias(serie):
    x = serie.dropna().astype(float).values
    n = len(x)
    k = int(round(1 + 3.322 * np.log10(n)))
    bordes = np.linspace(x.min(), x.max(), k + 1)
    amp = (x.max() - x.min()) / k

    idx = np.clip(np.searchsorted(bordes, x, side="right") - 1, 0, k - 1)

    t = pd.DataFrame({"li": bordes[:-1], "ls": bordes[1:]})
    t["fi"] = np.bincount(idx, minlength=k)
    t["xi"] = (t["li"] + t["ls"]) / 2
    t["fi_xi"] = t["fi"] * t["xi"]
    t["Fi"] = t["fi"].cumsum()
    t["hi_%"] = (t["fi"] / n * 100).round(3)
    t["Hi_%"] = (t["Fi"] / n * 100).round(3)

    media = t["fi_xi"].sum() / n
    var = (t["fi"] * (t["xi"] - media) ** 2).sum() / (n - 1)
    desv = np.sqrt(var)

    im = int(np.argmax(t["Fi"].values >= n / 2))
    fant = t["Fi"].values[im - 1] if im > 0 else 0
    mediana = t["li"].values[im] + ((n / 2 - fant) / t["fi"].values[im]) * amp

    imo = int(np.argmax(t["fi"].values))
    fm = t["fi"].values[imo]
    f0 = t["fi"].values[imo - 1] if imo > 0 else 0
    f2 = t["fi"].values[imo + 1] if imo < k - 1 else 0
    d1, d2 = fm - f0, fm - f2
    moda = t["li"].values[imo] + (d1 / (d1 + d2)) * amp if d1 + d2 > 0 else t["xi"].values[imo]

    sesgo = (t["fi"] * (t["xi"] - media) ** 3).sum() / (n * desv ** 3)
    curt = (t["fi"] * (t["xi"] - media) ** 4).sum() / (n * desv ** 4) - 3

    cierre = ["[{:.3f} - {:.3f})".format(a, b) for a, b in zip(t["li"][:-1], t["ls"][:-1])]
    cierre.append("[{:.3f} - {:.3f}]".format(t["li"].iloc[-1], t["ls"].iloc[-1]))
    t.insert(0, "clase", cierre)

    metricas = {"n": n, "clases": k, "amplitud": round(amp, 4), "min": x.min(), "max": x.max(),
                "sumatoria_fi": int(t["fi"].sum()), "sumatoria_fi_xi": round(t["fi_xi"].sum(), 3),
                "media": round(media, 4), "mediana": round(mediana, 4), "moda": round(moda, 4),
                "varianza": round(var, 4), "desv_est": round(desv, 4),
                "sesgo": round(sesgo, 4), "curtosis": round(curt, 4)}
    return t, metricas


tablas_frecuencia = {}
metricas_agrupadas = {}
for c in ["track_popularity", "danceability", "energy", "tempo", "duration_ms", "loudness", "valence"]:
    t, m = frecuencias(df[c])
    tablas_frecuencia[c] = t
    metricas_agrupadas[c] = m
    print(f"--- {c} ---")
    print(t[["clase", "xi", "fi", "Fi", "hi_%", "Hi_%", "fi_xi"]].to_string(index=False))
    for kk, vv in m.items():
        print(f"  {kk:16s} {vv}")
    print()

metricas_agrupadas = pd.DataFrame(metricas_agrupadas).T
print("Resumen de datos agrupados:")
print(metricas_agrupadas.to_string())


salidas = {
    "resumen_descriptivo": resumen.reset_index().rename(columns={"index": "variable"}),
    "resumen_categoricas": categoricas.reset_index().rename(columns={"index": "variable"}),
    "por_genero": por_genero.reset_index(),
    "por_subgenero": por_subgenero.reset_index(),
    "por_decada": por_decada.reset_index(),
    "genero_decada": genero_decada.reset_index(),
    "top_artistas": top_artistas.reset_index(),
    "correlacion": correlacion.reset_index().rename(columns={"index": "variable"}),
    "metricas_agrupadas": metricas_agrupadas.reset_index().rename(columns={"index": "variable"}),
    "canciones": df,
}
for c, t in tablas_frecuencia.items():
    salidas[f"frecuencias_{c}"] = t

for nombre, tabla in salidas.items():
    tabla.to_csv(f"dataframes/{nombre}.csv", index=False)

print(f"\n{len(salidas)} dataframes guardados en dataframes/")
for nombre, tabla in salidas.items():
    print(f"  {nombre:32s} {tabla.shape[0]:7d} x {tabla.shape[1]}")
