import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("graficas", exist_ok=True)

D = "../Semana 5/dataframes"
df = pd.read_csv(f"{D}/canciones.csv")
por_genero = pd.read_csv(f"{D}/por_genero.csv")

audio = ["danceability", "energy", "valence", "acousticness", "speechiness", "liveness", "instrumentalness"]
generos = sorted(df["playlist_genre"].unique())
colores = dict(zip(generos, plt.cm.Set2(np.linspace(0, 1, len(generos)))))
muestra = df.sample(4000, random_state=42)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


# HISTOGRAMAS
variables = ["track_popularity", "danceability", "energy", "loudness", "tempo",
             "duracion_min", "valence", "acousticness", "release_year"]
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, c in zip(axes.flat, variables):
    ax.hist(df[c], bins=30, color="#1DB954", edgecolor="white", linewidth=0.5)
    ax.axvline(df[c].mean(), color="#c0392b", ls="--", lw=1.5, label=f"media {df[c].mean():.2f}")
    ax.axvline(df[c].median(), color="#2c3e50", ls=":", lw=1.5, label=f"mediana {df[c].median():.2f}")
    ax.set_title(c, fontweight="bold")
    ax.legend(fontsize=7)
fig.suptitle("Histogramas", fontsize=15, fontweight="bold")
guardar(fig, "01_histogramas")


# PASTEL
categorias = ["playlist_genre", "decada", "mode", "key"]
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for ax, c in zip(axes, categorias):
    v = df[c].value_counts()
    if c != "playlist_genre":
        v = v.sort_index()
    chicos = v[v / v.sum() < 0.03]
    if len(chicos) > 1:
        v = pd.concat([v[v / v.sum() >= 0.03], pd.Series({"otros": chicos.sum()})])
    ax.pie(v, labels=[str(i) for i in v.index], autopct="%1.1f%%", startangle=90,
           colors=plt.cm.Set3(np.linspace(0, 1, len(v))),
           wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops=dict(fontsize=8))
    ax.set_title(c, fontweight="bold")
fig.suptitle("Distribucion por categoria", fontsize=15, fontweight="bold")
guardar(fig, "02_pastel")


# CAJA
fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1.3, 1]})
b = axes[0].boxplot([df[c] for c in audio], tick_labels=audio, patch_artist=True, showfliers=False)
for p, col in zip(b["boxes"], plt.cm.Set2(np.linspace(0, 1, len(audio)))):
    p.set_facecolor(col)
axes[0].set_title("Caracteristicas de audio", fontweight="bold")
axes[0].tick_params(axis="x", rotation=30)

b = axes[1].boxplot([df.loc[df["playlist_genre"] == g, "track_popularity"] for g in generos],
                    tick_labels=generos, patch_artist=True, showfliers=False)
for p, g in zip(b["boxes"], generos):
    p.set_facecolor(colores[g])
axes[1].set_title("Popularidad por genero", fontweight="bold")
guardar(fig, "03_caja")


# DISPERSION
pares = [("energy", "loudness"), ("danceability", "valence"), ("acousticness", "energy"),
         ("tempo", "energy"), ("duracion_min", "track_popularity"), ("release_year", "track_popularity")]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (x, y) in zip(axes.flat, pares):
    ax.scatter(muestra[x], muestra[y], s=8, alpha=0.25, color="#1DB954", edgecolors="none")
    m, b0 = np.polyfit(muestra[x], muestra[y], 1)
    xs = np.linspace(muestra[x].min(), muestra[x].max(), 50)
    ax.plot(xs, m * xs + b0, color="#c0392b", lw=1.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"r = {df[x].corr(df[y]):.3f}", fontweight="bold")
fig.suptitle("Dispersion", fontsize=15, fontweight="bold")
guardar(fig, "04_dispersion")


# BARRAS
metricas = ["conteo", "pop_media", "energy_media", "dance_media", "tempo_media"]
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, m in zip(axes, metricas):
    ax.bar(por_genero["playlist_genre"], por_genero[m],
           color=[colores[g] for g in por_genero["playlist_genre"]], edgecolor="black", linewidth=0.6)
    ax.set_title(m, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Metricas por genero", fontsize=15, fontweight="bold")
guardar(fig, "05_barras")

print(f"\n{len(os.listdir('graficas'))} graficas en graficas/")
