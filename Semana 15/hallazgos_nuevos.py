import os
import re

import numpy as np
import pandas as pd
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")
crudo = pd.read_csv("../spotify_songs.csv", usecols=["track_id", "track_album_release_date"])
audio = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "liveness", "valence", "tempo", "duracion_min", "release_year"]
version = (r"\b(remix(?:es)?|mix(?:es)?|edit|version|remaster(?:ed)?|live|acoustic|radio|extended|"
           r"instrumental|demo|mono|stereo|en vivo)\b")
nombres = {"remix": "remix", "remixes": "remix", "mix": "remix", "mixes": "remix", "extended": "remix",
           "edit": "radio edit", "radio": "radio edit", "remaster": "remaster", "remastered": "remaster",
           "live": "en vivo", "en vivo": "en vivo", "acoustic": "acustica"}


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


def titulo_base(t):
    t = re.sub(r"\s*[\(\[][^\)\]]*\b(feat|ft|with)\b[^\)\]]*[\)\]]", "", str(t).lower())
    t = re.split(r"\s+-\s+|\s*[\(\[]", t)[0]
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def tipo_version(t):
    t = str(t).lower()
    cola = " ".join(re.findall(r"[\(\[]([^\)\]]*)[\)\]]", t) + re.split(r"\s+-\s+", t)[1:])
    m = re.search(version, cola)
    return "principal" if m is None else nombres.get(m.group(1), "otra")


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# VERSIONES
df["base"] = df.track_name.map(titulo_base)
df["tipo"] = df.track_name.map(tipo_version)
df["artista"] = df.track_artist.str.lower().str.strip()

filas = []
for (a, b), g in df[df.base != ""].groupby(["artista", "base"]):
    principal = g[g.tipo == "principal"]
    if principal.empty or (g.tipo == "principal").all():
        continue
    for t, v in g[g.tipo != "principal"].groupby("tipo"):
        filas.append({"artista": a, "cancion": b, "tipo": t,
                      "titulo_principal": principal.track_name.iloc[0], "titulo_variante": v.track_name.iloc[0],
                      "pop_principal": principal.track_popularity.mean(), "pop_variante": v.track_popularity.mean()})
pares = pd.DataFrame(filas)
pares["diferencia"] = pares.pop_principal - pares.pop_variante
pares.to_csv("resultados/versiones_pares.csv", index=False)

filas = []
for t, g in list(pares.groupby("tipo")) + [("todas", pares)]:
    w = stats.wilcoxon(g.diferencia)
    filas.append({"tipo": t, "pares": len(g), "gana_principal": (g.diferencia > 0).mean(),
                  "empates": (g.diferencia == 0).mean(), "gana_variante": (g.diferencia < 0).mean(),
                  "mediana_diferencia": g.diferencia.median(), "media_diferencia": g.diferencia.mean(),
                  "wilcoxon": w.statistic, "p": w.pvalue, "significativo": w.pvalue < 0.05})
versiones = pd.DataFrame(filas)
versiones.to_csv("resultados/versiones_resumen.csv", index=False)
print(f"{len(pares)} pares de la misma cancion ({pares[['artista', 'cancion']].drop_duplicates().shape[0]} "
      f"canciones con al menos una version alternativa)")
print(versiones.round(4).to_string(index=False))


# ARTISTA
cuenta = df.groupby("artista").track_popularity.agg(["sum", "count"])
d = df.join(cuenta, on="artista")
filas = []
rng = np.random.default_rng(42)
for minimo in [2, 3, 5, 10]:
    s = d[d["count"] >= minimo].copy()
    s["por_artista"] = (s["sum"] - s.track_popularity) / (s["count"] - 1)
    orden = rng.permutation(len(s))
    ent, pru = orden[:int(len(s) * 0.8)], orden[int(len(s) * 0.8):]
    A = np.column_stack([np.ones(len(s)), s[audio].values])
    beta = np.linalg.lstsq(A[ent], s.track_popularity.values[ent], rcond=None)[0]
    y = s.track_popularity.values
    filas.append({"minimo_canciones": minimo, "canciones": len(s), "proporcion_del_total": len(s) / len(df),
                  "artistas": s.artista.nunique(),
                  "r2_audio_prueba": r2(y[pru], A[pru] @ beta),
                  "r2_artista_fuera_de_muestra": r2(y[pru], s.por_artista.values[pru]),
                  "r_artista": np.corrcoef(y, s.por_artista)[0, 1]})
    if minimo == 2:
        ejemplo = s[["track_name", "track_artist", "track_popularity", "por_artista"]]
artista = pd.DataFrame(filas)
artista["veces_mejor"] = artista.r2_artista_fuera_de_muestra / artista.r2_audio_prueba
artista.to_csv("resultados/artista_vs_audio.csv", index=False)
print("\n", artista.round(4).to_string(index=False))


# ESTRENOS
crudo = crudo.drop_duplicates("track_id")
crudo["fecha_completa"] = crudo.track_album_release_date.astype(str).str.len() == 10
f = df.merge(crudo[["track_id", "fecha_completa"]], on="track_id")
f["fecha"] = pd.to_datetime(f.track_album_release_date)
quitadas = (~f.fecha_completa | (f.fecha.dt.day == 1)).sum()
f = f[f.fecha_completa & (f.fecha.dt.day != 1)].copy()
f["dia"] = f.fecha.dt.dayofweek
f["anio"] = f.fecha.dt.year
dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

por_anio = (f[f.anio.between(2005, 2019)].groupby("anio").dia.value_counts(normalize=True)
            .unstack(fill_value=0).rename(columns=dict(enumerate(dias))))
por_anio["canciones"] = f[f.anio.between(2005, 2019)].groupby("anio").size()
por_anio.reset_index().to_csv("resultados/dia_de_estreno.csv", index=False)

antes, despues = f[f.anio.between(2010, 2014)], f[f.anio.between(2016, 2019)]
tabla = np.array([[(antes.dia == 4).sum(), (antes.dia != 4).sum()],
                  [(despues.dia == 4).sum(), (despues.dia != 4).sum()]])
chi2, p_chi, _, _ = stats.chi2_contingency(tabla)
vie, otro = despues[despues.dia == 4].track_popularity, despues[despues.dia != 4].track_popularity
u, p_u = stats.mannwhitneyu(vie, otro)
s = np.sqrt(((len(vie) - 1) * vie.var() + (len(otro) - 1) * otro.var()) / (len(vie) + len(otro) - 2))
estrenos = pd.DataFrame([{"fechas_usadas": len(f), "fechas_quitadas": quitadas,
                          "viernes_2010_2014": (antes.dia == 4).mean(), "viernes_2016_2019": (despues.dia == 4).mean(),
                          "martes_2010_2014": (antes.dia == 1).mean(), "martes_2016_2019": (despues.dia == 1).mean(),
                          "chi2": chi2, "p_chi2": p_chi,
                          "popularidad_viernes": vie.mean(), "popularidad_otro_dia": otro.mean(),
                          "d_cohen": (vie.mean() - otro.mean()) / s, "p_mann_whitney": p_u}])
estrenos.to_csv("resultados/viernes.csv", index=False)
print("\n", estrenos.round(4).T.to_string(header=False))

resumen = pd.DataFrame([{"pares_versiones": len(pares),
                         "gana_principal": versiones[versiones.tipo == "todas"].gana_principal.iloc[0],
                         "mediana_diferencia": versiones[versiones.tipo == "todas"].mediana_diferencia.iloc[0],
                         "p_versiones": versiones[versiones.tipo == "todas"].p.iloc[0],
                         "r2_artista": artista.r2_artista_fuera_de_muestra[0],
                         "r2_audio_mismas_canciones": artista.r2_audio_prueba[0],
                         "viernes_antes": estrenos.viernes_2010_2014[0],
                         "viernes_despues": estrenos.viernes_2016_2019[0]}])
resumen.to_csv("resultados/resumen.csv", index=False)


# GRAFICAS
colores = {"remix": "#c0392b", "radio edit": "#e67e22", "en vivo": "#8e44ad", "acustica": "#16a085",
           "remaster": "#2980b9", "otra": "#95a5a6"}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
v = versiones[versiones.tipo != "todas"].sort_values("gana_principal")
axes[0].barh(v.tipo, v.gana_principal * 100, color=[colores[t] for t in v.tipo],
             edgecolor="black", linewidth=0.6)
axes[0].axvline(50, color="black", ls="--", lw=1.2)
for i, (g_, n, p_) in enumerate(zip(v.gana_principal, v.pares, v.p)):
    axes[0].text(g_ * 100 + 1, i, f"{g_:.0%}  (n={n}{', *' if p_ < 0.05 else ''})", va="center", fontsize=9)
axes[0].set_xlim(0, 100)
axes[0].set_xlabel("% de pares donde la version principal es mas popular")
axes[0].set_title("La misma cancion contra su propia version alternativa", fontweight="bold")
axes[0].grid(alpha=0.3, axis="x")

for t, g in pares.groupby("tipo"):
    axes[1].scatter(g.pop_variante, g.pop_principal, s=14, alpha=0.6, color=colores[t], label=t,
                    edgecolors="none")
axes[1].plot([0, 100], [0, 100], color="black", lw=1.2)
axes[1].set_xlabel("popularidad de la version alternativa")
axes[1].set_ylabel("popularidad de la version principal")
axes[1].set_title("Arriba de la diagonal = gana la principal", fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.suptitle(f"{len(pares)} pares de la misma cancion y el mismo artista", fontsize=15, fontweight="bold")
guardar(fig, "01_versiones")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
x = np.arange(len(artista))
axes[0].bar(x - 0.2, artista.r2_audio_prueba, 0.4, color="#95a5a6", edgecolor="black", label="audio (11 variables)")
axes[0].bar(x + 0.2, artista.r2_artista_fuera_de_muestra, 0.4, color="#1DB954", edgecolor="black",
            label="promedio de las otras canciones del artista")
for i, (a, b) in enumerate(zip(artista.r2_audio_prueba, artista.r2_artista_fuera_de_muestra)):
    axes[0].text(i - 0.2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
    axes[0].text(i + 0.2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(x, [f"artistas con\n>= {m} canciones" for m in artista.minimo_canciones])
axes[0].set_ylabel("R2 en canciones que no se usaron para predecir")
axes[0].set_title("Que predice mejor la popularidad", fontweight="bold")
axes[0].legend(fontsize=9, loc="upper left")
axes[0].set_ylim(0, artista.r2_artista_fuera_de_muestra.max() * 1.25)
axes[0].grid(alpha=0.3, axis="y")

muestra = ejemplo.sample(4000, random_state=42)
axes[1].scatter(muestra.por_artista, muestra.track_popularity, s=8, alpha=0.25, color="#1DB954", edgecolors="none")
axes[1].plot([0, 100], [0, 100], color="#c0392b", lw=1.5)
axes[1].set_xlabel("popularidad media de las OTRAS canciones del artista")
axes[1].set_ylabel("popularidad de la cancion")
axes[1].set_title(f"r = {artista.r_artista[0]:.3f}", fontweight="bold")
axes[1].grid(alpha=0.3)
fig.suptitle("Saber quien canta dice mas que saber como suena", fontsize=15, fontweight="bold")
guardar(fig, "02_artista")

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(por_anio.index, por_anio.viernes * 100, "o-", color="#1DB954", lw=2.5, ms=6, label="viernes")
ax.plot(por_anio.index, por_anio.martes * 100, "o-", color="#2980b9", lw=2, ms=5, label="martes")
otros = por_anio[["lunes", "miercoles", "jueves", "sabado", "domingo"]].sum(axis=1)
ax.plot(por_anio.index, otros * 100, "o--", color="#95a5a6", lw=1.5, ms=4, label="los otros 5 dias juntos")
ax.axvline(2015, color="#c0392b", ls=":", lw=2)
ax.text(2015.1, 90, "2015: el estreno global\npasa a ser en viernes", color="#c0392b", fontsize=10, va="top")
ax.set_ylim(0, 100)
ax.set_xticks(por_anio.index)
ax.set_ylabel("% de los lanzamientos del anio")
ax.set_title(f"Dia de la semana en que salieron las canciones ({len(f):,} fechas exactas)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
guardar(fig, "03_estrenos")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
