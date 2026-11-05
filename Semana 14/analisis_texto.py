import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")
df = df.drop_duplicates(["track_name", "track_artist"]).reset_index(drop=True)

campo = "track_name"
vacias = set("""a al algo algun alguna alguno ante antes aqui asi aun cada como con contra cual
cuando de del desde donde dos el ella ellas ellos en entre era eran es esa ese eso esta estan
este esto estos ha hace hacia han hasta hay la las le les lo los mas me mi mis mucho muy nada
ni no nos o os otra otro para pero poco por porque que quien se sea segun ser si sin so sobre
solo son su sus tan te ti tu tus un una uno unos y ya yo
about after all also am an and any are as at be because been before being but by can cant did
do does doing dont down for from get got had has have he her here hers him his how i if im in
into is it its ive just me more most my no nor not now of off on once only or other our out
over own re s same she should so some such t than that the their them then there these they
this those through to too up ve very was we were what when where which while who whom why
will with would you your yours""".split())
marcadores = ["feat", "remix", "mix", "edit", "radio", "version", "remastered", "remaster",
              "original", "live", "acoustic"]
generos = sorted(df.playlist_genre.unique())
variables = ["duracion_min", "energy", "danceability", "track_popularity"]
tono_nube = ["#1DB954", "#117a37", "#2980b9", "#8e44ad", "#c0392b", "#e67e22", "#16a085", "#2c3e50"]


def guardar(fig, nombre):
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"{nombre}.png")


def limpiar(texto):
    t = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def separar(texto):
    return [p for p in re.findall(r"[a-z0-9']+", limpiar(texto)) if len(p) > 1]


def contar(listas_tokens, quitar_vacias=True):
    c = Counter()
    for l in listas_tokens:
        c.update(p for p in l if not (quitar_vacias and p in vacias))
    return c


def ajustar(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2


# TEXTO
listas = [separar(t) for t in df[campo]]
conjuntos = [set(l) for l in listas]
tokens = [p for l in listas for p in l]
frec_todo = Counter(tokens)
frec = contar(listas)
orden_todo = [p for p, _ in frec_todo.most_common()]
largos = np.array([len(p) for p in tokens])

print(f"{len(df)} canciones unicas, {len(tokens)} palabras en los titulos, "
      f"vocabulario {len(frec_todo)}")
print(f"{sum(1 for v in frec_todo.values() if v == 1)} palabras aparecen una sola vez")

tabla = pd.DataFrame(frec.most_common(), columns=["palabra", "frecuencia"])
tabla["proporcion"] = tabla.frecuencia / tabla.frecuencia.sum()
tabla["titulos"] = [sum(p in s for s in conjuntos) for p in tabla.palabra[:60]] \
                   + [np.nan] * (len(tabla) - 60)
tabla.to_csv("resultados/frecuencias.csv", index=False)
print("\n", tabla.head(20).to_string(index=False))


# ZIPF
rango = np.arange(1, len(frec_todo) + 1)
cuentas = np.array(sorted(frec_todo.values(), reverse=True), dtype=float)
b_zipf, r2_zipf = ajustar(np.log10(rango), np.log10(cuentas))

n_tokens = np.unique(np.linspace(100, len(tokens), 60).astype(int))
vistos, vocab = set(), []
for corte, previo in zip(n_tokens, np.concatenate([[0], n_tokens[:-1]])):
    vistos.update(tokens[previo:corte])
    vocab.append(len(vistos))
vocab = np.array(vocab, dtype=float)
b_heaps, r2_heaps = ajustar(np.log10(n_tokens), np.log10(vocab))

leyes = pd.DataFrame([{"ley": "Zipf", "formula": "log10(frec) = a + b*log10(rango)",
                       "a": b_zipf[0], "b": b_zipf[1], "r2": r2_zipf,
                       "interpretacion": f"exponente {-b_zipf[1]:.3f}"},
                      {"ley": "Heaps", "formula": "log10(vocab) = a + b*log10(tokens)",
                       "a": b_heaps[0], "b": b_heaps[1], "r2": r2_heaps,
                       "interpretacion": f"K={10 ** b_heaps[0]:.2f}, beta={b_heaps[1]:.3f}"}])
leyes.to_csv("resultados/leyes_zipf_heaps.csv", index=False)
print("\n", leyes.to_string(index=False))


# BIGRAMAS
bi = Counter()
for l in listas:
    bi.update(zip(l[:-1], l[1:]))
bigramas = pd.DataFrame([{"bigrama": f"{a} {b}", "frecuencia": f_} for (a, b), f_ in bi.most_common(40)])
bigramas.to_csv("resultados/bigramas.csv", index=False)
print("\n", bigramas.head(12).to_string(index=False))


# TFIDF
por_genero = {g: contar([l for l, x in zip(listas, df.playlist_genre) if x == g]) for g in generos}
doc_freq = Counter()
for g in generos:
    doc_freq.update(set(por_genero[g]))
total_global = sum(frec.values())
filas = []
for g in generos:
    total = sum(por_genero[g].values())
    for p, f_ in por_genero[g].items():
        if f_ < 10:
            continue
        idf = np.log(len(generos) / doc_freq[p])
        filas.append({"genero": g, "palabra": p, "frecuencia": f_, "tf": f_ / total,
                      "generos_con_la_palabra": doc_freq[p], "idf": idf, "tfidf": f_ / total * idf,
                      "razon_vs_global": (f_ / total) / (frec[p] / total_global)})
tfidf = pd.DataFrame(filas).sort_values(["genero", "tfidf"], ascending=[True, False])
tfidf.to_csv("resultados/tfidf_genero.csv", index=False)
distintivas = tfidf.groupby("genero").head(8)
print("\npalabras distintivas por genero (tf-idf, minimo 10 apariciones)")
for g in generos:
    d = distintivas[distintivas.genero == g]
    print(f"  {g:8s}", ", ".join(f"{r.palabra} ({r.frecuencia})" for r in d.itertuples()))


# MARCADORES
filas = []
for m in marcadores:
    tiene = np.array([m in s for s in conjuntos])
    for v in variables:
        con, sin = df[tiene][v].values, df[~tiene][v].values
        t_, p_ = stats.ttest_ind(con, sin, equal_var=False)
        s = np.sqrt((con.var(ddof=1) * (len(con) - 1) + sin.var(ddof=1) * (len(sin) - 1))
                    / (len(con) + len(sin) - 2))
        filas.append({"marcador": m, "variable": v, "n_con": len(con), "n_sin": len(sin),
                      "media_con": con.mean(), "media_sin": sin.mean(),
                      "diferencia": con.mean() - sin.mean(), "d_cohen": (con.mean() - sin.mean()) / s,
                      "t": t_, "p": p_, "significativo": p_ < 0.05})
marcas = pd.DataFrame(filas)
marcas.to_csv("resultados/marcadores.csv", index=False)
print("\n", marcas[marcas.variable == "duracion_min"].round(4).to_string(index=False))

resumen = pd.DataFrame([{"canciones": len(df), "palabras": len(tokens),
                         "vocabulario": len(frec_todo),
                         "hapax": sum(1 for v in frec_todo.values() if v == 1),
                         "riqueza_lexica": len(frec_todo) / len(tokens),
                         "palabras_por_titulo": len(tokens) / len(df),
                         "largo_medio_palabra": largos.mean(),
                         "palabras_sin_vacias": sum(frec.values()),
                         "proporcion_vacias": 1 - sum(frec.values()) / len(tokens),
                         "zipf_exponente": -b_zipf[1], "zipf_r2": r2_zipf,
                         "heaps_beta": b_heaps[1], "heaps_r2": r2_heaps,
                         "palabra_top": tabla.palabra[0], "frecuencia_top": tabla.frecuencia[0],
                         "titulos_con_marcador": np.mean([any(m in s for m in marcadores)
                                                          for s in conjuntos])}])
resumen.to_csv("resultados/resumen.csv", index=False)
print("\n", resumen.round(4).to_string(index=False))

perfil = pd.DataFrame([{"genero": g, "canciones": (df.playlist_genre == g).sum(),
                        "palabras": sum(por_genero[g].values()),
                        "vocabulario": len(por_genero[g]),
                        "palabras_por_titulo": sum(por_genero[g].values()) / (df.playlist_genre == g).sum(),
                        "top_1": por_genero[g].most_common(1)[0][0],
                        "top_3": ", ".join(p for p, _ in por_genero[g].most_common(3))}
                       for g in generos])
perfil.to_csv("resultados/perfil_genero.csv", index=False)
print("\n", perfil.to_string(index=False))


# NUBE
def medir(ax, palabra, tamano):
    t = ax.text(0, 0, palabra, fontsize=tamano, fontweight="bold", ha="left", va="bottom")
    caja = t.get_window_extent(ax.figure.canvas.get_renderer())
    t.remove()
    return caja.width, caja.height


def nube(ax, frecuencias, n=120, tam_max=95, tam_min=8, celda=3, semilla=42):
    ax.figure.canvas.draw()
    marco = ax.get_window_extent()
    ancho, alto = int(marco.width), int(marco.height)
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    rejilla = np.zeros((alto // celda + 2, ancho // celda + 2), dtype=bool)
    rng = np.random.default_rng(semilla)
    inicio = rng.uniform(0, 2 * np.pi)
    palabras = frecuencias.most_common(n)
    mayor = palabras[0][1]
    puestas = []
    for i, (palabra, f) in enumerate(palabras):
        tamano = tam_min + (tam_max - tam_min) * (f / mayor) ** 0.45
        w, h = medir(ax, palabra, tamano)
        w, h = w + 5, h + 5
        for paso in range(4000):
            angulo = inicio + 0.22 * paso
            radio = 1.6 * 0.22 * paso
            x = ancho / 2 + radio * np.cos(angulo) - w / 2
            y = alto / 2 + radio * np.sin(angulo) * alto / ancho - h / 2
            if radio > 1.5 * max(ancho, alto):
                break
            if x < 0 or y < 0 or x + w > ancho or y + h > alto:
                continue
            x0, y0 = int(x / celda), int(y / celda)
            x1, y1 = int((x + w) / celda) + 1, int((y + h) / celda) + 1
            if not rejilla[y0:y1, x0:x1].any():
                rejilla[y0:y1, x0:x1] = True
                ax.text(x + 2.5, y + 2.5, palabra, fontsize=tamano, fontweight="bold",
                        color=tono_nube[i % len(tono_nube)], ha="left", va="bottom")
                puestas.append({"palabra": palabra, "frecuencia": f, "tamano": tamano,
                                "x": x, "y": y, "ancho": w, "alto": h})
                break
    return puestas


# GRAFICAS
fig = plt.figure(figsize=(14, 8))
ax = fig.add_axes([0, 0, 1, 1])
puestas = nube(ax, frec, n=150)
guardar(fig, "01_nube")
colocadas = pd.DataFrame(puestas)
colocadas.to_csv("resultados/nube_colocacion.csv", index=False)
print(f"nube: {len(colocadas)} de 150 palabras colocadas")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, g in zip(axes.flat, generos):
    nube(ax, por_genero[g], n=60, tam_max=55, tam_min=7, semilla=42)
    ax.set_title(g, fontsize=14, fontweight="bold")
fig.suptitle(f"Palabras de los titulos por genero ({campo})", fontsize=16, fontweight="bold")
guardar(fig, "02_nubes_genero")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top = tabla.head(25).iloc[::-1]
axes[0].barh(top.palabra, top.frecuencia,
             color=["#c0392b" if p in marcadores else "#1DB954" for p in top.palabra],
             edgecolor="black", linewidth=0.5)
for i, (v, p) in enumerate(zip(top.frecuencia, top.palabra)):
    axes[0].text(v + 15, i, f"{v}", va="center", fontsize=8)
axes[0].set_xlabel("veces que aparece")
axes[0].set_title("25 palabras mas usadas (rojo = metadato de produccion)", fontweight="bold")
axes[0].grid(alpha=0.3, axis="x")

cuenta_largos = np.bincount(largos)[1:]
axes[1].bar(range(1, len(cuenta_largos) + 1), cuenta_largos, color="#2980b9",
            edgecolor="white", linewidth=0.5)
axes[1].axvline(largos.mean(), color="#c0392b", ls="--", lw=1.5,
                label=f"media {largos.mean():.2f} letras")
axes[1].set_xlabel("letras por palabra")
axes[1].set_ylabel("palabras")
axes[1].set_title("Largo de las palabras", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("Frecuencias en los titulos", fontsize=15, fontweight="bold")
guardar(fig, "03_frecuencias")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].scatter(rango, cuentas, s=6, color="#1DB954", alpha=0.5, edgecolors="none")
axes[0].plot(rango, 10 ** (b_zipf[0] + b_zipf[1] * np.log10(rango)), color="#c0392b", lw=2,
             label=f"pendiente {b_zipf[1]:.3f}, r2 = {r2_zipf:.3f}")
for i in range(6):
    axes[0].annotate(orden_todo[i], (rango[i], cuentas[i]), textcoords="offset points",
                     xytext=(9, 5) if i % 2 else (-4, -14), fontsize=8)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("rango de la palabra")
axes[0].set_ylabel("frecuencia")
axes[0].set_title("Ley de Zipf", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3, which="both")

axes[1].plot(n_tokens, vocab, "o-", color="#8e44ad", ms=4, lw=1.5, label="observado")
axes[1].plot(n_tokens, 10 ** (b_heaps[0] + b_heaps[1] * np.log10(n_tokens)), color="#c0392b",
             ls="--", lw=2, label=f"K={10 ** b_heaps[0]:.2f}, beta={b_heaps[1]:.3f}")
axes[1].set_xlabel("palabras leidas")
axes[1].set_ylabel("vocabulario distinto")
axes[1].set_title("Ley de Heaps: el vocabulario no se satura", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
fig.suptitle("Dos leyes empiricas del texto", fontsize=15, fontweight="bold")
guardar(fig, "04_zipf_heaps")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, g in zip(axes.flat, generos):
    d = distintivas[distintivas.genero == g].iloc[::-1]
    ax.barh(d.palabra, d.tfidf * 1000, color="#16a085", edgecolor="black", linewidth=0.5)
    for i, (v, f_) in enumerate(zip(d.tfidf * 1000, d.frecuencia)):
        ax.text(v, i, f" {f_}", va="center", fontsize=8)
    ax.set_title(g, fontweight="bold")
    ax.set_xlabel("tf-idf x 1000")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(alpha=0.3, axis="x")
fig.suptitle("Palabras distintivas por genero (numero = veces que aparece)",
             fontsize=16, fontweight="bold")
guardar(fig, "05_tfidf_genero")

cruce = marcas.pivot(index="marcador", columns="variable", values="d_cohen").loc[marcadores, variables]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
lim = np.abs(cruce.values).max()
im = axes[0].imshow(cruce.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[0].set_xticks(range(len(variables)), variables, rotation=20, ha="right")
axes[0].set_yticks(range(len(marcadores)), marcadores)
for i in range(len(marcadores)):
    for j in range(len(variables)):
        p_ = marcas[(marcas.marcador == marcadores[i]) & (marcas.variable == variables[j])].p.iloc[0]
        axes[0].text(j, i, f"{cruce.values[i, j]:+.2f}{'*' if p_ < 0.05 else ''}",
                     ha="center", va="center", fontsize=9,
                     color="white" if abs(cruce.values[i, j]) > lim * 0.6 else "black")
axes[0].set_title("Diferencia estandarizada (d de Cohen), * = p < 0.05", fontweight="bold")
fig.colorbar(im, ax=axes[0], shrink=0.8)

cuenta = marcas[marcas.variable == variables[0]].set_index("marcador").n_con.loc[marcadores]
axes[1].bar(cuenta.index, cuenta.values, color="#c0392b", edgecolor="black", linewidth=0.6)
for i, v in enumerate(cuenta.values):
    axes[1].text(i, v + 20, f"{v}\n{v / len(df):.1%}", ha="center", fontsize=8)
axes[1].set_ylabel("titulos que la contienen")
axes[1].set_title("Cuantos titulos traen cada marcador", fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("Las palabras de produccion en el titulo y las canciones que marcan",
             fontsize=15, fontweight="bold")
guardar(fig, "06_marcadores")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
