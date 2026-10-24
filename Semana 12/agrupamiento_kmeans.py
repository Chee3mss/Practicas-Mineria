import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")

variables = ["danceability", "energy", "loudness", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence", "tempo", "duracion_min"]

repetidas = df.duplicated(variables).sum()
df = df.drop_duplicates(variables).reset_index(drop=True)
print(f"{repetidas} filas con audio identico eliminadas, quedan {len(df)}")

X = df[variables].values
genero = df["playlist_genre"].astype(str).to_numpy()
generos = sorted(set(genero))

medias = X.mean(axis=0)
desvs = X.std(axis=0)
Z = (X - medias) / desvs


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


def distancias(A, B):
    d = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.maximum(d, 0)


def kmeans(Z, k, semilla, pasos=100):
    rng = np.random.default_rng(semilla)
    centros = Z[rng.choice(len(Z), k, replace=False)]
    historia = [centros]
    for paso in range(1, pasos + 1):
        grupo = distancias(Z, centros).argmin(axis=1)
        nuevos = np.array([Z[grupo == j].mean(axis=0) if (grupo == j).any()
                           else Z[rng.integers(len(Z))] for j in range(k)])
        if np.allclose(nuevos, centros):
            break
        centros = nuevos
        historia.append(centros)
    d = distancias(Z, centros)
    return centros, d.argmin(axis=1), d.min(axis=1).sum(), paso, historia


def mejor_kmeans(Z, k, inicios=10):
    corridas = [kmeans(Z, k, 42 + i) for i in range(inicios)]
    return min(corridas, key=lambda c: c[2])


def silueta(Z, grupo, idx):
    sub, g = Z[idx], grupo[idx]
    D = np.sqrt(distancias(sub, sub))
    mascaras = {c: g == c for c in set(g)}
    s = np.zeros(len(sub))
    for i in range(len(sub)):
        propio = mascaras[g[i]].copy()
        propio[i] = False
        if propio.sum() == 0:
            continue
        a = D[i, propio].mean()
        b = min(D[i, m].mean() for c, m in mascaras.items() if c != g[i])
        s[i] = (b - a) / max(a, b)
    return s


def davies_bouldin(Z, grupo, centros):
    disp = np.array([np.sqrt(((Z[grupo == j] - centros[j]) ** 2).sum(axis=1)).mean()
                     for j in range(len(centros))])
    sep = np.sqrt(distancias(centros, centros))
    peor = []
    for j in range(len(centros)):
        otros = [i for i in range(len(centros)) if i != j]
        peor.append(max((disp[j] + disp[i]) / sep[j, i] for i in otros))
    return np.mean(peor)


def tabla_cruzada(grupo, etiqueta, k, clases):
    return np.array([[((grupo == j) & (etiqueta == c)).sum() for c in clases] for j in range(k)])


def pureza(tabla):
    return tabla.max(axis=1).sum() / tabla.sum()


def pares(v):
    return (v * (v - 1) / 2).sum()


def rand_ajustado(tabla):
    n = tabla.sum()
    indice = pares(tabla.astype(float))
    a, b = pares(tabla.sum(axis=1).astype(float)), pares(tabla.sum(axis=0).astype(float))
    esperado = a * b / (n * (n - 1) / 2)
    return (indice - esperado) / ((a + b) / 2 - esperado)


# SELECCION DE K
rng = np.random.default_rng(42)
muestra = rng.choice(len(Z), 2000, replace=False)
ks = range(2, 11)
filas = []
modelos = {}
for k in ks:
    centros, grupo, inercia, pasos, historia = mejor_kmeans(Z, k)
    s = silueta(Z, grupo, muestra)
    t = tabla_cruzada(grupo, genero, k, generos)
    modelos[k] = (centros, grupo, historia)
    filas.append({"k": k, "inercia": inercia, "silueta": s.mean(),
                  "davies_bouldin": davies_bouldin(Z, grupo, centros),
                  "pureza_genero": pureza(t), "rand_ajustado_genero": rand_ajustado(t),
                  "grupo_menor": np.bincount(grupo).min(), "pasos": pasos})
    print(f"k={k} inercia={inercia:.0f} silueta={s.mean():.4f} pureza={pureza(t):.4f}")

seleccion = pd.DataFrame(filas)
seleccion["caida_inercia"] = -seleccion.inercia.diff()
seleccion.to_csv("resultados/seleccion_k.csv", index=False)

k_mejor = int(seleccion.loc[seleccion.silueta.idxmax(), "k"])
centros, grupo, historia = modelos[k_mejor]
print(f"\nmejor silueta con k={k_mejor}")


# AGRUPAMIENTO ELEGIDO
s = silueta(Z, grupo, muestra)
tabla = tabla_cruzada(grupo, genero, k_mejor, generos)

k_gen = len(generos)
centros_gen, grupo_gen, _ = modelos[k_gen]
tabla_gen = tabla_cruzada(grupo_gen, genero, k_gen, generos)

resumen = pd.DataFrame([{
    "n": len(Z), "filas_repetidas_eliminadas": repetidas, "variables": len(variables),
    "k": k_mejor, "inercia": seleccion.inercia[seleccion.k == k_mejor].iloc[0],
    "silueta": s.mean(), "silueta_negativa": (s < 0).mean(),
    "davies_bouldin": davies_bouldin(Z, grupo, centros),
    "grupo_menor": np.bincount(grupo).min(), "grupo_mayor": np.bincount(grupo).max(),
    "pureza_genero": pureza(tabla), "rand_ajustado_genero": rand_ajustado(tabla),
    "generos": len(generos),
    "pureza_con_k_igual_generos": pureza(tabla_gen),
    "rand_ajustado_con_k_igual_generos": rand_ajustado(tabla_gen),
}])
resumen.to_csv("resultados/resumen.csv", index=False)
print("\n", resumen.T.to_string())

perfil = pd.DataFrame(centros, columns=variables)
perfil.index.name = "grupo"
perfil.to_csv("resultados/centroides_estandarizados.csv")
(perfil * desvs + medias).to_csv("resultados/centroides.csv")
print("\n", (perfil * desvs + medias).round(3).to_string())

por_grupo = pd.DataFrame({
    "grupo": range(k_mejor),
    "canciones": np.bincount(grupo, minlength=k_mejor),
    "silueta": [s[grupo[muestra] == j].mean() for j in range(k_mejor)],
    "popularidad_media": [df.track_popularity[grupo == j].mean() for j in range(k_mejor)],
    "genero_dominante": [generos[i] for i in tabla.argmax(axis=1)],
    "proporcion_dominante": tabla.max(axis=1) / tabla.sum(axis=1),
})
por_grupo.to_csv("resultados/por_grupo.csv", index=False)
print("\n", por_grupo.to_string(index=False))

cruce = pd.DataFrame(tabla_gen, columns=generos)
cruce.index.name = "grupo"
cruce.to_csv("resultados/grupo_por_genero.csv")
cruce_norm = cruce.div(cruce.sum(axis=1), axis=0)
cruce_norm.to_csv("resultados/grupo_por_genero_normalizado.csv")

df_asignado = df[["track_id", "track_name", "track_artist", "playlist_genre"]].copy()
df_asignado["grupo"] = grupo
df_asignado[f"grupo_k{k_gen}"] = grupo_gen
df_asignado.to_csv("resultados/asignaciones.csv", index=False)


# COMPONENTES PRINCIPALES
U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
PC = (Z - Z.mean(axis=0)) @ Vt[:2].T
explicada = S ** 2 / (S ** 2).sum()
print(f"\nvarianza explicada por PC1 y PC2: {explicada[0]:.3f} y {explicada[1]:.3f}")


# GRAFICAS
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
axes[0].plot(seleccion.k, seleccion.inercia, "o-", color="#1DB954", lw=2)
axes[0].set_xlabel("k grupos")
axes[0].set_ylabel("inercia")
axes[0].set_title("Codo", fontweight="bold")

axes[1].plot(seleccion.k, seleccion.silueta, "o-", color="#2980b9", lw=2)
axes[1].axvline(k_mejor, color="#c0392b", ls="--", lw=1.5, label=f"k elegido = {k_mejor}")
axes[1].set_xlabel("k grupos")
axes[1].set_ylabel("silueta promedio")
axes[1].set_title("Silueta", fontweight="bold")
axes[1].legend(fontsize=9)

axes[2].plot(seleccion.k, seleccion.davies_bouldin, "o-", color="#e67e22", lw=2)
axes[2].set_xlabel("k grupos")
axes[2].set_ylabel("Davies-Bouldin (menor es mejor)")
axes[2].set_title("Davies-Bouldin", fontweight="bold")
for ax in axes:
    ax.set_xticks(list(ks))
    ax.grid(alpha=0.3)
fig.suptitle("Seleccion del numero de grupos", fontsize=15, fontweight="bold")
guardar(fig, "01_seleccion_k")

tono = plt.cm.tab10(np.arange(max(k_mejor, len(generos))))
idx = rng.choice(len(Z), 4000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for j in range(k_mejor):
    sel = idx[grupo[idx] == j]
    axes[0].scatter(PC[sel, 0], PC[sel, 1], s=10, alpha=0.4, color=tono[j],
                    edgecolors="none", label=f"grupo {j} (n={np.bincount(grupo)[j]})")
centros_pc = (centros - Z.mean(axis=0)) @ Vt[:2].T
axes[0].scatter(centros_pc[:, 0], centros_pc[:, 1], s=260, marker="X", c="white",
                edgecolors="black", linewidths=1.5, zorder=5)
axes[0].set_title(f"Grupos de k-means (k={k_mejor})", fontweight="bold")
axes[0].legend(fontsize=8, markerscale=2)

for i, g in enumerate(generos):
    sel = idx[genero[idx] == g]
    axes[1].scatter(PC[sel, 0], PC[sel, 1], s=10, alpha=0.4, color=tono[i],
                    edgecolors="none", label=g)
axes[1].set_title("Generos reales en el mismo plano", fontweight="bold")
axes[1].legend(fontsize=8, markerscale=2)
for ax in axes:
    ax.set_xlabel(f"PC1 ({explicada[0]:.1%} de la varianza)")
    ax.set_ylabel(f"PC2 ({explicada[1]:.1%})")
fig.suptitle("Proyeccion en las dos primeras componentes", fontsize=15, fontweight="bold")
guardar(fig, "02_grupos")

fig, ax = plt.subplots(figsize=(12, 1.2 * k_mejor + 2.5))
limite = np.abs(centros).max()
im = ax.imshow(centros, cmap="RdBu_r", vmin=-limite, vmax=limite, aspect="auto")
ax.set_xticks(range(len(variables)), variables, rotation=30, ha="right")
ax.set_yticks(range(k_mejor), [f"grupo {j}" for j in range(k_mejor)])
for j in range(k_mejor):
    for v in range(len(variables)):
        ax.text(v, j, f"{centros[j, v]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(centros[j, v]) > limite * 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.025, label="desviaciones estandar respecto a la media")
ax.set_title("Perfil de cada grupo", fontweight="bold")
guardar(fig, "03_perfiles")

fig, axes = plt.subplots(1, 2, figsize=(16, 1.1 * k_gen + 4))
im = axes[0].imshow(cruce_norm.values, cmap="Greens", vmin=0, aspect="auto")
for j in range(k_gen):
    for i in range(len(generos)):
        v = cruce_norm.values[j, i]
        axes[0].text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v > cruce_norm.values.max() * 0.6 else "black")
axes[0].set_xticks(range(len(generos)), generos, rotation=30, ha="right")
axes[0].set_yticks(range(k_gen), [f"grupo {j}" for j in range(k_gen)])
axes[0].set_title("Composicion de cada grupo", fontweight="bold")
fig.colorbar(im, ax=axes[0], fraction=0.035)

ancho = 0.8
abajo = np.zeros(k_gen)
for i, g in enumerate(generos):
    axes[1].bar(range(k_gen), cruce_norm[g], ancho, bottom=abajo, label=g, color=tono[i],
                edgecolor="white", linewidth=0.6)
    abajo += cruce_norm[g].values
axes[1].set_xticks(range(k_gen), [f"grupo {j}" for j in range(k_gen)])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("proporcion del grupo")
axes[1].set_title(f"pureza = {pureza(tabla_gen):.3f}   "
                  f"Rand ajustado = {rand_ajustado(tabla_gen):.3f}", fontweight="bold")
axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.suptitle(f"Los {k_gen} grupos frente a los {len(generos)} generos declarados",
             fontsize=15, fontweight="bold")
guardar(fig, "04_grupos_vs_genero")

orden = np.lexsort((-s, grupo[muestra]))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(orden)), s[orden], width=1,
       color=[tono[grupo[muestra][i]] for i in orden])
ax.axhline(s.mean(), color="#c0392b", ls="--", lw=1.5, label=f"promedio {s.mean():.3f}")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel(f"canciones de la muestra de {len(muestra)}, ordenadas por grupo")
ax.set_ylabel("silueta")
ax.set_title(f"Silueta individual  ({(s < 0).mean():.1%} mal asignadas)", fontweight="bold")
ax.legend(fontsize=9)
guardar(fig, "05_silueta")

camino = np.array([(c - Z.mean(axis=0)) @ Vt[:2].T for c in historia])
inercias = [distancias(Z, c).min(axis=1).sum() for c in historia]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].scatter(PC[idx, 0], PC[idx, 1], s=6, alpha=0.1, color="gray", edgecolors="none")
for j in range(k_mejor):
    axes[0].plot(camino[:, j, 0], camino[:, j, 1], "-o", ms=4, lw=1.5, color=tono[j],
                 label=f"centro {j}")
    axes[0].scatter(camino[0, j, 0], camino[0, j, 1], s=130, marker="s", color=tono[j],
                    edgecolors="black", linewidths=1.2, zorder=4)
    axes[0].scatter(camino[-1, j, 0], camino[-1, j, 1], s=230, marker="X", c="white",
                    edgecolors="black", linewidths=1.5, zorder=5)
axes[0].set_xlabel(f"PC1 ({explicada[0]:.1%} de la varianza)")
axes[0].set_ylabel(f"PC2 ({explicada[1]:.1%})")
axes[0].set_title("Recorrido de los centros (cuadro = inicio al azar, X = final)",
                  fontweight="bold")
axes[0].legend(fontsize=9)

axes[1].plot(range(1, len(inercias) + 1), inercias, "o-", color="#1DB954", lw=2)
axes[1].set_xticks(range(1, len(inercias) + 1, 5))
axes[1].set_xlabel("iteracion")
axes[1].set_ylabel("inercia")
axes[1].set_title(f"La inercia baja hasta estabilizarse en {len(inercias)} iteraciones",
                  fontweight="bold")
axes[1].grid(alpha=0.3)
fig.suptitle(f"Convergencia del k-means (k={k_mejor})", fontsize=15, fontweight="bold")
guardar(fig, "06_convergencia")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
