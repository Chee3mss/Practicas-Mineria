import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("resultados", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")

variables = ["track_popularity", "danceability", "energy", "valence",
             "loudness", "tempo", "acousticness", "duracion_min"]
generos = sorted(df["playlist_genre"].unique())
ALFA = 0.05


def grupos(var, etiqueta):
    return [df.loc[df[etiqueta] == g, var].dropna().values for g in sorted(df[etiqueta].unique())]


def pooled(x, y):
    return np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))


# SUPUESTOS
sup = []
for v in variables:
    gs = grupos(v, "playlist_genre")
    residuos = df[v] - df.groupby("playlist_genre")[v].transform("mean")
    W, p_norm = stats.shapiro(residuos.sample(500, random_state=42))
    L, p_lev = stats.levene(*gs, center="median")
    sup.append({"variable": v, "shapiro_W": W, "shapiro_p": p_norm,
                "normal": p_norm > ALFA, "levene_F": L, "levene_p": p_lev,
                "varianzas_iguales": p_lev > ALFA})
sup = pd.DataFrame(sup)
sup.to_csv("resultados/supuestos.csv", index=False)
print(sup.to_string(index=False))


# ANOVA
anova = []
for v in variables:
    gs = grupos(v, "playlist_genre")
    F, p = stats.f_oneway(*gs)
    todo = np.concatenate(gs)
    ss_ent = sum(len(g) * (g.mean() - todo.mean()) ** 2 for g in gs)
    ss_tot = ((todo - todo.mean()) ** 2).sum()
    anova.append({"variable": v, "gl_entre": len(gs) - 1, "gl_dentro": len(todo) - len(gs),
                  "F": F, "p": p, "eta2": ss_ent / ss_tot, "significativo": p < ALFA})
anova = pd.DataFrame(anova).sort_values("F", ascending=False)
anova.to_csv("resultados/anova.csv", index=False)
print("\n", anova.to_string(index=False))


# KRUSKAL
kruskal = []
for v in variables:
    gs = grupos(v, "playlist_genre")
    H, p = stats.kruskal(*gs)
    n = sum(len(g) for g in gs)
    kruskal.append({"variable": v, "gl": len(gs) - 1, "H": H, "p": p,
                    "epsilon2": (H - len(gs) + 1) / (n - len(gs)), "significativo": p < ALFA})
kruskal = pd.DataFrame(kruskal).sort_values("H", ascending=False)
kruskal.to_csv("resultados/kruskal_wallis.csv", index=False)
print("\n", kruskal.to_string(index=False))


# POST HOC
pares = list(combinations(generos, 2))
post = []
for v in variables:
    for a, b in pares:
        x = df.loc[df["playlist_genre"] == a, v].dropna()
        y = df.loc[df["playlist_genre"] == b, v].dropna()
        t, p_t = stats.ttest_ind(x, y, equal_var=False)
        U, p_u = stats.mannwhitneyu(x, y)
        post.append({"variable": v, "grupo_a": a, "grupo_b": b,
                     "media_a": x.mean(), "media_b": y.mean(),
                     "mediana_a": x.median(), "mediana_b": y.median(),
                     "t": t, "p_t": p_t, "U": U, "p_u": p_u,
                     "cohen_d": (x.mean() - y.mean()) / pooled(x, y),
                     "p_t_bonf": min(p_t * len(pares), 1),
                     "p_u_bonf": min(p_u * len(pares), 1),
                     "significativo": p_u * len(pares) < ALFA})
post = pd.DataFrame(post)
post.to_csv("resultados/post_hoc.csv", index=False)

resumen = post.groupby("variable")["significativo"].sum().rename("pares_significativos")
print("\n", resumen.to_string(), f"\nde {len(pares)} pares por variable")


# PRUEBA T
tt = []
for v in variables:
    x = df.loc[df["mode"] == 1, v].dropna()
    y = df.loc[df["mode"] == 0, v].dropna()
    t, p = stats.ttest_ind(x, y, equal_var=False)
    U, p_u = stats.mannwhitneyu(x, y)
    tt.append({"variable": v, "n_mayor": len(x), "n_menor": len(y),
               "media_mayor": x.mean(), "media_menor": y.mean(),
               "dif": x.mean() - y.mean(), "t": t, "p": p, "U": U, "p_u": p_u,
               "cohen_d": (x.mean() - y.mean()) / pooled(x, y), "significativo": p < ALFA})
tt = pd.DataFrame(tt)
tt.to_csv("resultados/prueba_t_mode.csv", index=False)
print("\n", tt.to_string(index=False))

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/")
