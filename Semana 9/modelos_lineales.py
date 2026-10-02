import os

import numpy as np
import pandas as pd
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")

simples = [("loudness", "energy"), ("danceability", "valence"), ("acousticness", "energy"),
           ("energy", "loudness"), ("tempo", "energy"), ("release_year", "track_popularity")]
predictores = ["danceability", "energy", "loudness", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence", "tempo", "duracion_min", "release_year"]
objetivo = "track_popularity"


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


def ajustar(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ beta
    res = y - pred
    sse = (res ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / sst
    n, p = len(y), A.shape[1] - 1
    s2 = sse / (n - p - 1)
    ee = np.sqrt(np.diag(np.linalg.inv(A.T @ A) * s2))
    return beta, pred, res, r2, 1 - (1 - r2) * (n - 1) / (n - p - 1), ee, n, p


# SIMPLES
filas = []
for x, y in simples:
    b, pred, res, r2, r2a, ee, n, p = ajustar(df[[x]].values, df[y].values)
    t = b[1] / ee[1]
    filas.append({"y": y, "x": x, "b0": b[0], "b1": b[1], "ee_b1": ee[1], "t": t,
                  "p": 2 * stats.t.sf(abs(t), n - 2), "r": df[x].corr(df[y]), "R2": r2,
                  "R2_ajustado": r2a, "error_estandar": np.sqrt((res ** 2).sum() / (n - 2))})
simple = pd.DataFrame(filas).sort_values("R2", ascending=False)
simple.to_csv("resultados/modelos_simples.csv", index=False)
print(simple.to_string(index=False))


# MULTIPLE
X = df[predictores].values
y = df[objetivo].values
b, pred, res, r2, r2a, ee, n, p = ajustar(X, y)
t = b / ee
F = (r2 / p) / ((1 - r2) / (n - p - 1))

tc = stats.t.ppf(0.975, n - p - 1)
coef = pd.DataFrame({"termino": ["intercepto"] + predictores, "beta": b, "error_estandar": ee,
                     "t": t, "p": 2 * stats.t.sf(np.abs(t), n - p - 1),
                     "ic_95_inf": b - tc * ee, "ic_95_sup": b + tc * ee})
coef["beta_estandarizado"] = [np.nan] + list(b[1:] * df[predictores].std().values / df[objetivo].std())
coef["significativo"] = coef["p"] < 0.05
coef.to_csv("resultados/coeficientes_multiple.csv", index=False)
print("\n", coef.to_string(index=False))

vif = []
for i, v in enumerate(predictores):
    otros = np.delete(X, i, axis=1)
    vif.append({"predictor": v, "VIF": 1 / (1 - ajustar(otros, X[:, i])[3])})
vif = pd.DataFrame(vif).sort_values("VIF", ascending=False)
vif.to_csv("resultados/vif.csv", index=False)

rng = np.random.default_rng(42)
corte = rng.permutation(n)
ent, pru = corte[:int(n * 0.8)], corte[int(n * 0.8):]
b_ent = ajustar(X[ent], y[ent])[0]
pred_pru = np.column_stack([np.ones(len(pru)), X[pru]]) @ b_ent
r2_pru = 1 - ((y[pru] - pred_pru) ** 2).sum() / ((y[pru] - y[pru].mean()) ** 2).sum()

resumen = pd.DataFrame([{"n": n, "predictores": p, "R2": r2, "R2_ajustado": r2a, "F": F,
                         "p_F": stats.f.sf(F, p, n - p - 1),
                         "error_estandar": np.sqrt((res ** 2).sum() / (n - p - 1)),
                         "R2_prueba": r2_pru, "VIF_max": vif["VIF"].max()}])
resumen.to_csv("resultados/resumen_multiple.csv", index=False)
print("\n", resumen.to_string(index=False))

df_res = pd.DataFrame({"observado": y, "predicho": pred, "residuo": res,
                       "residuo_estandarizado": res / res.std(ddof=p + 1)})
df_res.to_csv("resultados/residuos.csv", index=False)


# GRAFICAS
muestra = df.sample(4000, random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (x, yv) in zip(axes.flat, simples):
    fila = simple.query("x == @x and y == @yv").iloc[0]
    ax.scatter(muestra[x], muestra[yv], s=8, alpha=0.2, color="#1DB954", edgecolors="none")
    xs = np.linspace(muestra[x].min(), muestra[x].max(), 50)
    ax.plot(xs, fila.b0 + fila.b1 * xs, color="#c0392b", lw=2)
    ax.set_xlabel(x)
    ax.set_ylabel(yv)
    signo = "+" if fila.b1 >= 0 else "-"
    ax.set_title(f"{yv} = {fila.b0:.3f} {signo} {abs(fila.b1):.4f}·{x}\nR2 = {fila.R2:.4f}",
                 fontsize=9, fontweight="bold")
fig.suptitle("Regresiones lineales simples", fontsize=15, fontweight="bold")
guardar(fig, "01_regresiones_simples")

m = muestra.index
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes[0, 0].scatter(pred[m], res[m], s=8, alpha=0.2, color="#2980b9", edgecolors="none")
axes[0, 0].axhline(0, color="#c0392b", lw=1.5)
axes[0, 0].set_xlabel("predicho")
axes[0, 0].set_ylabel("residuo")
axes[0, 0].set_title("Residuos vs predichos", fontweight="bold")

stats.probplot(res[m], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Grafico Q-Q de residuos", fontweight="bold")
axes[0, 1].set_xlabel("cuantiles teoricos")
axes[0, 1].set_ylabel("cuantiles observados")
axes[0, 1].get_lines()[0].set(ms=2, alpha=0.4, color="#2980b9")
axes[0, 1].get_lines()[1].set(color="#c0392b", lw=1.5)

axes[1, 0].hist(res, bins=40, color="#2980b9", edgecolor="white", linewidth=0.5)
axes[1, 0].axvline(res.mean(), color="#c0392b", ls="--", lw=1.5, label=f"media {res.mean():.2e}")
axes[1, 0].set_title("Distribucion de residuos", fontweight="bold")
axes[1, 0].legend(fontsize=8)

axes[1, 1].scatter(y[m], pred[m], s=8, alpha=0.2, color="#2980b9", edgecolors="none")
lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
axes[1, 1].plot(lim, lim, color="#c0392b", lw=1.5)
axes[1, 1].set_xlabel("observado")
axes[1, 1].set_ylabel("predicho")
axes[1, 1].set_title(f"Observado vs predicho  R2 = {r2:.4f}", fontweight="bold")
fig.suptitle(f"Diagnostico del modelo multiple ({objetivo})", fontsize=15, fontweight="bold")
guardar(fig, "02_diagnostico")

orden = coef.iloc[1:].reindex(coef.iloc[1:].beta_estandarizado.abs().sort_values().index)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(orden.termino, orden.beta_estandarizado,
             color=["#27ae60" if v > 0 else "#c0392b" for v in orden.beta_estandarizado],
             edgecolor="black", linewidth=0.6)
axes[0].axvline(0, color="black", lw=1)
axes[0].set_title("Coeficientes estandarizados", fontweight="bold")

etiquetas = list(simple.y + " ~ " + simple.x) + [f"{objetivo} ~ multiple"]
valores = list(simple.R2) + [r2]
objetivos = list(simple.y) + [objetivo]
tono = dict(zip(sorted(set(objetivos)), plt.cm.Set2(np.linspace(0, 1, len(set(objetivos))))))
axes[1].bar(range(len(valores)), valores, color=[tono[o] for o in objetivos],
            edgecolor="black", linewidth=0.6)
axes[1].set_xticks(range(len(valores)), etiquetas, rotation=30, ha="right", fontsize=8)
for i, v in enumerate(valores):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
axes[1].set_ylabel("R2")
axes[1].set_title("R2 por modelo (comparable solo entre barras del mismo color)", fontweight="bold")
axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", label=f"y = {o}")
                        for o, c in tono.items()], fontsize=8)
guardar(fig, "03_coeficientes_r2")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
