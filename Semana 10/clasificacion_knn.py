import os

import numpy as np
import pandas as pd
import matplotlib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("../Semana 5/dataframes/canciones.csv")

predictores = ["danceability", "energy", "loudness", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence", "tempo", "duracion_min"]
objetivo = "playlist_genre"

repetidas = df.duplicated(predictores).sum()
df = df.drop_duplicates(predictores).reset_index(drop=True)
print(f"{repetidas} filas con audio identico eliminadas, quedan {len(df)}")

X = df[predictores].values
y = df[objetivo].astype(str).to_numpy()
clases = sorted(set(y))

X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


def modelo(k, pesos):
    return Pipeline([("escala", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=k, weights=pesos))])


# SELECCION DE K
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ks = [1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51]
filas = []
for pesos in ["uniform", "distance"]:
    for k in ks:
        s = cross_val_score(modelo(k, pesos), X_ent, y_ent, cv=cv, scoring="accuracy", n_jobs=-1)
        filas.append({"pesos": pesos, "k": k, "exactitud_cv": s.mean(), "desv_cv": s.std()})
        print(f"{pesos} k={k} -> {s.mean():.4f}")

seleccion = pd.DataFrame(filas)
seleccion.to_csv("resultados/seleccion_k.csv", index=False)

mejor = seleccion.loc[seleccion.exactitud_cv.idxmax()]
k_mejor, pesos_mejor = int(mejor.k), mejor.pesos
print(f"\nmejor: k={k_mejor} pesos={pesos_mejor} exactitud_cv={mejor.exactitud_cv:.4f}")


# MODELO FINAL
knn = modelo(k_mejor, pesos_mejor).fit(X_ent, y_ent)
pred = knn.predict(X_pru)

base = DummyClassifier(strategy="most_frequent").fit(X_ent, y_ent)
pred_base = base.predict(X_pru)

resumen = pd.DataFrame([{
    "n_total": len(y), "filas_repetidas_eliminadas": repetidas,
    "n_entrenamiento": len(y_ent), "n_prueba": len(y_pru),
    "k": k_mejor, "pesos": pesos_mejor, "predictores": len(predictores), "clases": len(clases),
    "exactitud_cv": mejor.exactitud_cv, "desv_cv": mejor.desv_cv,
    "exactitud_entrenamiento": accuracy_score(y_ent, knn.predict(X_ent)),
    "exactitud_prueba": accuracy_score(y_pru, pred),
    "exactitud_balanceada": balanced_accuracy_score(y_pru, pred),
    "f1_macro": f1_score(y_pru, pred, average="macro"),
    "f1_ponderado": f1_score(y_pru, pred, average="weighted"),
    "exactitud_base": accuracy_score(y_pru, pred_base),
}])
resumen.to_csv("resultados/resumen.csv", index=False)
print("\n", resumen.T.to_string())

reporte = pd.DataFrame(classification_report(y_pru, pred, output_dict=True)).T
reporte.index.name = "clase"
reporte.to_csv("resultados/metricas_por_clase.csv")
print("\n", classification_report(y_pru, pred, digits=4))

mc = confusion_matrix(y_pru, pred, labels=clases)
pd.DataFrame(mc, index=clases, columns=clases).to_csv("resultados/matriz_confusion.csv")

mc_norm = mc / mc.sum(axis=1, keepdims=True)
pd.DataFrame(mc_norm, index=clases, columns=clases).to_csv("resultados/matriz_confusion_normalizada.csv")

pd.DataFrame({"real": y_pru, "predicho": pred, "correcto": y_pru == pred}).to_csv(
    "resultados/predicciones.csv", index=False)


# GRAFICAS
fig, ax = plt.subplots(figsize=(10, 6))
for pesos, color in [("uniform", "#1DB954"), ("distance", "#2980b9")]:
    sub = seleccion.query("pesos == @pesos")
    ax.plot(sub.k, sub.exactitud_cv, "o-", color=color, label=pesos, lw=2, ms=5)
    ax.fill_between(sub.k, sub.exactitud_cv - sub.desv_cv, sub.exactitud_cv + sub.desv_cv,
                    color=color, alpha=0.15)
ax.axhline(resumen.exactitud_base[0], color="#7f8c8d", ls=":", lw=1.5,
           label=f"clase mayoritaria ({resumen.exactitud_base[0]:.3f})")
ax.axvline(k_mejor, color="#c0392b", ls="--", lw=1.5, label=f"k elegido = {k_mejor}")
ax.set_xlabel("k vecinos")
ax.set_ylabel("exactitud (validacion cruzada 5 pliegues)")
ax.set_title("Seleccion de k", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
guardar(fig, "01_seleccion_k")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (m, titulo, fmt) in zip(axes, [(mc, "Conteos", "d"), (mc_norm, "Proporcion por fila", ".2f")]):
    im = ax.imshow(m, cmap="Greens")
    ax.set_xticks(range(len(clases)), clases, rotation=30, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(titulo, fontweight="bold")
    limite = m.max() * 0.6
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, format(m[i, j], fmt), ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > limite else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Matriz de confusion  k={k_mejor}  exactitud={resumen.exactitud_prueba[0]:.4f}",
             fontsize=15, fontweight="bold")
guardar(fig, "02_matriz_confusion")

por_clase = reporte.loc[clases]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ancho = 0.26
for i, (metrica, color) in enumerate([("precision", "#1DB954"), ("recall", "#2980b9"),
                                      ("f1-score", "#e67e22")]):
    axes[0].bar(np.arange(len(clases)) + (i - 1) * ancho, por_clase[metrica], ancho,
                label=metrica, color=color, edgecolor="black", linewidth=0.6)
axes[0].axhline(1 / len(clases), color="#7f8c8d", ls=":", lw=1.5, label="azar (1/6)")
axes[0].set_xticks(range(len(clases)), clases, rotation=30, ha="right")
axes[0].set_ylim(0, 1)
axes[0].set_title("Metricas por genero", fontweight="bold")
axes[0].legend(fontsize=8, loc="lower right")

confusiones = pd.DataFrame([{"par": f"{clases[i]} -> {clases[j]}", "casos": mc[i, j],
                             "proporcion": mc_norm[i, j]}
                            for i in range(len(clases)) for j in range(len(clases)) if i != j])
confusiones = confusiones.sort_values("casos").tail(10)
confusiones.to_csv("resultados/confusiones_frecuentes.csv", index=False)
axes[1].barh(confusiones.par, confusiones.casos, color="#c0392b", edgecolor="black", linewidth=0.6)
for i, (c, p) in enumerate(zip(confusiones.casos, confusiones.proporcion)):
    axes[1].text(c + 3, i, f"{c}  ({p:.0%} de la clase)", va="center", fontsize=8)
axes[1].set_xlim(0, confusiones.casos.max() * 1.35)
axes[1].set_xlabel("canciones mal clasificadas")
axes[1].set_title("Confusiones mas frecuentes", fontweight="bold")
guardar(fig, "03_metricas_por_clase")

par = ["danceability", "tempo"]
ie = [predictores.index(v) for v in par]
knn2 = modelo(k_mejor, pesos_mejor).fit(X_ent[:, ie], y_ent)
print(f"\nexactitud solo con {par}: {accuracy_score(y_pru, knn2.predict(X_pru[:, ie])):.4f}")

rng = np.random.default_rng(42)
idx = rng.choice(len(X_ent), 1200, replace=False)
x1 = np.linspace(*np.quantile(X_ent[:, ie[0]], [0.005, 0.995]), 260)
x2 = np.linspace(*np.quantile(X_ent[:, ie[1]], [0.005, 0.995]), 260)
m1, m2 = np.meshgrid(x1, x2)
zonas = knn2.predict(np.column_stack([m1.ravel(), m2.ravel()])).reshape(m1.shape)
codigo = np.searchsorted(clases, zonas)

tono = plt.cm.tab10(np.arange(len(clases)))
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.pcolormesh(m1, m2, codigo, cmap=matplotlib.colors.ListedColormap(tono), alpha=0.3,
              shading="nearest", vmin=-0.5, vmax=len(clases) - 0.5)
for i, c in enumerate(clases):
    sel = idx[y_ent[idx] == c]
    ax.scatter(X_ent[sel, ie[0]], X_ent[sel, ie[1]], s=22, alpha=0.9, color=tono[i],
               edgecolors="white", linewidths=0.5, label=c)
ax.set_xlim(x1.min(), x1.max())
ax.set_ylim(x2.min(), x2.max())
ax.set_xlabel(par[0])
ax.set_ylabel(par[1])
ax.set_title(f"Frontera de decision con 2 de los {len(predictores)} predictores  (k={k_mejor})",
             fontweight="bold")
ax.legend(fontsize=9, markerscale=1.6, loc="center left", bbox_to_anchor=(1.01, 0.5))
guardar(fig, "04_frontera_decision")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
