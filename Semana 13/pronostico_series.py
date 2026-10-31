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

objetivo = "duracion_min"
series = ["duracion_min", "loudness", "valence"]
inicio, fin = 1980, 2019
horizonte = 5
prueba = 10
prueba_larga = 15
recientes = 5


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"graficas/{nombre}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"{nombre}.png")


def ajustar(t, y, grado=1):
    A = np.vander(t, grado + 1, increasing=True)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ beta
    sse = (res ** 2).sum()
    n = len(y)
    s2 = sse / (n - grado - 1)
    r2 = 1 - sse / ((y - y.mean()) ** 2).sum()
    return beta, res, r2, np.sqrt(s2), np.linalg.inv(A.T @ A) * s2, n


def predecir(beta, t, grado=1):
    return np.vander(np.atleast_1d(np.asarray(t, dtype=float)), grado + 1, increasing=True) @ beta


def banda(t0, beta, s, cov, n, grado=1):
    A0 = np.vander(np.atleast_1d(np.asarray(t0, dtype=float)), grado + 1, increasing=True)
    error = np.sqrt(s ** 2 + np.array([x @ cov @ x for x in A0]))
    tc = stats.t.ppf(0.975, n - grado - 1)
    p = A0 @ beta
    return p, p - tc * error, p + tc * error


# MODELOS
def ingenuo(t, y, h):
    return np.repeat(y[-1], h)


def media_movil(t, y, h, k=3):
    return np.repeat(y[-k:].mean(), h)


def deriva(t, y, h):
    return y[-1] + (y[-1] - y[0]) / (len(y) - 1) * np.arange(1, h + 1)


def tendencia(t, y, h, ventana=None, grado=1):
    if ventana is not None and ventana < len(y):
        t, y = t[-ventana:], y[-ventana:]
    beta = ajustar(t, y, grado)[0]
    return predecir(beta, t[-1] + np.arange(1, h + 1), grado)


def amortiguada(t, y, h, ventana=10, phi=0.8):
    m = ajustar(t[-ventana:], y[-ventana:])[0][1]
    return y[-1] + m * np.cumsum(phi ** np.arange(1, h + 1))


modelos = {"ingenuo": ingenuo,
           "media_movil_3": media_movil,
           "deriva": deriva,
           "deriva_10": lambda t, y, h: deriva(t, y[-10:], h),
           "amortiguada_10": amortiguada,
           "tendencia_todo": tendencia,
           "tendencia_20": lambda t, y, h: tendencia(t, y, h, ventana=20),
           "tendencia_10": lambda t, y, h: tendencia(t, y, h, ventana=10),
           "tendencia_5": lambda t, y, h: tendencia(t, y, h, ventana=5),
           "cuadratico_todo": lambda t, y, h: tendencia(t, y, h, grado=2)}


def mae(y, p):
    return np.abs(y - p).mean()


def rmse(y, p):
    return np.sqrt(((y - p) ** 2).mean())


def mape(y, p):
    return (np.abs(y - p) / np.abs(y)).mean() * 100


# SERIE
d = df[(df.release_year >= inicio) & (df.release_year <= fin)]
serie = d.groupby("release_year").agg(canciones=("track_id", "size"),
                                      **{v: (v, "mean") for v in series})
serie.index.name = "anio"
serie = serie.reset_index()
serie["t"] = serie.anio - inicio
serie.to_csv("resultados/serie_anual.csv", index=False)

t = serie.t.values.astype(float)
y = serie[objetivo].values
anios = serie.anio.values

print(f"serie anual {inicio}-{fin}: {len(serie)} puntos, "
      f"{serie.canciones.min()} a {serie.canciones.max()} canciones por anio")
print(f"{objetivo}: {y[0]:.3f} min en {anios[0]} a {y[-1]:.3f} min en {anios[-1]}")


# TENDENCIAS
filas = []
for v in series:
    yv = serie[v].values
    b, res, r2, s, cov, n = ajustar(t, yv)
    ee = np.sqrt(np.diag(cov))
    tt = b[1] / ee[1]
    b10 = ajustar(t[-10:], yv[-10:])[0]
    filas.append({"variable": v, "b0": b[0], "pendiente_anual": b[1], "error_estandar": ee[1],
                  "t": tt, "p": 2 * stats.t.sf(abs(tt), n - 2), "r2": r2,
                  "cambio_por_decada": b[1] * 10, "pendiente_ultimos_10": b10[1],
                  "r": np.corrcoef(t, yv)[0, 1]})
tendencias = pd.DataFrame(filas)
tendencias.to_csv("resultados/tendencias.csv", index=False)
print("\n", tendencias.round(4).to_string(index=False))


# VENTANAS
filas = []
for v in [len(serie), 20, 10, 5]:
    tv, yv = t[-v:], y[-v:]
    b, res, r2, s, cov, n = ajustar(tv, yv)
    futuro = tv[-1] + np.arange(1, horizonte + 1)
    p, lo, hi = banda(futuro, b, s, cov, n)
    filas.append({"ventana": f"{v} anios", "desde": int(anios[-v]), "n": n,
                  "pendiente_anual": b[1], "r2": r2, "error_estandar": s,
                  f"pronostico_{fin + 1}": p[0], f"pronostico_{fin + horizonte}": p[-1],
                  "ancho_ic_final": hi[-1] - lo[-1]})
ventanas = pd.DataFrame(filas)
ventanas.to_csv("resultados/ventanas.csv", index=False)
print("\n", ventanas.round(4).to_string(index=False))


# VALIDACION
detalle = []
for i in range(len(serie) - prueba, len(serie)):
    for nombre, f in modelos.items():
        detalle.append({"anio": int(anios[i]), "modelo": nombre, "observado": y[i],
                        "predicho": float(np.atleast_1d(f(t[:i], y[:i], 1))[0])})
detalle = pd.DataFrame(detalle)
detalle["error"] = detalle.observado - detalle.predicho
detalle.to_csv("resultados/validacion_un_paso.csv", index=False)

largo = []
for i in range(len(serie) - prueba_larga, len(serie) - horizonte + 1):
    for nombre, f in modelos.items():
        pred = np.atleast_1d(f(t[:i], y[:i], horizonte))
        for h in range(horizonte):
            largo.append({"origen": int(anios[i - 1]), "anio": int(anios[i + h]), "pasos": h + 1,
                          "modelo": nombre, "observado": y[i + h], "predicho": float(pred[h])})
largo = pd.DataFrame(largo)
largo["error"] = largo.observado - largo.predicho
largo.to_csv("resultados/validacion_multipaso.csv", index=False)

filas = []
ultimos = sorted(largo.origen.unique())[-recientes:]
for nombre in modelos:
    g = detalle[detalle.modelo == nombre]
    m = largo[largo.modelo == nombre]
    r = m[m.origen.isin(ultimos)]
    f = modelos[nombre]
    e = g.error.values
    filas.append({"modelo": nombre, "mae_un_paso": mae(g.observado.values, g.predicho.values),
                  "rmse_un_paso": rmse(g.observado.values, g.predicho.values),
                  "mape_un_paso": mape(g.observado.values, g.predicho.values),
                  "mae_multipaso": mae(m.observado.values, m.predicho.values),
                  "rmse_multipaso": rmse(m.observado.values, m.predicho.values),
                  "mape_multipaso": mape(m.observado.values, m.predicho.values),
                  "mae_recientes": mae(r.observado.values, r.predicho.values),
                  "mape_recientes": mape(r.observado.values, r.predicho.values),
                  "mae_paso_5": mae(m[m.pasos == horizonte].observado.values,
                                    m[m.pasos == horizonte].predicho.values),
                  "salto_inicial": float(np.atleast_1d(f(t, y, 1))[0]) - y[-1],
                  "autocorrelacion_error": np.corrcoef(e[:-1], e[1:])[0, 1]})
validacion = pd.DataFrame(filas).sort_values("mae_recientes").reset_index(drop=True)
validacion.to_csv("resultados/validacion_modelos.csv", index=False)
origenes = largo.origen.nunique()
print(f"\nvalidacion: {prueba} origenes a un paso ({int(anios[-prueba])}-{fin}), "
      f"{origenes} origenes a {horizonte} pasos, "
      f"{recientes} recientes (origen {ultimos[0]}-{ultimos[-1]})")
print("\n", validacion.round(4).to_string(index=False))

elegido = validacion.modelo[0]
mejor_todos = validacion.sort_values("mae_multipaso").modelo.iloc[0]
print(f"elegido por error a {horizonte} pasos en los origenes recientes: {elegido}   "
      f"(con todos los origenes habria ganado {mejor_todos})")


# PRONOSTICO
p = np.atleast_1d(modelos[elegido](t, y, horizonte))
g = largo[largo.modelo == elegido]
rmse_paso = np.array([rmse(g[g.pasos == h].observado.values, g[g.pasos == h].predicho.values)
                      for h in range(1, horizonte + 1)])

lo, hi = p - 1.96 * rmse_paso, p + 1.96 * rmse_paso

pronostico = pd.DataFrame({"anio": fin + np.arange(1, horizonte + 1), "pasos": np.arange(1, horizonte + 1),
                           "pronostico": p, "rmse_validacion": rmse_paso,
                           "ic_95_inf": lo, "ic_95_sup": hi,
                           "cambio_vs_2019": p - y[-1]})
pronostico.to_csv("resultados/pronostico.csv", index=False)

errores = detalle[detalle.modelo == elegido].reset_index(drop=True)
res = errores.error.values
dw = (np.diff(res) ** 2).sum() / (res ** 2).sum()
auto1 = np.corrcoef(res[:-1], res[1:])[0, 1]
errores.to_csv("resultados/errores_modelo.csv", index=False)

fila = tendencias[tendencias.variable == objetivo].iloc[0]
resumen = pd.DataFrame([{"objetivo": objetivo, "inicio": inicio, "fin": fin, "puntos": len(serie),
                         "ultimo_observado": y[-1], "modelo_elegido": elegido,
                         "mejor_con_todos_los_origenes": mejor_todos,
                         "pendiente_global": fila.pendiente_anual,
                         "pendiente_ultimos_10": fila.pendiente_ultimos_10,
                         "mae_un_paso": float(validacion.mae_un_paso[0]),
                         "mae_multipaso": float(validacion.mae_multipaso[0]),
                         "mae_recientes": float(validacion.mae_recientes[0]),
                         "mape_recientes": float(validacion.mape_recientes[0]),
                         "salto_inicial": float(validacion.salto_inicial[0]),
                         "durbin_watson": dw, "autocorrelacion_lag1": auto1,
                         f"pronostico_{fin + 1}": p[0], f"pronostico_{fin + horizonte}": p[-1]}])
resumen.to_csv("resultados/resumen.csv", index=False)
print("\n", resumen.round(4).to_string(index=False))
print("\n", pronostico.round(3).to_string(index=False))


# GRAFICAS
tono = {"duracion_min": "#1DB954", "loudness": "#2980b9", "valence": "#8e44ad"}
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, v in zip(axes.flat, series):
    yv = serie[v].values
    bv = ajustar(t, yv)[0]
    fila = tendencias[tendencias.variable == v].iloc[0]
    ax.plot(anios, yv, "o-", color=tono[v], ms=4, lw=1.5)
    ax.plot(anios, predecir(bv, t), color="#c0392b", lw=1.8, ls="--",
            label=f"tendencia global {fila.pendiente_anual:+.4f}/anio")
    ax.plot(anios[-10:], predecir(ajustar(t[-10:], yv[-10:])[0], t[-10:]), color="black", lw=1.8,
            label=f"ultimos 10 anios {fila.pendiente_ultimos_10:+.4f}/anio")
    ax.set_title(f"{v}   r2 global = {fila.r2:.3f}", fontweight="bold")
    ax.set_xlabel("anio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[1, 1].bar(anios, serie.canciones, color="#7f8c8d", edgecolor="white", linewidth=0.4)
axes[1, 1].set_title("Canciones por anio en la muestra", fontweight="bold")
axes[1, 1].set_xlabel("anio")
axes[1, 1].grid(alpha=0.3, axis="y")
fig.suptitle(f"Series anuales {inicio}-{fin}", fontsize=15, fontweight="bold")
guardar(fig, "01_series")

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(anios, y, "o-", color="#1DB954", ms=5, lw=1.8, label="observado", zorder=3)
colores = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad"]
for (_, fila), color in zip(ventanas.iterrows(), colores):
    v = int(fila.ventana.split()[0])
    bv = ajustar(t[-v:], y[-v:])[0]
    tramo = np.concatenate([t[-v:], t[-1] + np.arange(1, horizonte + 1)])
    ax.plot(inicio + tramo, predecir(bv, tramo), color=color, lw=1.8,
            label=f"{fila.ventana}: {fila.pendiente_anual:+.4f}/anio, "
                  f"{fin + horizonte} = {fila[f'pronostico_{fin + horizonte}']:.2f}")
ax.axvline(fin + 0.5, color="black", ls=":", lw=1.2)
ax.set_xlabel("anio")
ax.set_ylabel(f"{objetivo} (minutos)")
ax.set_title(f"La ventana de ajuste cambia el pronostico: cuatro rectas sobre la misma serie",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
guardar(fig, "02_ventanas")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
axes[0].plot(anios[-prueba:], y[-prueba:], "o-", color="black", ms=7, lw=2.5,
             label="observado", zorder=5)
color_modelo = dict(zip(modelos, plt.cm.tab10(np.linspace(0, 1, len(modelos)))))
for nombre in validacion.modelo:
    g = detalle[detalle.modelo == nombre]
    axes[0].plot(g.anio, g.predicho, "o--", ms=3, lw=1, color=color_modelo[nombre],
                 alpha=0.8, label=nombre)
axes[0].set_xlabel("anio")
axes[0].set_ylabel(f"{objetivo} (minutos)")
axes[0].set_title("Prediccion a un paso, origen que avanza", fontweight="bold")
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(alpha=0.3)

for nombre in validacion.modelo:
    g = largo[largo.modelo == nombre]
    g = g[g.origen.isin(ultimos)]
    por_paso = [mae(g[g.pasos == h].observado.values, g[g.pasos == h].predicho.values)
                for h in range(1, horizonte + 1)]
    axes[1].plot(range(1, horizonte + 1), por_paso, "o-",
                 color=color_modelo[nombre], lw=2.5 if nombre == elegido else 1.2,
                 alpha=1 if nombre == elegido else 0.6, label=nombre)
axes[1].set_xticks(range(1, horizonte + 1))
axes[1].set_xlabel("pasos adelante")
axes[1].set_ylabel("MAE (minutos)")
axes[1].set_title(f"Error por horizonte, origenes {ultimos[0]}-{ultimos[-1]}", fontweight="bold")
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(alpha=0.3)

orden = validacion.sort_values("mae_recientes", ascending=False)
axes[2].barh(orden.modelo, orden.mae_recientes,
             color=["#1DB954" if m == elegido else "#95a5a6" for m in orden.modelo],
             edgecolor="black", linewidth=0.6)
for i, v in enumerate(orden.mae_recientes):
    axes[2].text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
axes[2].set_xlabel(f"MAE a {horizonte} pasos, origenes {ultimos[0]}-{ultimos[-1]}")
axes[2].set_title(f"Mejor modelo: {elegido}", fontweight="bold")
axes[2].grid(alpha=0.3, axis="x")
fig.suptitle(f"Validacion con origen rodante ({origenes} origenes)", fontsize=15, fontweight="bold")
guardar(fig, "03_validacion")

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(anios, y, "o-", color="#1DB954", ms=5, lw=1.8, label="observado")
ax.plot(errores.anio, errores.predicho, "o--", color="#c0392b", ms=5, lw=1.5,
        label=f"prediccion a un paso ({elegido})")
ax.plot(pronostico.anio, p, "s--", color="#c0392b", ms=8, lw=2, label="pronostico")
ax.fill_between(pronostico.anio, lo, hi, color="#c0392b", alpha=0.15,
                label="+-1.96 x RMSE de validacion por paso")
ax.axhline(y[-1], color="gray", ls=":", lw=1.2, label=f"ultimo observado ({y[-1]:.2f})")
ax.axvline(fin + 0.5, color="black", ls=":", lw=1.2)
ax.annotate(f"{p[-1]:.2f} min", (pronostico.anio.iloc[-1], p[-1]), textcoords="offset points",
            xytext=(8, 0), fontsize=10, fontweight="bold", color="#c0392b")
ax.set_xlabel("anio")
ax.set_ylabel(f"{objetivo} (minutos)")
ax.set_title(f"Pronostico de la duracion media {fin + 1}-{fin + horizonte} con {elegido}",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
guardar(fig, "04_pronostico")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].stem(errores.anio, res, basefmt=" ", linefmt="#2980b9", markerfmt="o")
axes[0].axhline(0, color="#c0392b", lw=1.5)
axes[0].set_xlabel("anio")
axes[0].set_ylabel("observado - predicho")
axes[0].set_title(f"Error a un paso (Durbin-Watson = {dw:.2f})", fontweight="bold")
axes[0].grid(alpha=0.3)

axes[1].hist(res, bins=8, color="#2980b9", edgecolor="white", linewidth=0.6)
axes[1].axvline(res.mean(), color="#c0392b", ls="--", lw=1.5,
                label=f"sesgo medio {res.mean():+.3f}")
axes[1].set_xlabel("error")
axes[1].set_title(f"Distribucion del error (s = {res.std(ddof=1):.3f})", fontweight="bold")
axes[1].legend(fontsize=8)

axes[2].scatter(res[:-1], res[1:], s=45, color="#2980b9", edgecolors="black", linewidths=0.6)
axes[2].axhline(0, color="gray", lw=0.8)
axes[2].axvline(0, color="gray", lw=0.8)
axes[2].set_xlabel("error en t")
axes[2].set_ylabel("error en t+1")
axes[2].set_title(f"Autocorrelacion de orden 1 = {auto1:.3f}", fontweight="bold")
axes[2].grid(alpha=0.3)
fig.suptitle(f"Diagnostico del modelo elegido ({elegido})", fontsize=15, fontweight="bold")
guardar(fig, "05_residuos")

print(f"\n{len(os.listdir('resultados'))} archivos en resultados/, {len(os.listdir('graficas'))} graficas")
