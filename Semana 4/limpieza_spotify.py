import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

df = pd.read_csv("../spotify_songs.csv")
filas_iniciales = len(df)

df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

fecha = df["track_album_release_date"].astype(str)
fecha = fecha.where(fecha.str.len() != 4, fecha + "-01-01")
fecha = fecha.where(fecha.str.len() != 7, fecha + "-01")
df["track_album_release_date"] = pd.to_datetime(fecha, errors="coerce")
df["release_year"] = df["track_album_release_date"].dt.year

df = df[df["tempo"] > 0]
df = df[df["duration_ms"] >= 60000]

def mas_comun(s):
    vc = s.value_counts()
    empatados = vc[vc == vc.max()].index
    return empatados[0] if len(empatados) == 1 else rng.choice(empatados)


genero_de_subgenero = df.drop_duplicates("playlist_subgenre").set_index("playlist_subgenre")["playlist_genre"]
subgenero_final = df.groupby("track_id")["playlist_subgenre"].agg(mas_comun)
genero_final = subgenero_final.map(genero_de_subgenero)
num_playlists = df.groupby("track_id")["playlist_id"].nunique()

df = df.sort_values("track_popularity", ascending=False)
df = df.drop_duplicates(subset=["track_id"], keep="first")

df["playlist_genre"] = df["track_id"].map(genero_final)
df["playlist_subgenre"] = df["track_id"].map(subgenero_final)
df["num_playlists"] = df["track_id"].map(num_playlists)
df = df.drop(columns=["playlist_name", "playlist_id"])

df = df[[
    "track_id", "track_name", "track_artist", "track_popularity",
    "track_album_id", "track_album_name", "track_album_release_date",
    "release_year", "playlist_genre", "playlist_subgenre", "num_playlists",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]]
df = df.reset_index(drop=True)

df.to_csv("../spotify_songs_limpio.csv", index=False)

print("Filas al inicio:", filas_iniciales)
print("Filas al final:", len(df))
print("Filas eliminadas:", filas_iniciales - len(df))
print("Nulos:", df.isna().sum().sum())
print("Duplicados:", df.duplicated(subset=["track_id"]).sum())
print("Generos:", df["playlist_genre"].value_counts().to_dict())
