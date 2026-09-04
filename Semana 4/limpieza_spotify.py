# Semana 4 - Limpieza del dataset 

import pandas as pd

df = pd.read_csv("../spotify_songs.csv")
filas_iniciales = len(df)

# Quito las 5 filas a las que les falta nombre
df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])

# La fecha viene en 3 formatos distintos (solo año, año-mes y año-mes-dia), asi que a las incompletas les pongo 01 para poder convertirlas
fecha = df["track_album_release_date"].astype(str)
fecha = fecha.where(fecha.str.len() != 4, fecha + "-01-01")
fecha = fecha.where(fecha.str.len() != 7, fecha + "-01")
df["track_album_release_date"] = pd.to_datetime(fecha, errors="coerce")
df["release_year"] = df["track_album_release_date"].dt.year

# Saco los valores imposibles, una cancion no puede tener 0 BPM y las que duran menos de 1 minuto son intros o errores
df = df[df["tempo"] > 0]
df = df[df["duration_ms"] >= 60000]

# El genero es de la playlist y no de la cancion, entonces hay canciones con 2 o mas generos a la vez entonces le asigno el que mas se repite
genero_final = df.groupby("track_id")["playlist_genre"].agg(lambda s: s.mode()[0])
subgenero_final = df.groupby("track_id")["playlist_subgenre"].agg(lambda s: s.mode()[0])
num_playlists = df.groupby("track_id")["playlist_id"].nunique()

# Cada fila es una cancion dentro de una playlist, entonces me quedo con una sola fila por cancion y agarro la de mayor popularidad
df = df.sort_values("track_popularity", ascending=False)
df = df.drop_duplicates(subset=["track_id"], keep="first")

# Pongo el genero corregido y el conteo de playlists, y quito las columnas de playlist porque ya no aplican
df["playlist_genre"] = df["track_id"].map(genero_final)
df["playlist_subgenre"] = df["track_id"].map(subgenero_final)
df["num_playlists"] = df["track_id"].map(num_playlists)
df = df.drop(columns=["playlist_name", "playlist_id"])

# Acomodo las columnas dejando primero los datos de la cancion y luego las caracteristicas de audio
df = df[[
    "track_id", "track_name", "track_artist", "track_popularity",
    "track_album_id", "track_album_name", "track_album_release_date",
    "release_year", "playlist_genre", "playlist_subgenre", "num_playlists",
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]]
df = df.reset_index(drop=True)

df.to_csv("spotify_songs_limpio.csv", index=False)

print("Filas al inicio:", filas_iniciales)
print("Filas al final:", len(df))
print("Filas eliminadas:", filas_iniciales - len(df))
print("Nulos:", df.isna().sum().sum())
print("Duplicados:", df.duplicated(subset=["track_id"]).sum())
