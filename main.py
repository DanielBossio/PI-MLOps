from fastapi import FastAPI

import pandas as pd
import json
import re

#Instanciar FastAPI
app = FastAPI()
juegos, items, reviews = None, None, None

#Inicializar los modelos al empezar
@app.on_event("startup")
async def startup_event():
    global juegos, items, reviews
    juegos = pd.read_csv('datasets/games.csv')
    items = pd.read_csv('datasets/items.csv')
    reviews = pd.read_csv('datasets/reviews.csv')

    patron = re.compile(r'\d+.*\d*')

    juegos.release_date = pd.to_datetime(juegos.release_date)
    juegos['año'] = juegos.release_date.dt.year
    juegos["free"] = juegos.price.apply(lambda x: "free to play".find(x.lower().strip()) >= 0)
    juegos["price"] = juegos.price.apply(lambda x: float(x) if patron.fullmatch(x) else 0)
    juegos["price"] = juegos.price.astype(float, errors='ignore')

#Método de la página raíz
@app.get("/")
def root():
    return "Hola Mundo!"

@app.get("/index")
def index():
    return """Los métodos de búsqueda disponibles son:\n
    1. Informción de un desarrollador\n
    2. Información de un usuario\n
    3. Año con más horas jugadas para un género\n
    4. Usuario con más horas jugadas para un género\n
    5. Top 3 de juegos recomendados para un año\n
    6. Top 3 de desarrolladores con más juegos recomendados para un año\n
    7. Cantidad de reseñas para un desarrollador
    """

"""
  Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
"""
@app.get("/developer/{developer}")
def developer(developer: str):
  #Obtener la información de los juegos
  df = juegos[juegos["developer"] == developer][["año", "app_name", "free"]]

  #Agrupar por año y contar la cantidad de items y
  conteo = df.groupby("año")["app_name"].count().reset_index().sort_values(by="año")
  conteo.año = conteo.año.astype(int)

  #Agrupar por año y contar el porcentaje de Free to Play
  free = df.groupby("año")["free"].sum().reset_index().sort_values(by="año")
  free.año = free.año.astype(int)

  #Unir los dos dataframes
  conteo = pd.merge(conteo, free, on="año")
  conteo.free = round(conteo.free/conteo.app_name*100,2)
  return conteo

"""
  cantidad de dinero gastado por el usuario, porcentaje de recomendación en base a reviews.recommend y cantidad de items
"""
@app.get("/user/{user}")
def userdata(user: str):
  #Obtener la información del usuario y los items
  df = pd.merge(juegos, items, on='item_id')
  df = df[df["user_id"] == user]
  df = pd.merge(df, reviews, on=['item_id', 'user_id'])
  df = df[["price", "app_name", "recommend"]]

  #Obtener el total de dinero gastado, número de items y número de items recomendados
  precio = df.price.sum()
  cantidad = df.app_name.count()
  recomendacion = df.recommend.sum() / cantidad * 100

  return {
      "Usuario": user,
      "Dinero gastado": f"{precio} USD",
      "Cantidad de items": cantidad,
      "Porcentaje de recomendación": f"{recomendacion}%"
  }

"""
  Devolver el  año con más horas jugadas para un género dado.
  Input: genero, string
"""
@app.get("/play_time_genre/{genero}")
def play_time_genre(genero: str):
  #Obtener la información de los juegos y los items (que contiene las horas jugadas por juego y por usuario)
  df = pd.merge(juegos, items, on='item_id')

  # - Filtrar por el género dado y seleccionar las columnas año y playtime_forever (horas totales jugadas)
  # - Agrupar por año y sumar las horas jugadas por año
  # - Ordenar por horas jugadas de mayor a menor y seleccionar el primer elemento (año con más horas jugadas)
  df = df[df.genres.str.contains(genero)][["año", "playtime_forever"]]\
  .groupby("año").sum()\
  .sort_values(by="playtime_forever", ascending=False)\
  .reset_index()
  df.año = df.año.astype(int)

  #Año con más horas jugadas
  a = int(df.iloc[0]["año"])

  #Retorno
  return {f"Año de lanzamiento con más horas jugadas para Género {genero}":a}

"""
|  Devolver el usuario con más horas jugadas para un género dado.
|  Input: genero, string
"""
@app.get("/user_for_genre/{genero}")
def user_for_genre(genero: str):
  #Obtener la información de los juegos y los items (que contiene las horas jugadas por juego y por usuario)
  df = pd.merge(juegos, items, on='item_id')

  # - Filtrar por el género dado y seleccionar las columnas user_id, año y playtime_forever (horas totales jugadas)
  # - Agrupar por año y sumar las horas jugadas por año
  df = df[df.genres.str.contains(genero)][["user_id", "año", "playtime_forever"]]\
  .groupby(["año", "user_id"]).sum()
  df.año = df.año.astype(int)

  # Obtener la suma del tiempo jugado por usuario
  sum_usr = df.groupby("user_id")["playtime_forever"].sum().sort_values(ascending=False)
  # Obtener el usuario con más horas jugadas
  max_usr = sum_usr.idxmax()

  #Obtener las horas jugadas por el usuario en cada año y ordenar por año
  df = df[df.user_id == max_usr].loc[:,["año", "playtime_forever"]]\
  .set_index("año").sort_index()
  horas = [{"Año":row.año, "Horas":row.playtime_forever} for row in df.reset_index().itertuples()]

  #Retornar el usuario y las horas jugadas por año como un diccionario
  return {
      "Usuario con más horas jugadas": f"{max_usr}",
      "Horas jugadas por año": horas
  }

"""
  Top 3 de juegos recomendados por usuarios para el año dado.
  Input: año, int
"""
@app.get("/users_recommend/{año}")
def users_recommend(año: int):
  #Obtener la información de los juegos y las reviews (que contiene las recomendaciones y el análisis de sentimientos)
  df = pd.merge(juegos, reviews, on='item_id')

  #Obtener las columnas necesarias
  df = df[df["año"] == año][["app_name", "recommend", "sentiment_analysis"]]
  #Filtrar los juegos recomendados y con reseñas positivas o neutrales
  mask = (df["recommend"] == True) & (df["sentiment_analysis"] > 0)
  df = df[mask]

  #Contar las recomendaciones por juego y ordenar
  df = df.groupby("app_name").count().sort_values(by="sentiment_analysis", ascending=False).reset_index()

  #Guardar los tres juegos más recomendados y retornar
  return [
      {"Puesto 1": f"{df.iloc[0]['app_name']}"},
      {"Puesto 2": f"{df.iloc[1]['app_name']}"},
      {"Puesto 3": f"{df.iloc[2]['app_name']}"}
  ]

"""
  Top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
  Input: año, int
"""
@app.get("/best_developer_year/{año}")
def best_developer_year(año: int):
  #Obtener la información de los juegos y las reviews (que contiene las recomendaciones y el análisis de sentimientos)
  df = pd.merge(juegos, reviews, on='item_id')

  #Obtener las columnas necesarias
  df = df[df["año"] == año][["developer", "recommend", "sentiment_analysis"]]
  #Filtrar los juegos recomendados y con reseñas positivas o neutrales
  mask = (df["recommend"] == True) & (df["sentiment_analysis"] > 0)
  df = df[mask]

  #Contar las recomendaciones por desarrollador y ordenar
  df = df.groupby("developer").count().sort_values(by="sentiment_analysis", ascending=False).reset_index()
  #Guardar los tres desarrolladores más recomendados y retornar
  return [
      {"Puesto 1": f"{df.iloc[0]['developer']}"},
      {"Puesto 2": f"{df.iloc[1]['developer']}"},
      {"Puesto 3": f"{df.iloc[2]['developer']}"}
  ]

"""
  Resumen de la cantidad de reseñas positivas, neutrales y negativas para un desarrollador.
  Input: año, int
"""
def developer_reviews_analysis(dev: str):
  #Obtener la información de los juegos y las reviews (que contiene las recomendaciones y el análisis de sentimientos)
  df = pd.merge(juegos, reviews, on='item_id')

  #Obtener las columnas necesarias
  df = df[df["developer"] == dev][["sentiment_analysis"]].value_counts()
  df.index = ["Positive", "Neutral", "Negative"]

  #Retornar el resumen como un diccionario
  puntajes = []
  for i in df.index:
    puntajes.append(f"{i} = {df[i]}")
  return {f"{dev}":puntajes}

