from fastapi import FastAPI

import pandas as pd
import json
import re

from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity

#Instanciar FastAPI
app = FastAPI()
juegos, items, reviews, games_data = None, None, None, None

#Inicializar los modelos al empezar
juegos = pd.read_csv('Datasets/games.csv')
reviews = pd.read_csv('Datasets/reviews.csv')
items_arr = []
for i in range(6):
    items_arr.append(pd.read_csv(f'Datasets/items{i}.csv'))
items = pd.concat(items_arr)

patron = re.compile(r'\d+.*\d*')

juegos.release_date = pd.to_datetime(juegos.release_date)
juegos['year'] = juegos.release_date.dt.year
juegos["free"] = juegos.price.apply(lambda x: "free to play".find(x.lower().strip()) >= 0)
juegos["price"] = juegos.price.apply(lambda x: float(x) if patron.fullmatch(x) else 0)
juegos["price"] = juegos.price.astype(float, errors='ignore')

#Matriz de similaridad de juegos
def init_similarity_games():
    try:
        global games_data
        games_data = juegos.copy()
        games_data = games_data[["item_id","price","free","year","genres"]]
        genres = pd.read_csv('Datasets/genres.csv')
        
        for gen in genres.genre:
            games_data[gen] = games_data.genres.apply(lambda x: 1 if gen in x else 0)
        games_data.drop(columns=['genres'], inplace=True)
   
        games_data.free = games_data.free.astype(int)
    
        games_data.set_index("item_id",inplace=True)
    
        games_data.year = games_data.year.fillna(games_data.year.median()).astype(int)
        games_data.year = minmax_scale(games_data.year)
        
        # Convert to sparse matrix to reduce memory usage
        global games_data_sparse
        games_data_sparse = sparse.csr_matrix(games_data)
    except:
        return

#Crear matrices de similaridad de los juegos
init_similarity_games()

#Método de la página raíz
@app.get("/")
def root():
    return "Hola Mundo!"

@app.get("/index")
def index():
    return """Los métodos de búsqueda disponibles son: 
    1. Información de un desarrollador; 
    2. Información de un usuario; 
    3. year con más horas jugadas para un género; 
    4. Usuario con más horas jugadas para un género; 
    5. Top 3 de juegos recomendados para un year; 
    6. Top 3 de desarrolladores con más juegos recomendados para un year; 
    7. Cantidad de reseñas para un desarrollador
    """

"""
  Cantidad de items y porcentaje de contenido Free por year según empresa desarrolladora
"""
@app.get("/developer/{developer}")
def developer(developer: str):
    #Obtener la información de los juegos
    df = juegos[juegos["developer"] == developer][["year", "app_name", "free"]]
    
    #Agrupar por year y contar la cantidad de items y
    conteo = df.groupby("year")["app_name"].count().reset_index().sort_values(by="year")
    conteo.year = conteo.year.astype(int)
    
    #Agrupar por year y contar el porcentaje de Free to Play
    free = df.groupby("year")["free"].sum().reset_index().sort_values(by="year")
    free.year = free.year.astype(int)
    
    #Unir los dos dataframes
    conteo = pd.merge(conteo, free, on="year")
    conteo.free = round(conteo.free/conteo.app_name*100,2)
    return conteo.to_dict()

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
  Devolver el  year con más horas jugadas para un género dado.
  Input: genero, string
"""
@app.get("/play_time_genre/{genero}")
def play_time_genre(genero: str):
    #Obtener la información de los juegos y los items (que contiene las horas jugadas por juego y por usuario)
    df = pd.merge(juegos, items, on='item_id')
    
    # - Filtrar por el género dado y seleccionar las columnas year y playtime_forever (horas totales jugadas)
    # - Agrupar por year y sumar las horas jugadas por year
    # - Ordenar por horas jugadas de mayor a menor y seleccionar el primer elemento (year con más horas jugadas)
    df = df[df.genres.str.contains(genero)][["year", "playtime_forever"]]\
    .groupby("year").sum()\
    .sort_values(by="playtime_forever", ascending=False)\
    .reset_index()
    df.year = df.year.astype(int)
    
    #year con más horas jugadas
    a = int(df.iloc[0]["year"])
    
    #Retorno
    return {f"year de lanzamiento con más horas jugadas para Género {genero}":a}

"""
|  Devolver el usuario con más horas jugadas para un género dado.
|  Input: genero, string
"""
@app.get("/user_for_genre/{genero}")
def user_for_genre(genero: str):
    #Obtener la información de los juegos y los items (que contiene las horas jugadas por juego y por usuario)
    df = pd.merge(juegos, items, on='item_id')
    
    # - Filtrar por el género dado y seleccionar las columnas user_id, year y playtime_forever (horas totales jugadas)
    # - Agrupar por year y sumar las horas jugadas por year
    df = df[df.genres.str.contains(genero)][["user_id", "year", "playtime_forever"]]\
    .groupby(["year", "user_id"]).sum().reset_index()
    df.year = df.year.astype(int)
    
    # Obtener la suma del tiempo jugado por usuario
    sum_usr = df.groupby("user_id")["playtime_forever"].sum().sort_values(ascending=False)
    # Obtener el usuario con más horas jugadas
    max_usr = sum_usr.idxmax()
    
    #Obtener las horas jugadas por el usuario en cada year y ordenar por year
    df = df[df.user_id == max_usr].loc[:,["year", "playtime_forever"]]\
    .set_index("year").sort_index()
    horas = [{"year":row.year, "Horas":row.playtime_forever} for row in df.reset_index().itertuples()]
    
    #Retornar el usuario y las horas jugadas por year como un diccionario
    return {
      "Usuario con más horas jugadas": f"{max_usr}",
      "Horas jugadas por year": horas
    }

"""
  Top 3 de juegos recomendados por usuarios para el year dado.
  Input: year, int
"""
@app.get("/users_recommend/{year}")
def users_recommend(year: int):
    #Obtener la información de los juegos y las reviews (que contiene las recomendaciones y el análisis de sentimientos)
    df = pd.merge(juegos, reviews, on='item_id')
    
    #Obtener las columnas necesarias
    df = df[df["year"] == year][["app_name", "recommend", "sentiment_analysis"]]
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
  Top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el year dado.
  Input: year, int
"""
@app.get("/best_developer_year/{year}")
def best_developer_year(year: int):
    #Obtener la información de los juegos y las reviews (que contiene las recomendaciones y el análisis de sentimientos)
    df = pd.merge(juegos, reviews, on='item_id')
    
    #Obtener las columnas necesarias
    df = df[df["year"] == year][["developer", "recommend", "sentiment_analysis"]]
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
  Input: year, int
"""
@app.get("/developer_reviews_analysis/{dev}")
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

"""
    Recomendar 5 juegos similares al ingresado
    Input: id del juego, int
"""
@app.get("/recomendacion_juego/{item_id}")
def recomendacion_juego(item_id: int):
    try:
        # Calculate cosine similarity on the fly
        similarity = cosine_similarity(games_data_sparse[games_data.index.get_loc(item_id)], games_data_sparse).flatten()
        games_rec = pd.Series(similarity, index=games_data.index).sort_values(ascending=False).drop(item_id).head(5).index.to_list()
        for i in range(5):
            if games_rec[i] in games.item_id.values:
                games_rec[i] = games.loc[games['item_id'] == games_rec[i], 'app_name'].values[0]
    
        return games_rec
    except Exception as e:
        return {"Error: ":str(e)}
