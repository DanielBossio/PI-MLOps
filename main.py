from fastapi import FastAPI

import pandas as pd
import json
import re

from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity

#Instanciar FastAPI
app = FastAPI()
juegos, items, reviews, similarity_users, similarity_games, users_vs_games = None, None, None, None, None, None

#Matriz de similaridad de juegos
async def init_similarity_games():
    global similarity_games
    juegos_data = juegos["item_id","price","free","year","genres"]
    generos = pd.read_csv('genres.csv')
    
    #Codificar la columna géneros, indicando con 1 o 0 si el juego  contiene o no la categoría
    for gen in generos.genre:
        juegos_data[gen] = juegos_data.genres.apply(lambda x: 1 if gen in x else 0)
    juegos_data.drop(columns=['genres'], inplace=True)

    #Cambiar la columna Free a entero
    juegos_data.free = juegos_data.free.astype(int)

    #Poner como índice el item_id
    juegos_data.set_index("item_id",inplace=True)

    #Imputar faltantes en la columna year y cambiar a entero
    juegos_data.year.fillna(juegos_data.year.median(),inplace=True)
    juegos_data.year = juegos_data.year.astype(int)

    #Escalar los datos(columna year)
    juegos_data.year = minmax_scale(juegos_data.year)
    #Sparsity de los datos: 0.18

    #Matriz de similaridad
    similarity_games = cosine_similarity(juegos_data)
    similarity_games = pd.DataFrame(similarity_games, index=juegos_data.index, columns=juegos_data.index)

#Matriz de similaridad de usuarios
async def init_similarity_users():
    global similarity_users, users_vs_games
    #Inicializar la matriz de similaridad de juegos si no se ha hecho aún
    if similarity_games is None:
        await init_similarity_games()

    #El puntaje a operar es el promedio entre el análisis de sentimientos y si se recomienda o no (entre 0 y 1.5)
    reviews_data = reviews["item_id","user_id","recommend","sentiment_analysis"]
    reviews_data["puntaje"] = (reviews_data.sentiment_analysis+reviews_data.recommend)/2
    reviews_data.drop(columns=["sentiment_analysis","recommend"],inplace=True)

    #Crear tabla pivote con el puntaje de los juegos por usuario, sparsity = 6*10^-4
    users_vs_games = reviews_data.pivot_table(index="user_id",columns="item_id",values="puntaje").fillna(0)

    #Utilizar las horas jugadas por los usuarios para agregar datos de estudio, sparsity = 0.016
    cols = users_vs_games.index
    for item in users_vs_games.columns:
        #Obtener los usuarios y las horas jugadas para cada juego/item
        items_usr = items[items.item_id == item][["playtime_forever", "user_id"]].drop_duplicates(subset="user_id").set_index("user_id")["playtime_forever"]
        #Para cada usuario, multiplicar el puntaje actual por las horas jugadas y asignar a la tabla pivote
        #Si jugó más de 350 horas (75%), el puntaje se multiplica por 10
        #De esta forma, si el usuario jugó muchas horas un juego pero tiene reseñas negativas y/o no se recomienda, no se considerará
        for user in [x for x in items_usr.index if x in cols]:
            hrs = items_usr[user]
            if hrs >= 350:
                users_vs_games.loc[user,item] *= 10
            else:
                users_vs_games.loc[user,item] *= hrs/35 if hrs is not None else 1

    #Matriz de similaridad de usuarios
    similarity_users = cosine_similarity(users_vs_games)
    similarity_users = pd.DataFrame(similarity_users, index=users_vs_games.index, columns=users_vs_games.index)

#Inicializar los modelos al empezar
@app.on_event("startup")
async def startup_event():
    global juegos, items, reviews
    juegos = pd.read_csv('games.csv')
    reviews = pd.read_csv('reviews.csv')
    items_arr = []
    for i in range(6):
        items_arr.append(pd.read_csv(f'items{i}.csv'))
    items = pd.concat(items_arr)

    patron = re.compile(r'\d+.*\d*')

    juegos.release_date = pd.to_datetime(juegos.release_date)
    juegos['year'] = juegos.release_date.dt.year
    juegos["free"] = juegos.price.apply(lambda x: "free to play".find(x.lower().strip()) >= 0)
    juegos["price"] = juegos.price.apply(lambda x: float(x) if patron.fullmatch(x) else 0)
    juegos["price"] = juegos.price.astype(float, errors='ignore')

    #Crear matrices de similaridad de los juegos y de los usuarios
    await init_similarity_games()
    await init_similarity_users()

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
    .groupby(["year", "user_id"]).sum()
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
    #Si la matriz de similaridad de juegos no está inicializada, hacerlo
    if similarity_games is None:
        await init_similarity_games()

    #Buscar los juegos similares
    juegos_rec = similarity_games[item_id].sort_values(ascending=False).drop(item_id).head(5).index.to_list()
    #Si los juegos recomendados están dentro de la información de juegos disponible, reemplazar la id por el nombre
    for i in range(5):
        if juegos_rec[i] in juegos.item_id:
            juegos_rec[i] = juegos[juegos.item_id == item_id].app_name.values[0]

    #Retornar
    return juegos_rec

"""
    Recomendar 5 juegos para el usuario ingresado
"""
@app.get("/recomendacion_usuario/{user_id}")
def recomendacion_usuario(user_id: str):
    #Si la matriz de similaridad de usuarios no está inicializada, hacerlo
    #Tener en cuenta que también se inicializa la matriz users_vs_games
    if similarity_users is None:
        await init_similarity_users()

    num_usrs = 5
    iters = 10
    juegos_rec = {}
    #Juegos jugados por el usuario
    juegos_user = users_vs_games.loc[user_id]
    juegos_user = juegos_user[juegos_user > 0].index.to_list()
    #Buscar los usuarios con gustos similares, limitando a 10 iteraciones
    while iters > 0:
        #Encontrar n usuarios
        usr_similares = similarity_users[user_id].sort_values(ascending=False).drop(user_id).head(num_usrs).index.to_list()
        #Buscar los juegos jugados por esos usuarios que no hayan sido jugados por el usuario ingresado
        for usr in usr_similares.index:
            for item, val in users_vs_games.loc[usr].sort_values(ascending=False).head(num_usrs).items():
                #Comparar que el juego no haya sido jugado por el usuario
                if item not in juegos_user:
                    #Utilizar el valor en users_vs_games como medida de recomendación
                    if item in juegos_rec:
                        juegos_rec[item] += val
                    else:
                        juegos_rec[item] = val
        if len(juegos_rec) >= 5: break
        num_usrs += 5
        iters -= 1

    #Obtener los 5 juegos más recomendados
    juegos_rec = pd.Series(juegos_rec).sort_values(ascending=False).head(5).index.to_list()
    #Si los juegos recomendados están dentro de la información de juegos disponible, reemplazar la id por el nombre
    for i in range(len(juegos_rec)):
        if juegos_rec[i] in juegos.item_id:
            juegos_rec[i] = juegos[juegos.item_id == item].app_name.values[0]

    #Retornar
    return juegos_rec
