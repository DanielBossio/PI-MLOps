# PI-MLOps
## Proyecto Individual #1
## Bootcamp Data Science - Henry
## *Daniel Andrés Bossio Pérez - DataFT21*
# 1. Proceso ETL
Archivo principal: [PI-MLOps-ETL]()<br>
Se proporcionaron tres datasets: games, reviews e items.<br>
*Para el archivo games:*
- Se leyó el archivo con pandas.read_json y se eliminaron las filas nulas.
- Se utilizaron los valores de las columnas tags y specs para imputar nulos en la columna genres.
- Se seleccionaron las columnas id, app_name, genres, price, release_date y developer.
- Se trataron las filas duplicadas y los nulos en las columnas aparte de genres.
- Se convirtieron las columnas a los tipos adecuados y se renombró la columna id como item_id.
- Se guardó el dataframe de juegos y la lista de géneros.<br><br>
*Para el archivo reviews:*
- Se leyó el archivo como un diccionario con gzip.open.
- Se convirtió el diccionario a dataframe y se obtuvieron las reviews.
- Se eliminaron los duplicados y se seleccionaron las columnas item_id, recommend, review y user_id.
- Se realizó el procesamiento y el análisis de sentimientos sobre la columna review, guardando los resultados en la columna sentiment_analysis.
- Se guardó el dataframe.<br><br>
*Para el archivo items:*
- Se leyó el archivo como un diccionario con gzip.open.
- Se convirtió el diccionario a dataframe y se obtuvieron las horas de juego.
- Se eliminaron los duplicados y se seleccionaron las columnas item_id, playtime_forever, user_id (por cuestiones del volumen de los datos).
- Se guardó el dataframe, dividiéndolo en 6 archivos.
# 2. Desarrollo de la API
Archivo principal: [main.py](main.py)
<br>La API fue implementada utilizando el framework FastAPI y disponibilizada con el servicio en la nube Render. Además de mensajes root e index, las funciones desarrolladas son:
- **developer**: Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. Parámetro: developer, string.
- **userdata**: Cantidad de dinero gastado por el usuario, porcentaje de recomendación en base a reviews.recommend y cantidad de items del usuario. Parámetro: user, string.
- **play_time_genre**: Año con más horas jugadas para un género dado. Parámetro: género, string.
- **user_for_genre**: Usuario con más horas jugadas para un género dado, y horas jugadas por año de dicho usuario. Parámetro: género, string.
- **users_recommend**: Top 3 de juegos recomendados por usuarios para el año dado. Parámetro: year, int.
- **best_developer_year**: Top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado. Parámetro: year, int.
- **developer_reviews_analysis**: Resumen de la cantidad de reseñas positivas, neutrales y negativas para un desarrollador. Parámetro: year, int.
# 3. EDA
Archivo principal: [PI-MLOps-EDA]()<br>
En resumen, el EDA hecho para los archivos es el siguiente:
- Archivo games (juegos): Revisar medidas estadísticas (media, std, cuartiles), revisar rangos de valores de precios y años, frecuencias por free_to_play, géneros y desarrolladoras, promedio y mediana de los precios entre los desarrolladores y entre los géneros, cantidad de juegos por año para los desarrolladores.
- Archivo reviews: Revisar medidas estadísticas (media, std, cuartiles), frecuencias de recomendaciones y valoraciones positivas, neutrales o negativas.
- Archivo items: Revisar medidas estadísticas (media, std, cuartiles), distribución de valores de la cantidad de horas jugadas.
# 4. Modelo de ML
Archivo principal: [PI-MLOps-Modelo](), [main.py](main.py)<br>
Se planteó desarrollar un sistema de recomendación para proponer ítems/juegos similares en base al id de un ítem o un usuario proporcionado, y desplegarlos como endpoints en el API en Render.<br>
A continuación se explica el funcionamiento de los endpoint:<br>
- **recomendacion_juego**: En base a la operación similitud del coseno, recomendar 5 juegos similares al ingresado como parámetro *(item_id, string)*. Al iniciar la aplicación se calcula la matriz de similitudes de juegos, y cuando se presenta una solicitud se busca en la misma los juegos con mayor similitud al ingresado.
- **recomendacion_usuario**: Recomendar 5 juegos basándose en los gustos del usuario ingresado como parámetro *(user_id, string)*. Al iniciar la aplicación, se calcula el peso de cada juego por cada usuario *(matriz users_vs_games)*, utilizando el valor de la recomendación (True o False), el análisis de sentimientos (0 a 2) y las horas jugadas; asimismo se calcula la matriz de similitudes de usuarios utilizando la operación similitud del coseno. Cuando se presenta una petición, se buscan los usuarios con gustos similares al ingresado, y se recomiendan los juegos jugados por dichos usuarios, que no hayan sido jugados por el usuario objetivo.
