from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

#aca cargo todos los dataframe que vamos a utilizar
df_1_1 = pd.read_csv(r"datasets/df_funcion3.csv")
df_1_2 = pd.read_csv(r"datasets/df_funcion4.csv")
df_1_3 = pd.read_csv(r"datasets/df_funcion5.csv")
df_1_4 = pd.read_csv(r"datasets/df_funcion6.csv")
# Endpoints 1
@app.get('/PlayTimegenre/{genero}')
def PlayTimegenre(genero: str ):
    """
    Esta Función al introducion cualquier Genero devolvera el año con mas horas jugadas.
    """
    # Verifica que genero sea un string
    if not isinstance(genero, str):
        return {"Error": f"'{genero}' is not a string"}
   
    # Normaliza genero esto con el fin si escribe mayuscula o  minuscula noy hay problema
    genero = genero.strip().title()
    #aca voy primero a flitar el genero que me piden:
    df_generoAccion =df_1_1[df_1_1['genres'] == genero]
    #ahora vamos agrupas los datos por año y hacemos la suma de cada año
    anio_mayor = str(df_generoAccion.groupby('release_date')['suma_playtime_forever'].sum().idxmax())
    resultado1= {"Para el genero dado ":genero}
    return resultado1,anio_mayor

# Endpoints 2 

@app.get('/UserForGenre/{genero}')
def UserForGenre( genero : str ):
    """
    Esta función devuelve el usuario que acumula mas horas jugadas para el genero dado, y una 
    lista de acumulación de horas jugadas por año de lanzamiento.
    """

    # Verifica que genero sea un string
    if not isinstance(genero, str):
        return {"Error": f"'{genero}' is not a string"}
   
    # Normaliza genero esto con el fin si escribe mayuscula o  minuscula noy hay problema
    genero = genero.strip().title()
    
    
    #Filtro primero el genero  que piden:
    f_generoAccion = df_1_1[df_1_1['genres'] == genero]
    
    # aca vamos agrupar los datos por user_id y sumamos  la columna suma_playtime_forever para luego buscar el usuario que mas sume
    usuario_mas_horas = f_generoAccion.groupby('user_id')['suma_playtime_forever'].sum().idxmax()
   
    #ahora hacemos un dataframe con el usuario  obtenido en el paso anterior.
    df_usuario_max_horas=f_generoAccion[f_generoAccion['user_id']==usuario_mas_horas]
    
    # En este ultimo paso vamos agrupar la suma total se horas jugadas por cada una de las fechas.
    df_agrupado_anio = df_usuario_max_horas.groupby('release_date')['suma_playtime_forever'].sum()
    diccionario = df_agrupado_anio.to_dict()
    resultado2 = {"Usuario con más horas jugadas para Género ":genero}
    return resultado2,usuario_mas_horas,diccionario


# Endpoint 3

@app.get('/UsersRecommend/{anio}')
def UsersRecommend( anio : int ):
    # Filtrar el dataset por el año  deseado
    df_filtrado = df_1_2[df_1_2['posted'] == anio]
    # Agrupar los datos por el app_name  y la columna boolena recomend  
    df_agrupado = df_filtrado.groupby('app_name')['recommend'].value_counts().reset_index()
    df_ordenado = df_agrupado.sort_values(by='count', ascending=False)
    app_names = df_ordenado['app_name'].head(3).tolist()
    # Crear una lista de diccionarios en el formato requerido
    lista_puestos = [{"Puesto " + str(i+1): app_name} for i, app_name in enumerate(app_names)]
    return lista_puestos

#Endpoint 4
@app.get('/UsersWorstDeveloper/{anio}')
def UsersWorstDeveloper(anio:int):
     # Filtrar el dataset por el año  deseado
    df_filtrado = df_1_2[df_1_2['posted'] == anio]
    df_grouped = df_filtrado.groupby(['user_id', 'recommend'], as_index=False).agg({'developer': 'first', 'recommend': 'count'})
    df_sorted = df_grouped.sort_values(by='recommend', ascending=True)
    app_developer = df_sorted['developer'].head(3).tolist()
    # Crear una lista de diccionarios en el formato requerido
    lista_puestos = [{"Puesto " + str(i+1): app_developer} for i, app_developer in enumerate(app_developer)]
    return lista_puestos

#Endpoint 5
@app.get('/Sentiment_analisis/{desarrolladora}')
def Sentiment_analisis( desarrolladora : str ): 
    # Filtrar las filas del DataFrame para  el developer dado
    df_filtered = df_1_3[df_1_3['developer'] == desarrolladora]
    # Cuenta la cantidad de registros para cada categoría de sentimiento
    sentiment_counts = df_filtered['sentiment-analysis'].value_counts()
    # Convierte el resultado en un diccionario
    result = {
    desarrolladora:{    
    'Negative': int(sentiment_counts.get(0, 0)),
    'Neutro': int(sentiment_counts.get(1,0)),
    'Positive': int(sentiment_counts.get(2, 0))}
    }
    return result


# Modelo de Machine LEarning:

# Crear un vectorizador TF-IDF para convertir títulos en vectores numéricos
# Crear una matriz TF-IDF para representar los juegos
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_1_4.head()['title'] + ' ' + df_1_4.head()['tags'] + ' ' + df_1_4.head()['genres'])

# Calcular la similitud de coseno entre los juegos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Endpoints 6


# Función definida para el sistema de recomendación item-item:
@app.get('/recomendacion_juego/{title}')
def recomendacion_juego(title, num_recommendations=5):
    """
    Esta función  devuelve 5 juegos recomendados. Esta función se hizo con la similitud del coseno entre los juegos.
    Para esta función se  tuvo que crear un vectorizador TF_IDF y una matriz TF-IDF.
    """
    idx = df_1_4[df_1_4['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    num_recommendations = int(num_recommendations)
    sim_scores = sim_scores[1:num_recommendations+1]  
    game_indices = [i[0] for i in sim_scores]
    resultado3 = {"Juegos recomendados  por  Item-Item ":title}
    diccionario2 = df_1_4['title'].iloc[game_indices].tolist()
    return resultado3,diccionario2