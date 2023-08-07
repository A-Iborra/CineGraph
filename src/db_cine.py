from imports import *


### Chargement et manipulation des données -----------------------------------------------------------------------------

## Données oscars
data_url = 'https://datahub.io/rufuspollock/oscars-nominees-and-winners/datapackage.json'

# to load Data Package into storage
package = Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
        print (data)


data.to_sql('raw_oscar', engine, if_exists='replace', index=False)


## Données allocine 2021
allocine_2021 = pd.read_csv('data/allocine_movies_2021.csv')
allocine_2021['release_date'] = pd.to_datetime(allocine_2021['release_date'], format='%Y-%m-%d', errors='coerce')
allocine_2021_clean = allocine_2021[allocine_2021['release_date'] < pd.to_datetime('2021-10-01')]
allocine_2021_clean.to_sql('raw_allocine', engine, if_exists='replace', index=False)

""" 
import allocine_dataset_scraper
from allocine_dataset_scraper.scraper import AllocineScraper

scraper = AllocineScraper(
    number_of_pages=150,
    from_page=1,
    output_dir="data",
    output_csv_name="allocine_movies.csv",
    pause_scraping= [0,1],
    append_result=False
)

scraper.scraping_movies()
"""

## Données imdb
imbdb = pd.read_csv('data/Movie-Dataset-Latest.csv')
imbdb.to_sql('raw_imbdb', engine, if_exists='replace', index=False)


### Créatione et manipulation données SQLLite
import sqlite3

# Créer une connexion à la base de données SQLite
engine = create_engine('sqlite:///movies.db')

# Connexion à la base de données
conn = sqlite3.connect('movies.db')

# Création d'un curseur
cursor = conn.cursor()

raw_oscar = pd.read_sql_query("SELECT * FROM raw_oscar", conn)

raw_kaggle = pd.read_sql_query("SELECT * FROM raw_kaggle", conn)

raw_allocine = pd.read_sql_query("SELECT * FROM raw_allocine", conn)

raw_imbdb = pd.read_sql_query("SELECT * FROM raw_imbdb", conn)

# Fermeture de la connexion
conn.close()

raw_kaggle = raw_kaggle.rename(columns={'movie_title': 'title'})
raw_kaggle['title'] = raw_kaggle['title'].str.lower().str.strip()
raw_imbdb['title'] = raw_imbdb['title'].str.lower().str.strip()

select_col1 = ['title', 'production_date', 'genres', 'runtime_minutes', 'director_name',
               'movie_averageRating','Production budget $','Worldwide gross $']

select_col2 = ['title', 'overview', 'popularity', 'vote_average', 'vote_count']

data_clust_v1 = pd.merge(raw_kaggle[select_col1], raw_imbdb[select_col2], on='title', how='inner')

data_clust_v1.reset_index(drop=True, inplace=True)


import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import plotly.express as px

# Conversion de la colonne "production_date" en format de date
data_clust_v1["production_date"] = pd.to_datetime(data_clust_v1["production_date"])

# Prétraitement des données catégorielles "genres" avec One-Hot Encoding
encoder = OrdinalEncoder()
data_clust_v1["genres_encoded"] = encoder.fit_transform(data_clust_v1[["genres"]])


# Mise à l'échelle des variables numériques
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_clust_v1[["runtime_minutes", "movie_averageRating", "Production budget $", "Worldwide gross $"]])

# Fusion des caractéristiques
features = pd.concat([data_clust_v1["genres_encoded"], pd.DataFrame(scaled_features, columns=["runtime_minutes", "movie_averageRating", "Production budget $", "Worldwide gross $"])], axis=1)

# Appliquer l'algorithme K-Prototypes
kproto = KPrototypes(n_clusters=3, init='Huang', n_init=10, verbose=1)
clusters = kproto.fit_predict(features, categorical=[0])

# Réduction de dimensionnalité pour visualisation
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(features)

# Ajout des clusters aux données d'origine
data_clust_v1["cluster"] = clusters

# Afficher les résultats
print(data_clust_v1)

# Visualisation en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Séparer les données pour chaque cluster
for cluster in set(clusters):
    cluster_data = reduced_features[data_clust_v1["cluster"] == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f"Cluster {cluster}")

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA 3D Visualization')
ax.legend()

plt.show()



# Création de la figure interactive avec plotly express
fig = px.scatter_3d(data_clust_v1, x=reduced_features[:, 0], y=reduced_features[:, 1], z=reduced_features[:, 2],
                    color="cluster", hover_data=["title"])

# Afficher la figure interactive
fig.show()

import json

# Enregistrer les résultats du PCA dans un fichier JSON
pca_results = {
    "principal_component_1": reduced_features[:, 0].tolist(),
    "principal_component_2": reduced_features[:, 1].tolist(),
    "principal_component_3": reduced_features[:, 2].tolist()
}

with open("pca_results.json", "w") as f:
    json.dump(pca_results, f)

# Enregistrer les informations sur les films dans un fichier CSV
data_clust_v1.to_csv("film_data.csv", index=False)