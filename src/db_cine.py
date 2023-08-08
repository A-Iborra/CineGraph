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



import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

numeric_columns = ['runtime_minutes', 'movie_averageRating', 'Production budget $', 'Worldwide gross $']
numeric_data = data_clust_v1[numeric_columns]

numeric_data.fillna(0, inplace=True)  # Remplacer les valeurs manquantes par 0


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)

data_with_pca = data_clust_v1.copy()
data_with_pca['x'] = pca_result[:, 0]
data_with_pca['y'] = pca_result[:, 1]
data_with_pca['z'] = pca_result[:, 2]

graph_data = {
    "nodes": [],
    "links": []
}

for index, row in data_with_pca.iterrows():
    node = {
        "id": index,
        "name": row['title'],
        "x": row['x'],
        "y": row['y'],
        "z": row['z']
    }
    graph_data["nodes"].append(node)

import json

# Chemin vers le fichier JSON de sauvegarde
output_json_file = 'graph_datav2.json'

# Enregistrez graph_data dans un fichier JSON
with open(output_json_file, 'w') as json_file:
    json.dump(graph_data, json_file)


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Chargement des données
data_clust_v1 = pd.read_csv('data_clust_v1.csv')

# Sélectionnez les variables pour l'analyse PCA
# Ici, nous sélectionnons toutes les colonnes sauf 'title' qui n'est pas nécessaire pour la réduction de dimension
# Remplacez les colonnes par celles que vous voulez inclure dans l'analyse
X = data_clust_v1.drop(columns=['title'])

# Effectuez l'analyse PCA pour réduire les dimensions à 3
pca = PCA(n_components=3)
pca_results = pca.fit_transform(scaled_data)

# Effectuez le clustering avec KMeans pour attribuer chaque film à un cluster
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(features)

# Créez un DataFrame avec les coordonnées réduites et les clusters associés
graph_data = pd.DataFrame({'x': reduced_features[:, 0], 'y': reduced_features[:, 1], 'z': reduced_features[:, 2], 'cluster': clusters})

# Convertissez le DataFrame en un dictionnaire au format requis pour ForceGraph3D
graph_data_dict = {'nodes': graph_data.to_dict(orient='records')}

# Utilisez le dictionnaire pour créer le graph_data dans le format attendu par ForceGraph3D
graph_data = {'nodes': graph_data_dict['nodes'], 'links': []}

# Ajouter les coordonnées réduites et les clusters dans le DataFrame data_clust_v1
data_clust_v1['x'] = reduced_features[:, 0]
data_clust_v1['y'] = reduced_features[:, 1]
data_clust_v1['z'] = reduced_features[:, 2]
data_clust_v1['cluster'] = clusters

nodes = data_clust_v1.to_dict(orient='records')

distances = np.zeros((len(data_clust_v1), len(data_clust_v1)))
for i in range(len(data_clust_v1)):
    for j in range(i+1, len(data_clust_v1)):
        dist = np.linalg.norm(reduced_features[i] - reduced_features[j])
        distances[i, j] = dist
        distances[j, i] = dist

# Créer les liens en utilisant la distance minimale
# Créer les liens en utilisant la distance minimale
links = []
for i in range(len(data_clust_v1)):
    nearest_film_index = np.argsort(distances[i])[1]
    link = {'source': data_clust_v1.iloc[i]['title'], 'target': data_clust_v1.iloc[nearest_film_index]['title']}
    links.append(link)


graph_data = {
    'nodes': nodes,
    'links': links
}

graph_data
