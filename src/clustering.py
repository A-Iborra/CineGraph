from imports import *
import sqlite3
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import json

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


raw_kaggle = raw_kaggle.rename(columns={'movie_title': 'title'})
raw_kaggle['title'] = raw_kaggle['title'].str.lower().str.strip()
raw_imbdb['title'] = raw_imbdb['title'].str.lower().str.strip()

select_col1 = ['title', 'production_date', 'genres', 'runtime_minutes', 'director_name',
               'movie_averageRating','Production budget $','Worldwide gross $']

select_col2 = ['title', 'overview', 'popularity', 'vote_average', 'vote_count']

data_clust_v1 = pd.merge(raw_kaggle[select_col1], raw_imbdb[select_col2], on='title', how='inner')

data_clust_v1.reset_index(drop=True, inplace=True)
data_clust_v1['id'] = range(0, len(data_clust_v1))

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

# Calculer la matrice des distances entre les films
dist_matrix = np.linalg.norm(reduced_features[:, np.newaxis, :] - reduced_features[np.newaxis, :, :], axis=2)

# Créer les nodes
nodes = []
for i, row in data_clust_v1.iterrows():
    node = {
        "id": str(row["id"]),
        "title": row["title"],
        "genres": row["genres"],
        "production_date": row["production_date"],
        "runtime_minutes": row["runtime_minutes"],
        "director_name": row["director_name"],
        "movie_averageRating": row["movie_averageRating"],
        "Production budget $": row["Production budget $"],
        "overview": row["overview"],
        "cluster": str(row["cluster"])
    }
    nodes.append(node)

links = []
for i in range(len(data_clust_v1)):
    closest_film_idx = np.argsort(dist_matrix[i])[1]  # Index du film le plus proche (à l'exception de lui-même)
    link = {
        "source": str(data_clust_v1.loc[i, "id"]),
        "target": str(data_clust_v1.loc[closest_film_idx, "id"])
    }
    links.append(link)

# Créer l'objet JSON final
graph_data_vf = {
    "nodes": nodes,
    "links": links
}

# Enregistrez graph_data dans un fichier JSON
with open("graph_data_vf.json", "w") as f:
    json.dump(graph_data_vf, f, indent=2)
