# web_scraping.py

import requests
from bs4 import BeautifulSoup
import mysql.connector

def scrape_movie_data():
    # Code pour le web scraping (comme dans l'exemple précédent)
    # ...

    # Connexion à la base de données MySQL via phpMyAdmin
    db_connection = mysql.connector.connect(
        host="localhost",     # Remplacez par votre hôte MySQL
        user="your_username", # Remplacez par votre nom d'utilisateur MySQL
        password="your_password", # Remplacez par votre mot de passe MySQL
        database="film_database" # Remplacez par le nom de votre base de données
    )

    # Création du curseur pour exécuter les requêtes SQL
    cursor = db_connection.cursor()

    # Enregistrement des données de films dans la base de données
    for movie in movie_data:
        title = movie["title"]
        year = movie["year"]
        # ... autres détails du film que vous souhaitez enregistrer dans la base de données

        # Exemple d'une requête d'insertion en utilisant SQL
        sql_query = "INSERT INTO movies (title, year) VALUES (%s, %s)"
        values = (title, year)
        cursor.execute(sql_query, values)

    # Valider les changements et fermer la connexion
    db_connection.commit()
    db_connection.close()
