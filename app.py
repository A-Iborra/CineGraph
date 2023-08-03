# app.py

from flask import Flask, render_template

app = Flask(__name__)

# Routes pour afficher la page principale et les données du graphe
@app.route('/')
def index():
    # Récupérer les données du clustering à partir de la base de données ou de toute autre source
    clusters = [...]  # Remplacez [...] par les données de clustering réelles
    return render_template('index.html', clusters=clusters)

if __name__ == '__main__':
    app.run(debug=True)
