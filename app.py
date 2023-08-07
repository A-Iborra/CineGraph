from flask import Flask, jsonify, render_template
import pandas as pd
import json

app = Flask(__name__,template_folder='templates')

# Charger les résultats du PCA à partir du fichier JSON
with open("pca_results.json", "r") as f:
    pca_results = json.load(f)

# Charger les informations sur les films à partir du fichier CSV
film_data = pd.read_csv("film_data.csv")

@app.route("/")
def index():
    return render_template("index.html", pca_results=pca_results, film_data=film_data.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)