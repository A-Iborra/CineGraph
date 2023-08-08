from flask import Flask, jsonify, render_template
import pandas as pd
import json

app = Flask(__name__,template_folder='templates')

# Chargez le fichier JSON contenant graph_data
with open('test.json', 'r') as json_file:
    data = json.load(json_file)


@app.route("/")
def index():
    return render_template("index.html",data_film = json.dumps(data))

if __name__ == "__main__":
    app.run(debug=True)

