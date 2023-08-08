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