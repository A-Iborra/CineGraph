import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import requests
from bs4 import BeautifulSoup
from datapackage import Package
from sqlalchemy import create_engine