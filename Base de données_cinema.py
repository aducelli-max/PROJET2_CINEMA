
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1) Chargement optimisé
# -----------------------------
N = 2_000_000

names = pd.read_csv("name_basics.tsv.gz", sep="\t", compression="gzip", na_values="\\N", nrows=N)

titles = pd.read_csv("title.basics.tsv.gz", sep="\t", compression="gzip", na_values="\\N", nrows=N)

# Retirer les contenus adultes
titles = titles[titles["isAdult"] == 0]

# Garder uniquement les colonnes utiles (inclut GENRES)
titles = titles[["tconst", "primaryTitle", "titleType", "startYear", "genres"]]

principals = pd.read_csv("title.principals.tsv.gz", sep="\t", compression="gzip", na_values="\\N", nrows=N)
ratings = pd.read_csv("title.ratings.tsv.gz", sep="\t", compression="gzip", na_values="\\N", nrows=N)

# -----------------------------
# 2) Filtrer les vrais acteurs
# -----------------------------

# Nettoyage et découpage des professions
names["primaryProfession"] = names["primaryProfession"].fillna("").str.split(",")

# 1) Garder uniquement ceux dont la profession principale est actor/actress
names = names[
    names["primaryProfession"].apply(lambda profs: len(profs) > 0 and profs[0] in ["actor", "actress"])
]

# 2) Filtrer les rôles dans title.principals : garder uniquement actor/actress
principals = principals[principals["category"].isin(["actor", "actress"])]

# 3) Nettoyage des noms (éviter les prénoms seuls, pseudos, etc.)
names = names[names["primaryName"].fillna("").str.contains(" ")]

# -----------------------------
# 3) Jointure ACTEURS × TITRES
# -----------------------------
acteurs_titres = principals.merge(
    names[["nconst", "primaryName", "birthYear"]],
    on="nconst",
    how="inner"
).merge(
    titles,   # <-- contient maintenant genres
    on="tconst",
    how="inner"
).merge(
    ratings[["tconst", "averageRating", "numVotes"]],
    on="tconst",
    how="left"
)

# -----------------------------
# 4) Nettoyage des années
# -----------------------------
acteurs_titres["startYear"] = pd.to_numeric(acteurs_titres["startYear"], errors="coerce")
acteurs_titres["birthYear"] = pd.to_numeric(acteurs_titres["birthYear"], errors="coerce")

# -----------------------------
# 5) Décennie (robuste)
# -----------------------------
acteurs_titres["decade"] = (acteurs_titres["startYear"] // 10) * 10
acteurs_titres.loc[acteurs_titres["decade"] < 1880, "decade"] = np.nan

# -----------------------------
# 6) Catégorie simplifiée
# -----------------------------
mapping_categorie = {
    "movie": "Film",
    "tvMovie": "Téléfilm",
    "tvSeries": "Série"
}

acteurs_titres["categorie"] = acteurs_titres["titleType"].map(mapping_categorie)

# -----------------------------
# 7) Colonnes utiles
# -----------------------------
colonnes_utiles = [
    "nconst", "primaryName", "birthYear",
    "tconst", "primaryTitle", "titleType", "categorie",
    "genres",                      # <-- AJOUT ICI
    "startYear", "decade",
    "averageRating", "numVotes"
]

acteurs_titres = acteurs_titres[colonnes_utiles]

# -----------------------------
# 8) Ordonner les types de titres
# -----------------------------
ordre_types = ["movie", "tvMovie", "tvSeries"]

acteurs_titres["titleType"] = pd.Categorical(
    acteurs_titres["titleType"],
    categories=ordre_types,
    ordered=True
)

# -----------------------------
# 9) Tri final
# -----------------------------
acteurs_titres = acteurs_titres.sort_values(
    by=["primaryName", "titleType", "startYear"]
)

# -----------------------------
# 10) Analyse finale
# -----------------------------
since_year = 1990
birth_min = 1940
birth_max = 2026
top_n_global = 3000
top_n_per_actor = None

films = acteurs_titres.copy()
films = films[films["startYear"].ge(since_year)]

if birth_min is not None:
    films = films[films["birthYear"].ge(birth_min)]
if birth_max is not None:
    films = films[films["birthYear"].le(birth_max)]

films = films.sort_values(["startYear","averageRating","numVotes"], ascending=[False,False,False])
top_global = films.drop_duplicates("tconst").head(top_n_global).reset_index(drop=True)

top_par_acteur = (
    films.sort_values(["primaryName","startYear","averageRating","numVotes"], ascending=[True,False,False,False])
         .groupby("primaryName", as_index=False)
         .head(top_n_per_actor)
         .reset_index(drop=True)
)

cols = [
    "primaryName","birthYear","tconst","primaryTitle",
    "categorie","genres","startYear","decade","averageRating","numVotes"
]

display(top_global[cols])
display(top_par_acteur[cols])

acteurs_titres






# -----------------------------
# 11) Top acteurs prolifiques + décennie dominante
# -----------------------------

# Top 20 acteurs les plus prolifiques
top_prolifique = (
    acteurs_titres
    .groupby("primaryName")["tconst"]
    .nunique()
    .sort_values(ascending=False)
    .head(20)
    .reset_index(name="nb_titres")
)

# Décennie dominante pour chaque acteur
decade_actor = (
    acteurs_titres
    .groupby(["primaryName", "decade"])
    .size()
    .reset_index(name="nb")
)

# On garde la décennie où chaque acteur a tourné le plus
decade_actor = decade_actor.loc[
    decade_actor.groupby("primaryName")["nb"].idxmax()
]

# Fusion
top_prolifique = top_prolifique.merge(
    decade_actor[["primaryName", "decade"]],
    on="primaryName",
    how="left"
)

display(top_prolifique)


# -----------------------------
# 12) Graphique Plotly : Top prolifiques par décennie
# -----------------------------
import plotly.express as px

fig = px.bar(
    top_prolifique,
    x="nb_titres",
    y="primaryName",
    color="decade",
    orientation="h",
    title="Top 20 acteurs prolifiques par décennie dominante",
    color_continuous_scale="Viridis"
)

fig.update_layout(height=700)
fig.show()


# -----------------------------
# Présences Films / Séries
# -----------------------------

# Compter les présences par acteur et par type
presence_type = (
    films
    .groupby(["primaryName", "categorie"])
    .size()
    .reset_index(name="nb")
)

# Pour avoir un top des acteurs les plus présents (tous types confondus)
top_acteurs_presence = (
    presence_type
    .groupby("primaryName")["nb"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index
)

# Filtrer pour ne garder que les 20 acteurs les plus présents
presence_type_top = presence_type[presence_type["primaryName"].isin(top_acteurs_presence)]


# -----------------------------
# Ratio présence / popularité
# -----------------------------

# 1) Nombre de titres par acteur
presence = (
    films
    .groupby("primaryName")["tconst"]
    .nunique()
    .reset_index(name="nb_titres")
)

# 2) Popularité totale (somme des votes)
popularite = (
    films
    .groupby("primaryName")["numVotes"]
    .sum()
    .reset_index(name="total_votes")
)

# 3) Fusion
ratio_df = presence.merge(popularite, on="primaryName", how="left")

# 4) Calcul du ratio
ratio_df["ratio_presence_popularite"] = (
    ratio_df["nb_titres"] / ratio_df["total_votes"]
)

# 5) Nettoyage (éviter division par zéro)
ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_presence_popularite"])

# 6) Top 30 acteurs avec ratio le plus élevé
ratio_top = ratio_df.sort_values("ratio_presence_popularite", ascending=False).head(30)


import plotly.express as px

fig = px.bar(
    ratio_top,
    x="ratio_presence_popularite",
    y="primaryName",
    orientation="h",
    title="Ratio Présence / Popularité (Présence / Votes IMDb)",
    labels={
        "ratio_presence_popularite": "Ratio présence / popularité",
        "primaryName": "Acteur"
    },
    color="ratio_presence_popularite",
    color_continuous_scale="Viridis"
)

fig.update_layout(height=900)
fig.show()

# -----------------------------
# Âge moyen des acteurs selon popularité (Top 10 + filtre genre)
# -----------------------------

# 1) Calcul de l'âge au moment du film
films["age"] = films["startYear"] - films["birthYear"]

# Nettoyage : retirer âges impossibles
films = films[(films["age"] > 10) & (films["age"] < 100)]

# 2) Explosion des genres (car "Action,Drama" doit devenir 2 lignes)
films_genres = films.assign(
    genre=films["genres"].str.split(",")
).explode("genre")

# 3) Popularité par genre = somme des votes
popularite_genre = (
    films_genres
    .groupby("genre")["numVotes"]
    .sum()
    .reset_index(name="total_votes")
)

# 4) Top 10 genres les plus populaires
top10_genres = popularite_genre.sort_values("total_votes", ascending=False).head(10)["genre"]

# 5) Filtrer les films sur ces 10 genres
films_top_genres = films_genres[films_genres["genre"].isin(top10_genres)]

# 6) Calcul de l'âge moyen par genre
age_moyen_genre = (
    films_top_genres
    .groupby("genre")["age"]
    .mean()
    .reset_index(name="age_moyen")
)
# -----------------------------
# SUNBURST : Genre → Décennie → Âge moyen
# -----------------------------

# 1) Calcul de l'âge au moment du film
films["age"] = films["startYear"] - films["birthYear"]

# Nettoyage des âges impossibles
films = films[(films["age"] > 10) & (films["age"] < 100)]

# 2) Explosion des genres
films_genres = films.assign(
    genre = films["genres"].str.split(",")
).explode("genre")

# Retirer les genres vides
films_genres = films_genres[films_genres["genre"].notna()]

# 3) Groupby Genre × Décennie → Âge moyen
age_genre_decade = (
    films_genres
    .groupby(["genre", "decade"])["age"]
    .mean()
    .reset_index(name="age_moyen")
)
ZK
# 4) Sunburst Plotly
import plotly.express as px

fig = px.sunburst(
    age_genre_decade,
    path=["genre", "decade"],     # hiérarchie
    values="age_moyen",           # taille = âge moyen
    color="age_moyen",            # couleur = âge moyen
    color_continuous_scale="Viridis",
    title="Âge moyen des acteurs — Sunburst Genre → Décennie",
)

fig.update_layout

#✅ 1) Top 10 par décennie — MOVIE

top10_movie = (
    films[films["titleType"] == "movie"]
    .groupby(["decade", "primaryName"])["tconst"]
    .nunique()
    .reset_index(name="nb_titres")
    .sort_values(["decade", "nb_titres"], ascending=[True, False])
    .groupby("decade")
    .head(10)
    .reset_index(drop=True)
)

top10_movie

#✅ 2) Top 10 par décennie — TV MOVIE

top10_tvmovie = (
    films[films["titleType"] == "tvMovie"]
    .groupby(["decade", "primaryName"])["tconst"]
    .nunique()
    .reset_index(name="nb_titres")
    .sort_values(["decade", "nb_titres"], ascending=[True, False])
    .groupby("decade")
    .head(10)
    .reset_index(drop=True)
)

top10_tvmovie

