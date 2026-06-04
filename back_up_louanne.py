import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
movies = pd.read_csv("movies_final.csv")

#--------------------------------------------------------------------------------------------
# KPI 1 - EVOLUTION DUREE DES FILMS PAR DECENNIE
#--------------------------------------------------------------------------------------------
runtime_decennie = movies.groupby("decennie")["runtime_final"].mean().reset_index()
print(runtime_decennie.head())

#Graphique 1
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=runtime_decennie,
    x="decennie",
    y="runtime_final",
    marker="o"
)
plt.title("Évolution de la durée moyenne des films par décennie")
plt.xlabel("Décennie")
plt.ylabel("Durée moyenne en minutes")
plt.show()

#--------------------------------------------------------------------------------------------
# KPI 2 - TOP DES GENRES LES PLUS REPRESENTES
#--------------------------------------------------------------------------------------------

movies_genres = movies.copy()
movies_genres["genres_final"] = movies_genres["genres_final"].str.split(",")
movies_genres = movies_genres.explode("genres_final")
kpi_genres = movies_genres["genres_final"].value_counts(normalize=True) * 100
kpi_genres = kpi_genres.reset_index()
print(kpi_genres.head(10))

#Graphique 2
plt.figure(figsize=(12, 6))
sns.barplot(
    data=kpi_genres.head(10),
    x="proportion",
    y="genres_final"
)
plt.title("Top 10 des genres les plus représentés")
plt.xlabel("Pourcentage de films")
plt.ylabel("Genre")
plt.show()

#--------------------------------------------------------------------------------------------
# KPI 3 - REPARTITION DES FILMS PAR GENRE AU COURS DU TEMPS
#--------------------------------------------------------------------------------------------
genre_decennie = movies_genres.groupby(
    ["decennie", "genres_final"]
).size().reset_index(name="nb_films")
genre_decennie["total_decennie"] = genre_decennie.groupby("decennie")["nb_films"].transform("sum")
genre_decennie["pourcentage"] = (
    genre_decennie["nb_films"] / genre_decennie["total_decennie"]
) * 100
top_genres = movies_genres["genres_final"].value_counts().head(8).index

genre_decennie_top = genre_decennie[
    genre_decennie["genres_final"].isin(top_genres)
]

#Graphique 3
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=genre_decennie_top,
    x="decennie",
    y="pourcentage",
    hue="genres_final",
    marker="o"
)
plt.title("Évolution des genres par décennie")
plt.xlabel("Décennie")
plt.ylabel("Pourcentage de films")
plt.show()

#--------------------------------------------------------------------------------------------
# KPI 4 - RENTABILITE PAR BUGDET ET REVENU
#--------------------------------------------------------------------------------------------
profit = movies_genres.dropna(
    subset=["budget_tmdb", "revenue_tmdb", "genres_final"]
)
profit = profit[
    (profit["budget_tmdb"] > 0) &
    (profit["revenue_tmdb"] > 0)
]
profit["roi"] = (
    profit["revenue_tmdb"] - profit["budget_tmdb"]
) / profit["budget_tmdb"]
rentabilite_genre = profit.groupby("genres_final")["roi"].mean().reset_index()
rentabilite_genre = rentabilite_genre.sort_values(
    by="roi",
    ascending=False
)
print(rentabilite_genre.head(10))

#Graphique 4
plt.figure(figsize=(12, 6))
sns.barplot(
    data=rentabilite_genre.head(10),
    x="roi",
    y="genres_final"
)
plt.title("Genres les plus rentables selon le ROI moyen")
plt.xlabel("ROI moyen")
plt.ylabel("Genre")
plt.show()


#--------------------------------------------------------------------------------------------
# KPI 1 - EVOLUTION DUREE DES FILMS PAR DECENNIE
#--------------------------------------------------------------------------------------------

# Préparer top genres
movies_tmp = movies.copy()
movies_tmp["genres_final"] = movies_tmp["genres_final"].str.split(",")
movies_tmp = movies_tmp.explode("genres_final")
top_genres = movies_tmp["genres_final"].value_counts().head(6).index.tolist()

df = (movies_tmp[movies_tmp["genres_final"].isin(top_genres)]
      .groupby(["decennie", "genres_final"])["runtime_final"]
      .mean()
      .reset_index())
df["decennie_num"] = pd.to_numeric(df["decennie"], errors="coerce")

fig = px.line(
    df.sort_values("decennie_num"),
    x="decennie_num",
    y="runtime_final",
    color="genres_final",
    facet_col="genres_final",
    facet_col_wrap=3,
    markers=True,
    title="Durée moyenne par décennie pour les principaux genres",
    labels={"decennie_num":"Décennie", "runtime_final":"Durée (min)"}
)
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)
fig.update_layout(height=700, title_font_size=18, margin=dict(t=100, l=60, r=40, b=60), showlegend=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # nettoie les titres de facets
fig.show()
