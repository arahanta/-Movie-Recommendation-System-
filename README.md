# Movie Recommendation System

## Introduction
Welcome to the Movie Recommendation System! This project utilizes content-based filtering to provide personalized movie recommendations. By analyzing various features of movies, including genres, keywords, cast, and crew, the system recommends movies similar to the one selected by the user. It uses NLP techniques and the cosine similarity metric to recommend movies based on user selection.

##  Features

   - Movie Selection: Choose a movie from the dropdown menu.
   - Recommendations: Get top 5 movie recommendations along with their posters.

## Installation and Setup
To begin, install the required packages:
```bash
pip install streamlit pandas requests scikit-learn nltk

```
Additionally, download the required datasets

## Dataset

The datasets used in this project are:

    tmdb_5000_movies.csv: Contains movie details.
    tmdb_5000_credits.csv: Contains movie credits information.
## Streamlit App
Interactive User Interface:
        Display a title and a dropdown menu for movie selection.
        Provide a button to trigger recommendations.
        Display recommended movie titles and posters in a grid layout.

![image](https://github.com/user-attachments/assets/5f65801d-98b5-4a3b-8106-8273a0e9abb0)
## Usage

To run the Movie Recommendation System, execute the script with Streamlit:
```bash
streamlit run your_script.py
```
