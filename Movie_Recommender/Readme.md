
# 🎬 Movie Recommendation System

A simple and effective **Content-Based Movie Recommendation System** that suggests movies similar to a selected movie using metadata such as genres, keywords, cast, and crew.

---

## 🚀 Overview
This system analyzes movie descriptions and metadata, converts them into numerical vectors, and uses **Cosine Similarity** to find movies closest in meaning/content.  
It is built inside the Jupyter Notebook: `Movie_Recommender_System.ipynb`.

---

## 📂 Dataset
TMDB Movie Dataset (movies + credits)
- Title  
- Overview  
- Genres  
- Keywords  
- Cast & Crew  

---

## 🛠 Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-Learn (CountVectorizer, Cosine Similarity)  
- NLTK (Stemming)  
- Ast  

---

## 🧹 Preprocessing Steps
- Remove null values  
- Convert JSON-like strings into Python lists  
- Create a unified **tags** column  
- Apply stemming  
- Vectorize tags (5000 features)  
- Build cosine similarity matrix  
