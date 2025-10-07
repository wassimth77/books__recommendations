import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import chardet
import warnings

# Filtrer les warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()


# Fonction pour détecter l'encodage du fichier
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


# Chargement des données
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


# CORRECTION : Créer la colonne mapped_categories_clean comme dans votre autre notebook
def create_mapped_categories(books_df):
    """Crée la colonne mapped_categories_clean comme dans votre autre notebook"""
    category_mapping = {
        'Fiction': 'Fiction',
        'Juvenile Fiction': "Children's Fiction",
        'Comics & Graphic Novels': 'Fiction',
        'Drama': 'Fiction',
        'Poetry': 'Fiction',
        'Biography & Autobiography': 'Nonfiction',
        'History': 'Nonfiction',
        'Literary Criticism': 'Nonfiction',
        'Philosophy': 'Nonfiction',
        'Religion': 'Nonfiction',
        'Juvenile Nonfiction': "Children's Nonfiction",
        'Science': 'Nonfiction'
    }

    # Appliquer le mapping
    mapped_categories = books_df["categories"].map(category_mapping)

    # Garder seulement les catégories valides (non-NaN)
    mapped_categories_clean = mapped_categories.dropna()

    return mapped_categories_clean


# Vérifier et créer la colonne mapped_categories_clean si elle n'existe pas
if "mapped_categories_clean" not in books.columns:
    print("⚠️ Colonne 'mapped_categories_clean' non trouvée, création...")

    try:
        # Créer la colonne mapped_categories_clean
        mapped_categories_clean = create_mapped_categories(books)

        # Ajouter la colonne au DataFrame principal
        # Pour garder l'alignement, nous créons une série avec les mêmes index
        books["mapped_categories_clean"] = books["categories"].map({
            'Fiction': 'Fiction',
            'Juvenile Fiction': "Children's Fiction",
            'Comics & Graphic Novels': 'Fiction',
            'Drama': 'Fiction',
            'Poetry': 'Fiction',
            'Biography & Autobiography': 'Nonfiction',
            'History': 'Nonfiction',
            'Literary Criticism': 'Nonfiction',
            'Philosophy': 'Nonfiction',
            'Religion': 'Nonfiction',
            'Juvenile Nonfiction': "Children's Nonfiction",
            'Science': 'Nonfiction'
        })

        # Remplir les valeurs manquantes par "Unknown"
        books["mapped_categories_clean"] = books["mapped_categories_clean"].fillna("Unknown")

        print("✅ Colonne 'mapped_categories_clean' créée avec succès")

    except Exception as e:
        print(f"❌ Erreur lors de la création de mapped_categories_clean: {e}")
        # Fallback : créer une colonne par défaut
        books["mapped_categories_clean"] = "Unknown"

# Détection automatique de l'encodage
try:
    encoding = detect_encoding("tagged_description.txt")
    print(f"Encodage détecté: {encoding}")
    raw_documents = TextLoader("tagged_description.txt", encoding=encoding).load()
except Exception as e:
    print(f"Erreur avec l'encodage détecté: {e}")
    # Fallback vers UTF-8 avec gestion d'erreurs
    raw_documents = TextLoader("tagged_description.txt", encoding='utf-8', errors='ignore').load()

# Configuration optimisée du TextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

documents = text_splitter.split_documents(raw_documents)
print(f"✅ Nombre de documents créés: {len(documents)}")

# Utilisation de Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': False,
        'batch_size': 32
    }
)

db_books = Chroma.from_documents(documents, embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # CORRECTION : Utiliser mapped_categories_clean au lieu de simple_categories
    if category != "All":
        book_recs = book_recs[book_recs["mapped_categories_clean"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# CORRECTION : Utiliser mapped_categories_clean pour les catégories
try:
    categories = ["All"] + sorted(books["mapped_categories_clean"].unique())
    print(f"✅ Catégories disponibles: {categories}")
except KeyError as e:
    print(f"❌ Erreur avec mapped_categories_clean: {e}")
    # Fallback : utiliser les colonnes disponibles
    if "categories" in books.columns:
        categories = ["All"] + sorted(books["categories"].unique())
    else:
        categories = ["All"]

tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()