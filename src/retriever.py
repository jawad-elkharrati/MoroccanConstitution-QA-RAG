"""retriever.py: Gère la génération des embeddings, la création du magasin vectoriel et la récupération des documents."""

import json
import os

# Modèle d'embedding recommandé
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Par défaut rapide et efficace

PROCESSED_DATA_PATH = r"C:\Users\dell\Downloads\constitution_qa_project\processed_constitution_chunks.json"
VECTOR_STORE_PATH = r"C:\Users\dell\Downloads\constitution_qa_project\vector_store\faiss_index"

# Création du dossier pour le magasin vectoriel si inexistant
if not os.path.exists(os.path.dirname(VECTOR_STORE_PATH)):
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH))

def load_documents_from_json(file_path: str) -> list:
    """Charge les documents (chunks) depuis un fichier JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from langchain_core.documents import Document
        documents = [Document(page_content=item["text"], metadata=item["metadata"]) for item in data]
        print(f"{len(documents)} documents chargés depuis {file_path}")
        return documents
    except FileNotFoundError:
        print(f"Fichier introuvable : {file_path}")
        return []
    except Exception as e:
        print(f"Erreur lors du chargement des documents : {e}")
        return []

def create_and_save_vector_store(documents: list, embedding_model_name: str, store_path: str):
    """Crée et enregistre un magasin vectoriel FAISS."""
    if not documents:
        print("Aucun document à indexer.")
        return None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print("Création du magasin vectoriel FAISS...")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(store_path)
        print(f"Magasin vectoriel enregistré à {store_path}")
        return vector_store
    except ImportError as ie:
        print(f"Erreur d'import : {ie}. Installation des dépendances...")
        os.system("pip3 install langchain sentence-transformers faiss-cpu tiktoken langchain-community")
        print("Relancez le script après l'installation.")
        return None
    except Exception as e:
        print(f"Erreur lors de la création du magasin vectoriel : {e}")
        return None

def load_vector_store(store_path: str, embedding_model_name: str):
    """Charge un magasin vectoriel FAISS existant."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        if not os.path.exists(store_path):
            print(f"Magasin vectoriel introuvable à {store_path}")
            return None

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"Chargement du magasin vectoriel depuis {store_path}...")
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        print("Magasin vectoriel chargé avec succès.")
        return vector_store
    except ImportError as ie:
        print(f"Erreur d'import : {ie}. Assurez-vous que toutes les dépendances sont installées.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement du magasin vectoriel : {e}")
        return None

def get_relevant_documents(query: str, vector_store, k: int = 5) -> list:
    """Récupère les k documents les plus pertinents pour une requête donnée."""
    if not vector_store:
        print("Magasin vectoriel indisponible.")
        return []
    try:
        print(f"Recherche des documents pertinents pour : '{query}'")
        results = vector_store.similarity_search(query, k=k)
        print(f"{len(results)} documents trouvés.")
        return results
    except Exception as e:
        print(f"Erreur lors de la recherche : {e}")
        return []

if __name__ == "__main__":
    print("Démarrage du script de récupération...")
    
    docs = load_documents_from_json(PROCESSED_DATA_PATH)
    if docs:
        force_recreate_store = False
        vs = None

        if not force_recreate_store and os.path.exists(VECTOR_STORE_PATH):
            print("Chargement du magasin vectoriel existant...")
            vs = load_vector_store(VECTOR_STORE_PATH, EMBEDDING_MODEL)

        if vs is None:
            print("Création d’un nouveau magasin vectoriel...")
            vs = create_and_save_vector_store(docs, EMBEDDING_MODEL, VECTOR_STORE_PATH)

        if vs:
            sample_query = "Quelles sont les dispositions générales de la monarchie ?"
            relevant_docs = get_relevant_documents(sample_query, vs, k=3)
            if relevant_docs:
                print(f"\nTop 3 documents pour la requête : '{sample_query}'")
                for i, doc in enumerate(relevant_docs):
                    print(f"\n--- Document {i+1} (Article : {doc.metadata.get('article_number', 'N/A')}) ---")
                    print(doc.page_content[:200] + "...")
            else:
                print("Aucun document pertinent trouvé.")
        else:
            print("Impossible de charger ou créer le magasin vectoriel.")
    else:
        print("Échec du chargement des documents.")

    print("Fin du script.")