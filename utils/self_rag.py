from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np  # ✅ Needed for FAISS compatibility

def load_embeddings_from_documents(doc_folder="documents"):
    """
    Loads text documents from a folder, generates embeddings using a SentenceTransformer model,
    and builds a FAISS index for fast similarity search.

    Args:
        doc_folder (str): Path to the folder containing text files.

    Returns:
        model (SentenceTransformer): The loaded embedding model.
        index (faiss.IndexFlatL2): FAISS index built from the document embeddings.
        doc_texts (list[str]): Original text contents of all documents.
    """
    # Load a pre-trained embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []   # To hold embedding vectors
    doc_texts = []   # To hold original text content

    # Read and embed each document in the folder
    for filename in os.listdir(doc_folder):
        path = os.path.join(doc_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            doc_texts.append(text)
            embedding = model.encode(text)
            documents.append(embedding)

    # Convert list of embeddings to a 2D NumPy array
    embeddings_array = np.vstack(documents).astype("float32")

    # Create FAISS index with appropriate vector dimensions
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)  # ✅ Correctly formatted array

    return model, index, doc_texts


def retrieve_context(query, model, index, texts, k=3):
    """
    Retrieves the top-k most relevant documents for a given query using FAISS similarity search.

    Args:
        query (str): The input query or sentence to find relevant context for.
        model (SentenceTransformer): Embedding model used for encoding the query.
        index (faiss.IndexFlatL2): FAISS index containing document embeddings.
        texts (list[str]): Original texts corresponding to the indexed embeddings.
        k (int): Number of top documents to retrieve.

    Returns:
        context (str): A string containing the top-k retrieved documents concatenated.
    """
    embedding = model.encode([query])   # Encode the query as a single vector (2D)
    D, I = index.search(embedding, k)   # Retrieve top-k similar documents
    context = "\n".join([texts[i] for i in I[0]])  # Concatenate retrieved texts
    return context
