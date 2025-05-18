from sentence_transformers import SentenceTransformer
import faiss
import os

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
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for embedding sentences
    documents = []    # List to hold embeddings
    doc_texts = []    # List to hold raw text content

    # Loop over all files in the folder
    for filename in os.listdir(doc_folder):
        path = os.path.join(doc_folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            doc_texts.append(text)              # Save raw text
            documents.append(model.encode(text)) # Generate and store embeddings

    # Initialize FAISS index with correct vector dimensions
    dimension = documents[0].shape[0]
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric (Euclidean)
    index.add(documents)                  # Add embeddings to the index

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
    embedding = model.encode([query])   # Encode the query into a vector
    D, I = index.search(embedding, k)   # Perform similarity search
    context = "\n".join([texts[i] for i in I[0]])  # Gather the retrieved documents
    return context
