import os
import faiss
from sentence_transformers import SentenceTransformer # Package for converting writing to create word embeddings to train LLM's on.
import pickle

EMBED_MODEL = "all-MiniLM-L6-v2" # Sentence Transformer model
essays_dir = "essayTrain/essays"
index_dir = "index"

# Take all writing samples in the essays_dir folder and put them into a list.
def load_essays(path=essays_dir):
    docs = []
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

# Split input document into chuncks of 500 words.
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def build_index():
    model = SentenceTransformer(EMBED_MODEL) # Load embedding model
    docs = load_essays() # Get training essays
    
    # Place all text chunks into one list
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d))
    
    #print(f"Total chunks: {len(chunks)}")

    # Turn each chunk into a vector 
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # Get the embedding size
    dim = embeddings.shape[1]
    # Insert all embeddings into the data set. 
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    
    os.makedirs(index_dir, exist_ok=True) # Make the output directory.
    faiss.write_index(index, os.path.join(index_dir, "essays.index")) # Save embeddings to essays.index
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f: # Save the original text chunks to chunks.pkl
        pickle.dump(chunks, f)

    print("Index built and saved.")

if __name__ == "__main__":
    build_index()
