import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

# 1. Embed the question to a vector.
# 2. Search FAISS to find the nearest essay chunks.
# 3. Feed the chunks and question into the LLM.


# Load index and chunks
index = faiss.read_index("essayTrain/index/essays.index")
with open("essayTrain/index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Initialize embedder and LLM
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=GROQ_API_KEY)

def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True) # Embed the search query to a vector.
    D, I = index.search(q_emb, k) # I = Top 3 vectorized results that match the search query
    return [chunks[i] for i in I[0]] # Return the string equivelents for the 3 nearest vectors

# Create and ask the full prompt for the LLM
def ask(query):
    retrieved = retrieve(query) # Get the most relevant training data to the search query
    context = "\n\n".join(retrieved)
    
    prompt = f"""You are an assistant that writes exactly in my style.
            I am a college sophomore who is knowledgeable and interested in AI.
            Your job is to answer prompts so that the response reads like my essays.

            Here are my sample writings:
            {context}

            STYLE REQUIREMENTS:
            - Match the tone, word choice, sentence rhythm, and structure of the samples.
            - Write as if I'm speaking: direct, clear, sometimes imperfect. Use "I" and personal perspectives.
            - Show human emotion, but keep it natural. Allow for a few natural filler phrases.
            - Include at least one small grammatical mistake, as well as one or two sentences being a little redundant or not perfectly structured
            - Slightly informal but still professional, like a high school or college essay.
            - Do NOT make every sentence polished.


            PROHIBITED LANGUAGE:
            - Never use corporate / marketing jargon.
            - Especially avoid words commonly associated with AI, such as: "dive into," "unleash," 
            "game-changing," "revolutionary," "transformative," "leverage," "optimize," "unlock potential."
            - Don't use em-dashes

            OUTPUT RULES:
            1. Must sound like something I would actually write for school.
            2. Use normal vocabulary.
            4. Gets to the point.
            5. Include a reflection or conclusion in my writing style.
            6. Above all: it MUST sound like the sample writing.

            Now, respond to the following prompt in this style:
            {query}
            """
#            3. Feels genuine and honest, not artificial.

    # Get the repsonse from the LLM
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        top_p=0.9
    )

    # Parse the response from the generated .json file
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n--- Response ---")
        print(ask(q))
