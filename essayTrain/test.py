from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)
models = client.models.list()

for m in models.data:
    print(m.id)
