import os

# The API key for the AI hosting website
# If you want to use this code yourself, you will need to get your own at the groq website. 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "API Key goes here")

# LLM model
GROQ_MODEL = "llama-3.3-70b-versatile"
