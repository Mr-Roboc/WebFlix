from sentence_transformers import SentenceTransformer

# 1. Load the model
#model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define your text input

#text = "Vector embeddings capture semantic meaning."

#print(f"Model loaded: {model}")
#print(f"Max sequence length: {model.max_seq_length}")

#model.encode(text)

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Model Loaded : {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")



def verify_model():
    s = SemanticSearch()

    

