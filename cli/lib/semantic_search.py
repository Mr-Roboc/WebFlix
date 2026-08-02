from sentence_transformers import SentenceTransformer



class SemanticSearch:

    def __init__(self):

        # Load the model.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Model Loaded : {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self,text:str)->list[float]:

        if not text or not text.strip():
            raise ValueError("The text can't be empty")


        # Generating an embedding for the input text and returns a list of 384 floating point numbers.

        return self.model.encode([text])[0] # We only care about the first element from the output list..



def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
        
def verify_model():
    s = SemanticSearch()
    
    

