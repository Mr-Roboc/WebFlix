from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

from .search_utils import load_movies
class SemanticSearch:

    def __init__(self):

        # Load the model.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path =Path("cache/movie_embeddings.npy")

        print(f"Model Loaded : {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self,text:str)->list[float]:

        if not text or not text.strip():
            raise ValueError("The text can't be empty")


        # Generating an embedding for the input text and returns a list of 384 floating point numbers.

        return self.model.encode([text])[0] # We only care about the first element from the output list..

    def build_embeddings(self,documents:list[dict]):
        self.document_map = {}
        self.documents = documents
        movie_list = []
        
        for doc in self.documents:
            self.document_map[doc['id']] = doc 
            doc_title = doc['title']
            doc_desc  = doc['description']
            movie_list.append(f"{doc_title}: {doc_desc}")


        # .encode() on movie_list

        self.embeddings = self.model.encode(movie_list,show_progress_bar = True)
        
        np.save(self.embeddings_path,self.embeddings)
        
        return self.embeddings
    
    def load_embeddings(self,documents):
        self.documents= documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc['id']] =doc

        # Check if the if .npy file exists
        if self.embeddings_path.exists():
           self.embeddings =  np.load(self.embeddings_path)

           if len(self.embeddings)==len(documents):
               return self.embeddings

           
            
        return self.build_embeddings(documents)
            

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
        
def verify_model():
    s = SemanticSearch()


def verify_embeddings():
    es = SemanticSearch()

    documents = load_movies()

    embeddings = es.load_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
