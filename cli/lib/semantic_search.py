from numpy.linalg import norm

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

        return self.model.encode([text])

    def build_embeddings(self,documents):
        self.document_map = {}
        self.documents = documents
        movie_list = []
        
        for doc in self.documents:
            self.document_map[doc['id']] = doc 
            doc_title = doc['title']
            doc_desc  = doc['description']
            movie_list.append(f"{doc_title}: {doc_desc}")


        # .encode() on movie_list
        self.embeddings = self.model.encode(movie_list,show_progress_bar = True) # self.embeddings stores an ndarray of document embeddings
        
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


    # search function

    def search(self,query,limit):

        if self.embeddings is None:
            raise ValueError("No embeddings loaded, Call load_embeddings method first")


        # Generate a single query vector.
        query_embedding  = self.generate_embedding(query)

        results = []


        # Comparing the query vector with each document embedding
        for doc_embed, doc in zip(self.embeddings,self.documents):
            score = cosine_similarity(query_embedding,doc_embed)
            results.append((score,doc))




        # Sort descending by similarity score
        results.sort(key=lambda x: x[0], reverse=True)

        score_list = []
        for sc,doc in results[:limit]:
            
            score_list.append({'score': sc, 'title': doc['title']})


        return score_list

        
                

def search(query,limit):

    s = SemanticSearch()
    movies = load_movies()
    
    s.load_embeddings(movies)

    search_result = s.search(query,limit)

    for idx,res in enumerate(search_result):

        print(idx,res['title'], ":", "Score : ", res['score'])
       
        
        
def verify_model():
    s = SemanticSearch()


def verify_embeddings():
    es = SemanticSearch()

    documents = load_movies()

    embeddings = es.load_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query(query:str):
    s = SemanticSearch()

    embedding = s.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape[0]}") # returns tuple : (no of times,vector size)


def cosine_similarity(vec1:np.ndarray,vec2:np.ndarray): # The two input vectors are basically the query and movie embeddings.
    dot_product = np.dot(vec1,vec2)
    norm_1 = np.linalg.norm(vec1) # calculates the math norm of the vector
    norm_2 = np.linalg.norm(vec2)
    if norm_1==0 or norm_2 ==0:
        return 0.0


    return dot_product/(norm_1*norm_2)







               


    
