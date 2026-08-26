from types import NoneType

from collections import defaultdict
import json
from numpy.linalg import norm

import re

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

from .search_utils import load_movies,format_search_result

MODEL_NAME = "all-MiniLM-L6-v2"
class SemanticSearch:

    def __init__(self,model_name:str):

        # Load the model.
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path =Path("cache/movie_embeddings.npy")

        print(f"Model Loaded : {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self,text:str)->list[float]:

        if not text or not text.strip():
            raise ValueError("The text can't be empty")


        # Generating an embedding for the input text and returns a list of 384 fTheloating point numbers.

        return self.model.encode([text])[0]

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





class ChunkedSemanticSearch(SemanticSearch):
    chunked_embeddings_path = Path("cache/chunk_embeddings.npy")
    chunked_embeddings_JSON = Path("cache/chunk_metadata.json")
    

    def __init__(self, model_name: str = MODEL_NAME)->None:

        super().__init__(model_name)

        self.chunk_embeddings = None
        self.chunk_metadata = None
        



    def build_chunk_embeddings(self,documents)->np.ndarray:

        self.documents= documents
        self.docmap= {doc['id']: doc for doc in documents}
        
        all_chunks:list[str] = []

        chunk_metadata:list[dict] = [] # hold the metadata of each chunk

        for doc in documents:
             if doc.get('description') is None:
                 continue
                 
             
             # Get chunks for this specific movie
             doc_chunks = semantic_chunk(doc['description'],overlap = 1, max_chunk_size = 4)

             
             for chunk_text in doc_chunks:
                 all_chunks.append(f"{doc['title']}: {chunk_text}") 


                 chunk_metadata.append({"movie_idx" : doc['id'],"chunk_idx": len(all_chunks)-1,"total_chunks": len(all_chunks)})


        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar =True) # stores the embeddings of all the chunks
     
        np.save(ChunkedSemanticSearch.chunked_embeddings_path,self.chunk_embeddings)


        with open(ChunkedSemanticSearch.chunked_embeddings_JSON,"w") as f:
             json.dump({"chunks": chunk_metadata, "total_chunks":len(all_chunks)}, f,indent = 2)

        

        self.chunk_metadata= chunk_metadata

        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self,documents:list)->np.ndarray:

        self.documents = documents
        self.document_map = {doc['id']:doc for doc in documents}

      
        # Check if both cache files exist
        if (ChunkedSemanticSearch.chunked_embeddings_path.exists() and 
            ChunkedSemanticSearch.chunked_embeddings_JSON.exists()):
            
            
            self.chunk_embeddings = np.load(ChunkedSemanticSearch.chunked_embeddings_path)
            
            with open(ChunkedSemanticSearch.chunked_embeddings_JSON, "r") as f:
                 data = json.load(f)
                 self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings


        # If cache files don't exist, create them
        return self.build_chunk_embeddings(documents)

    
    def search_chunks(self,query,limit:int = 10)->list[dict]:
        query_embedding = self.generate_embedding(query)

        chunk_score = []

        
        # Calculate cosine similarity for each query vector and append the metadata to chunk_score
        for idx,chunk in enumerate(self.chunk_embeddings):
            query_score = cosine_similarity(query_embedding,chunk)


            metadata = self.chunk_metadata[idx]
            chunk_score.append({"movie_idx":metadata['movie_idx'], "chunk_idx":metadata['chunk_idx'],"score":query_score})

        

        # Max-Pooling
        movie_scores = defaultdict(int)

        
        for score in chunk_score:
            current_score = score['score']
            if score['movie_idx'] not in movie_scores or current_score>movie_scores[score['movie_idx']]:
                movie_scores[score['movie_idx']] = current_score
        

        # Sort by score (index 1) descending, and slice up to `limit`
        sorted_movie_scores = sorted(
            movie_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:limit]


        formatted_result = []
        for movie_idx,score in sorted_movie_scores:
            doc= self.document_map[movie_idx]
            
            formatted_result.append(format_search_result(doc['id'],doc['title'],doc['description'],score))

        return formatted_result
            
            
def search(query,limit):

    s = SemanticSearch(MODEL_NAME)
    movies = load_movies()
    
    s.load_embeddings(movies)

    search_result = s.search(query,limit)

    for idx,res in enumerate(search_result):

        print(idx,res['title'], ":", "Score : ", res['score'])
       
        
        
def verify_model():
    s = SemanticSearch(MODEL_NAME)


def verify_embeddings():
    es = SemanticSearch(MODEL_NAME)

    documents = load_movies()

    embeddings = es.load_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query(query:str):
    s = SemanticSearch(MODEL_NAME)

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


def fixed_size(text:str,overlap,chunk_size:int)->list:

    words = text.split(" ")

    chunks = []

    step_size = chunk_size-overlap
    for i in range(0,len(words),step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words)<=overlap:
            break
            
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks


def chunk_text(text,overlap,chunk_size):
    chunks = fixed_size(text,overlap,chunk_size)



    print(f"Chunking {len(text)} characters")

    for idx, chunk in enumerate(chunks,1):
        print(idx,chunk)

def semantic_chunk(text:str, overlap:int, max_chunk_size):
    sentences = re.split(r"(?<=[.!?])\s+",text)

    chunked_strings = []


    step_size = max(1, max_chunk_size - overlap)

    for i in range(0,len(sentences),step_size):

        chunk_sentences = sentences[i:i+max_chunk_size]
        
        if not chunk_sentences:
            break


        chunked_strings.append("".join(chunk_sentences))
        

    return chunked_strings
            

    



def semantic_chunk_text(text,overlap=0,max_chunk_size=4):
    chunks = semantic_chunk(text,overlap,max_chunk_size)



    print(f"Chunking {len(text)} characters")

    for idx, chunk in enumerate(chunks,1):
        print(idx,chunk)



def embedded_chunk():
    c= ChunkedSemanticSearch(MODEL_NAME)
    movies = load_movies()
    embeddings = c.load_or_create_chunk_embeddings(movies)

    
    print(f"Generated {len(embeddings)} chunked embeddings")
    



def chunk_search(query,limit):
    c= ChunkedSemanticSearch(MODEL_NAME)
    movies = load_movies()
    embeddings = c.load_or_create_chunk_embeddings(movies)

    sorted_scores = c.search_chunks(query,limit)
    for index,doc in enumerate(sorted_scores,1):
        print(f"{index}. {doc['title']} : {doc['score']:.4f}")