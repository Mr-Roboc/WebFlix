import os
import math
import pickle
import string
from collections import defaultdict, Counter # Counter is used for counting frequencies of hashable objects.

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    load_golden_dataset,
    load_movies,
)


class InvertedIndex:

    BM25_K1 =1.5

    
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int,dict] = {}
        self.term_frequencies = {} #  keeping track of how many times each term appears in each document
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path= os.path.join(CACHE_DIR,"doc_lengths.pkl")
        



    def get_document(self,term):
      
        doc_ids = self.index.get(term,set())

        return sorted(list(doc_ids))


    def load(self):


        frequency = os.path.join(CACHE_DIR,"term_frequencies.pkl")

    
        if not os.path.exists(idx_path) or not os.path.exists(doc_path) or not os.path.exists(frequency):
            
            raise FileNotFoundError("The file cannot be found")
        
        
        with open("cache/index.pkl","rb") as f:
            self.index = pickle.load(f)
            

        with open("cache/docmap.pkl","rb") as f:
            self.docmap = pickle.load(f)

        
        with open("cache/term_frequencies.pkl","rb") as f:
             self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path,"rb") as f:
            self.doc_lengths = pickle.load(f)
        

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie['id']

            self.docmap[doc_id] = movie


            text_to_index = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id,text_to_index)

    def save(self):
        

        if not os.path.exists("cache"):
            os.makedirs("cache")

        
        with open("cache/index.pkl","wb") as f:
            pickle.dump(self.index,f)


        with open("cache/docmap.pkl","wb") as f:
            pickle.dump(self.docmap,f)
            

        with open("cache/term_frequencies.pkl","wb") as f:
            
            pickle.dump(self.term_frequencies,f)

        with open(self.doc_lengths_path,"wb") as f:
            pickle.dump(self.doc_lengths,f)
        

    def __add_document(self, doc_id,text_to_index):
        tokens = tokenize_text(text_to_index)

        for token in tokens:
            self.index[token].add(doc_id)


        self.doc_lengths[doc_id] = len(tokens) # Storing it in a defaultdict

        

        self.term_frequencies[doc_id] = Counter(tokens) # If Document 4651 has the clean tokens ["brave", "princess", "brave"], self.term_frequencies[4651] becomes: Counter({"brave": 2, "princess": 1})
      


    def idf(self, term:str)->float:
        # 1. Load the index and docmap into memory using your class method
        try:
            self.load()
        except FileNotFoundError:
            print("Error: The index files were not found. Please run 'build' first.")
            

        # 2. Extract the clean, single string token from the tokenized list
        tokens = self.tokenize_term(term)
        if not tokens:
            print("Error: Provided term produced no valid tokens.")
            return
        t = tokens

        # 3. Calculate total documents (N) from the docmap
        doc_count = len(self.docmap)

        # 4. Safely get Document Frequency (DF). Fallback to empty set if token is missing.
        matching_docs = self.index.get(t, set())
        df_count = len(matching_docs)

        # 5. Apply the corrected smoothed IDF formula
        IDF = math.log((doc_count + 1) / (df_count + 1))

        # 6. Print the final calculated value
        return IDF



    def tfidf(self,term: str,doc_id: int):
        
        try:
            self.load()

        except FileNotFoundError:
               print("index files not found, run build first.. ")


        # Formula: TF-IDF = TF * IDF

        TF = self.tf(doc_id,term)

        IDF  = self.idf(term)

        tf_idf = TF*IDF
        
        return tf_idf
             

        
    def bm25idf(self,term)->float:
        single_term = self.tokenize_term(term)

        if not single_term:
            return 0.0

        N = len(self.docmap)
    
    
        matching_docs = self.index.get(single_term, set())
        df = len(matching_docs)


        
        result= math.log((N-df + 0.5)/(df+0.5)+1)
        return result
    


    def get_bm25_tf(self,doc_id,term,k1= BM25_K1,b = BM25_B):       

        """
        Calculates the length-normalized BM25 Term Frequency component.
        Assumes load() has already been called before running queries.
        """

        
        doc_id = int(doc_id)  # Guard against string IDs from CLI
        tf = self.tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1.0

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

        
        
        
        
            

    def tokenize_term(self, TERM):
        
        try:
            term_tokenize = tokenize_text(TERM)

            if not (len(term_tokenize)) == 1:
                raise ValueError("No single token recieved ")

            return term_tokenize[0] # returning the token as a string

        except Exception as e:
            print(f"Error: {e}")

    def tf(self, doc_id, term) -> int:

       
        doc_id = int(doc_id)

        doc_counter = self.term_frequencies.get(doc_id, {})# State of doc_counter: Counter({"brave": 2, "princess": 2, "merida": 1})

        return doc_counter.get(term, 0)  # Get the count of the term. If it's not in the Counter, it safely returns 0



    def __get_avg_doc_length(self)->float:

        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def bm25(self,doc_id,term):

        bm25_tf_score =self.get_bm25_tf(doc_id,term)
        bm25_idf_score = self.bm25idf(term)

        return bm25_tf_score * bm25_idf_score


    def bm25_search(self,query, limit=5):

        # Load from disk once before looping.
        if not self.index or not self.docmap:
            self.load()

        query_tokenize = tokenize_text(query)
        print(query_tokenize)

        scores = {}

    
    

        for doc_id in self.docmap:
             total_score = 0.0

             for query_token in query_tokenize:
                   total_score += self.bm25(doc_id,query_token)
                   
             scores[doc_id] = total_score


        sorted_scores = sorted(scores.items(),key=lambda x:x[1], reverse = True)

        
        results = []
        for doc_id, score in sorted_scores[:limit]:
            movie = self.docmap[doc_id]
            results.append((movie, score))  # Return tuple of (movie_dict, score)

        return results


        
            
                     

            
        

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1,b = BM25_B):

    doc_id = int(doc_id)
    idx = InvertedIndex()
    idx.load()

    
    print("DEBUG - Tokens in Doc 1:", idx.term_frequencies.get(doc_id, {}))
    tokenizeTerm = idx.tokenize_term(term)
    
    return idx.get_bm25_tf(doc_id, tokenizeTerm, k1,b)
        

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    
   #


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):

    idx = InvertedIndex()


    try:
        idx.load()

    except FileNotFoundError as e:
        print(f"Error{e}")
        return
       
    
    tokenize = tokenize_text(query)
    results = []

    seen_doc_ids = set()# Tracks unique movies so we don't return duplicates

    for token in tokenize:

        matched_ids = idx.get_document(token) # gets the ID assigned to the token in self.index.

        for doc_id in matched_ids:

            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)

            movie = idx.docmap[doc_id]
            results.append(movie)


            if len(results)>=limit:
                return results

            
    
    

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
