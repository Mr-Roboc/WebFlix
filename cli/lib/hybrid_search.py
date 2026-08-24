import os
from collections import defaultdict
import numpy as np
from lib.search_utils import load_movies
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch



def weighted_search(query,alpha= 0.5, limit = 0.5):
    movies= load_movies()
    hs= HybridSearch(movies)
    results = hs.weighted_search(query,alpha,limit)
    for idx,r in enumerate(results[:limit]):
        print(f"{idx+1} {r['title']}")
        print(f"Hybrid Score: {r['hybrid_score']}")
        print(f"BM25: {r['bm25_score']}, Semantic: {r['sem_score']}")

class HybridSearch:
    def __init__(self,documents:list[dict])->None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents) # Creating the chunked embeddings for the document
        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.idx_path):
        
            self.idx.build()
            self.idx.save()


    def _bm25_search(self,query:str,limit:int)->list[tuple]:
        self.idx.load()

        return self.idx.bm25_search(query,limit)
        


    

    def weighted_search(self,query:str, alpha:float, limit:int)->list[dict]:

        bm25_results = self._bm25_search(query,limit*500) # list[tuple]
        semantic_results = self.semantic_search.search_chunks(query,limit*500) # list['score']

        combined_results = combine_search_results(bm25_results,semantic_results)

        return combined_results

        # bm25_scores = np.array([score for movie,score in bm25_results])
        # semantic_scores = np.array([item["score"] for item in semantic_results])

        # # Normalizing the scores
        # bm25_norm = normalize_score(bm25_scores)
        # semantic_norm = normalize_score(semantic_scores)

       


    def rrf_search(self,query:str,k:int,limit:int=10) ->list[dict]:
        raise NotImplementedError("RRF search is not implemented yet")





def normalize_search_results(results):
    scores = [normalize_scores(r['score'] for r in results)]

    norm_scores = normalize_scores(scores)
    for idx,result in enumerate(results):
        result['normalized_score'] = norm_scores[idx]

    
    return results

def combine_search_results(bm25_results,semantic_results):
    bm25_norm  = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(sem_results)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm['doc_id']
        combined_norm[doc_id] = {
            "doc_id": doc_id,
            "bm25_Score": norm['normalized_score'],
            'sem_score':0,
            'title': norm['title'],
            'description':norm['description']

        }

    
    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                'doc_id':doc_id,
                'bm25_score':0.,
                'sem_score':0.,
                'title': norm['title'],
                'description':norm[description]

            }

        combined_norm[doc_id]['sem_score'] = norm['normalized_score']

    for k,v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'])


    results= sorted(combined_norm.values(),key=lambda x: x[0],reverse=True)
    return results



def hybrid_score(bm25_score: float,semantic_score:float, alpha:float = 0.5)->float:
        return alpha * bm25_score + ((1 - alpha) * semantic_score)


def normalize_score(scores):
   
    if not scores: return []


    max_score = max(scores)
    min_score = min(scores)


    if max_score==min_score: return [1.]*len(scores)

    score_range = max_score-min_score

    return [(score-min_score)/score_range for score in scores]



