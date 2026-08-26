import os
from collections import defaultdict

from lib.search_utils import load_movies
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch



def weighted_search(query, alpha=0.5, limit=5):
    if isinstance(query, list):
        query = " ".join(query)

    movies = load_movies()
    hs = HybridSearch(movies)
    
    limit_int = int(limit)
    results = hs.weighted_search(query, alpha, limit_int)

    for idx, r in enumerate(results[:limit_int]):
        
        print(f"{idx + 1}. {r['title']}")
        
        
        print(f"  Hybrid Score: {r['hybrid_score']:.3f}")
        print(f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['sem_score']:.3f}")
        
       
        desc_snippet = r['description'][:100] + "..." if len(r['description']) > 100 else r['description']
        print(f"  {desc_snippet}")


def rrf_search(query,k:int,limit:int):
    movies = load_movies()
    h = HybridSearch(movies)

    rrf_result = h.rrf_search(query,k = 60,limit = 5)

    sorted_result = sorted(rrf_result.values(),key=lambda x: x['rrf_score'],reverse=True)

    for idx,result in enumerate(sorted_result[:limit]):
        print(f"{idx}. {result['doc_title']}")
        print(f"RRF Score:{result['rrf_score']:.3f}")
        print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n ")






class HybridSearch:
    def __init__(self,documents:list[dict])->None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents) 
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


    def rrf_search(self,query:str,k:int,limit:int=10) ->list[dict]:

        
        bm25_search_result = self._bm25_search(query,limit*500)

        semantic_chunk_result= self.semantic_search.search_chunks(query,limit*500)


        return combine_rrf_search(bm25_search_result,semantic_chunk_result,k)





def combine_rrf_search(bm25_search_result,semantic_chunk_result,k)->dict:


    combined = {}
    for rank, (doc,_) in enumerate(bm25_search_result,start = 1):
        doc_id = doc['id']
        doc_title = doc['title']
        
        rrf_comp= rrf_score(rank,k)

        combined[doc_id]= {
            'doc_title':doc_title,
            'bm25_rank':rank,
            'semantic_rank':0,
            'rrf_score': rrf_comp
        }

    for rank,doc in enumerate(semantic_chunk_result,start=1):
        doc_id= doc['id']
        doc_title = doc['title']
        
        semantic_rrf = rrf_score(rank,k)

        if doc_id in combined:
            combined[doc_id]['rrf_score'] +=semantic_rrf
            combined[doc_id]['semantic_rank'] = rank

        else:
            combined[doc_id]={
                'doc_title':doc_title,
                 'bm25_rank':0,
                 'semantic_rank':rank,
                 'rrf_score':semantic_rrf
            }


    return combined


    



def rrf_score(rank:int, k:int = 60)->float:
    return 1/(k+rank)




def normalize_search_results(results):
    if not results:
        return []

    
    scores = [r['score'] if isinstance(r, dict) else r[1] for r in results]

    
    norm_scores = normalize_score(scores)


    for idx, result in enumerate(results):
        if isinstance(result, dict):
            result['normalized_score'] = norm_scores[idx]
        else:
         
            results[idx] = {
                'id': result[0]['id'] if isinstance(result[0], dict) else result[0],
                'title': result[0]['title'] if isinstance(result[0], dict) else '',
                'description': result[0]['description'] if isinstance(result[0], dict) else '',
                'score': result[1],
                'normalized_score': norm_scores[idx]
            }

    # Return the updated list
    return results
    


def combine_search_results(bm25_results,semantic_results):
    bm25_norm  = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(semantic_results)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm['id']
        combined_norm[doc_id] = {
            "doc_id": doc_id,
            "bm25_score": norm['normalized_score'],
            'sem_score':0,
            'title': norm['title'],
            'description':norm['description']

        }

    
    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                'doc_id':doc_id,
                'bm25_score':0.0,
                'sem_score':0.0,
                'title': norm.get('title',''),
                'description':norm.get('description','')

            }

        combined_norm[doc_id]['sem_score'] = norm['normalized_score']

    for k,v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'])


    results= sorted(combined_norm.values(),key=lambda x: x['hybrid_score'],reverse=True)
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



