from PIL import Image
from sentence_transformers import SentenceTransformer

import json

from lib.search_utils import load_movies
from lib.semantic_search import cosine_similarity

def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch()

    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding Shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    movies= load_movies()
    model = MultimodalSearch(movies)
    return model.search_with_image(image_path)


class MultimodalSearch:

    def __init__(self,docs,model_name = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.docs = docs

        self.texts = []

        for idx,_ in enumerate(self.docs):
            self.texts.append(f"{self.docs[idx]['title']}: {self.docs[idx]['description']}")

        self.text_embeddings = self.model.encode(self.texts,show_progress_bar=True)


        



    def embed_image(self,image_path):
        image = Image.open(image_path)

        return self.model.encode([image])[0]


    def search_with_image(self,image_path):
        embedded_image = self.embed_image(image_path)

        score_metadata = []
        for idx,text_embed in enumerate(self.text_embeddings):
            cos_score= cosine_similarity(embedded_image,text_embed)
            meta_data = {"doc_id": idx+1, "title": self.docs[idx]["title"], "description":self.docs[idx]["description"], "score": cos_score}


            score_metadata.append(meta_data)

        return sorted(score_metadata, key=lambda x: x["score"],reverse=True)[:5]

    
    
    