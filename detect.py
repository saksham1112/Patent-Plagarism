import os
import pickle
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class PlagiarismDetector:
    def __init__(self):
        self.EMBEDDINGS_PATH = os.path.join("embeddings", "trained_embeddings.pkl")
        self.THRESHOLD = 0.9
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.db = None
        self.load_resources()
    
    def load_resources(self):
        try:
            if not os.path.exists(self.EMBEDDINGS_PATH):
                raise FileNotFoundError(f"Embeddings not found at: {os.path.abspath(self.EMBEDDINGS_PATH)}")
            
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to(self.DEVICE)
            
            with open(self.EMBEDDINGS_PATH, 'rb') as f:
                self.db = pickle.load(f)
                
        except Exception as e:
            raise Exception(f"Failed to load resources: {str(e)}")

    def text_to_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, padding=True, max_length=512).to(self.DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    
    def analyze_text(self, text):
        if not self.db:
            raise ValueError("Reference database not loaded")
        
        input_embedding = self.text_to_embedding(text)
        
        results = []
        for ref_name, ref_embed in self.db.items():
            similarity = cosine_similarity(
                input_embedding, 
                ref_embed.numpy().reshape(1, -1)
            )[0][0]
            results.append({
                'document': os.path.basename(ref_name),
                'similarity': float(similarity),
                'is_match': bool(similarity > self.THRESHOLD)
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'top_matches': results[:3],
            'potential_plagiarism': any(r['is_match'] for r in results),
            'average_similarity': float(np.mean([r['similarity'] for r in results]))
        }

plagiarism_detector = PlagiarismDetector()

def predict(text):
    try:
        results = plagiarism_detector.analyze_text(text)
        return {
            'top_matches': [
                {
                    'document': str(match['document']),
                    'similarity': float(match['similarity']),
                    'is_match': bool(match['is_match'])
                } for match in results['top_matches']
            ],
            'potential_plagiarism': bool(results['potential_plagiarism']),
            'average_similarity': float(results['average_similarity'])
        }
    except Exception as e:
        return {'error': str(e)}