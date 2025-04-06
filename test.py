# detect.py - Optimized for 100 test files
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Fix for Windows encoding

import os
import glob
import pickle
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

# Configuration 
TEST_FOLDER = "test_dataset/*.txt"  # Path to your 100 test files
EMBEDDINGS_PATH = "embeddings/trained_embeddings.pkl"
SIMILARITY_THRESHOLD = 0.90  # Adjust as needed (0.8-0.9 recommended)
BATCH_SIZE = 8  # Optimal for 100 files (reduce if GPU memory errors occur)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load pre-trained models and embeddings"""
    print("[1/3] Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    
    print("[2/3] Loading document embeddings...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings_db = pickle.load(f)
    
    return tokenizer, model, embeddings_db

def process_files(file_paths, tokenizer, model):
    """Process files in batches"""
    file_embeddings = {}
    for i in tqdm(range(0, len(file_paths), BATCH_SIZE), desc="Processing files"):
        batch_paths = file_paths[i:i+BATCH_SIZE]
        batch_texts = []
        
        # Read files
        for path in batch_paths:
            with open(path, 'r', encoding='utf-8') as f:
                batch_texts.append(f.read())
        
        # Generate embeddings
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                         truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeds = torch.mean(outputs.last_hidden_state, dim=1).cpu()
        
        # Store results
        for j, path in enumerate(batch_paths):
            file_embeddings[os.path.basename(path)] = batch_embeds[j]
    
    return file_embeddings

def analyze_results(test_embeddings, db):
    """Compare against reference embeddings"""
    results = {}
    for test_name, test_embed in tqdm(test_embeddings.items(), desc="Analyzing"):
        similarities = []
        for ref_name, ref_embed in db.items():
            sim = cosine_similarity(test_embed.numpy().reshape(1, -1), 
                                  ref_embed.numpy().reshape(1, -1))[0][0]
            similarities.append((ref_name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results[test_name] = {
            'is_plagiarism': similarities[0][1] > SIMILARITY_THRESHOLD,
            'top_match': similarities[0],
            'all_matches': [m for m in similarities if m[1] > SIMILARITY_THRESHOLD]
        }
    return results

def generate_report(results):
    """Print formatted results"""
    print("\n[3/3] Results:")
    for file, res in results.items():
        status = "PLAGIARISM" if res['is_plagiarism'] else "ORIGINAL"
        print(f"\n• File: {file}")
        print(f"  Status: {status}")
        print(f"  Best match: {res['top_match'][0]} ({res['top_match'][1]:.2%})")
        if res['is_plagiarism']:
            print(f"  Total matches: {len(res['all_matches'])}")
            print("  Top 3 matches:")
            for match in res['all_matches'][:3]:
                print(f"    - {match[0]} ({match[1]:.2%})")

def main():
    # Load resources
    tokenizer, model, db = load_models()
    
    # Get test files
    test_files = sorted(glob.glob(TEST_FOLDER))[:100]  # Ensure exactly 100
    print(f"Found {len(test_files)} test files")
    
    # Process test documents
    test_embeddings = process_files(test_files, tokenizer, model)
    
    # Compare against references
    results = analyze_results(test_embeddings, db)
    
    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()