# train.py - Processes training files into embeddings
import os
import glob
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Configuration
TRAIN_DATA_PATH = "train_dataset/*.txt"  # Path to your all training files
EMBEDDINGS_FILE = "embeddings/trained_embeddings.pkl"  # Output file
BATCH_SIZE = 16  # Adjust based on your GPU memory 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embedding(text, tokenizer, model):
    """Generate BERT embedding for a single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding='max_length', max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).cpu()  # [1, 768] shape

def process_files(file_paths, tokenizer, model):
    """Process files in batches for efficiency"""
    embeddings = {}
    for i in tqdm(range(0, len(file_paths), BATCH_SIZE), desc="Processing files"):
        batch_paths = file_paths[i:i+BATCH_SIZE]
        batch_texts = []
        
        # Read all files in batch
        for path in batch_paths:
            with open(path, 'r', encoding='utf-8') as f:
                batch_texts.append(f.read())
        
        # Batch process embeddings
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                         padding='max_length', max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeds = torch.mean(outputs.last_hidden_state, dim=1).cpu()
        
        # Store results
        for j, path in enumerate(batch_paths):
            embeddings[os.path.basename(path)] = batch_embeds[j]
    
    return embeddings

def main():
    # Setup output directory
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    
    # Initialize BERT
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    
    # Get all training files
    train_files = glob.glob(TRAIN_DATA_PATH)  # This line was corrected
    
    if not train_files:
        raise FileNotFoundError(f"No training files found in {TRAIN_DATA_PATH}")
    
    print(f"Found {len(train_files)} training files")
    
    # Process files
    embeddings = process_files(train_files, tokenizer, model)
    
    # Save embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"\n✅ Successfully processed {len(embeddings)} files")
    print(f"Embeddings saved to {EMBEDDINGS_FILE} (Size: {os.path.getsize(EMBEDDINGS_FILE)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()