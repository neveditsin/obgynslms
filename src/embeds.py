import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.data_loader import load_documents


def get_top_similar(filename, efile='embeddings.pkl', k=5):
    # Load embeddings
    with open(efile, 'rb') as f:
        file_to_emb = pickle.load(f)
    
    if filename not in file_to_emb:
        raise ValueError(f"File '{filename}' not found in embeddings.")
    
    # Get embedding for the given file
    target_emb = file_to_emb[filename]
    
    # Get all files and embeddings
    files = list(file_to_emb.keys())
    all_embs = np.array(list(file_to_emb.values()))
    
    # Compute cosine similarities
    dots = np.dot(all_embs, target_emb)
    norms_all = np.linalg.norm(all_embs, axis=1)
    norm_target = np.linalg.norm(target_emb)
    sims = dots / (norms_all * norm_target)
    
    # Get indices of top similarities, excluding itself
    self_index = files.index(filename)
    sims[self_index] = -1  # Mask self to exclude
    top_indices = np.argsort(sims)[-k:][::-1]  # Top k descending
    
    # Calculate mean and median (excluding self)
    valid_sims = np.delete(sims, self_index)
    mean_sim = np.mean(valid_sims)
    median_sim = np.median(valid_sims)
    
    # Return list of (file, similarity score) tuples, mean, and median
    top_similar = [(files[i], sims[i]) for i in top_indices]
    return top_similar, mean_sim, median_sim
