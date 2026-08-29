import sys
import torch
import torch.nn as nn
from typing import List, Tuple, Set

def is_target_script(token_str: str) -> bool:
    """
    Checks if a token belongs strictly to English, Persian, numbers, or standard punctuation.
    Discards tokens containing Cyrillic, CJK, Devanagari, Greek, Hebrew, Thai, etc.
    """
    for char in token_str:
        cp = ord(char)
        # Standard ASCII / English (space to ~)
        if 0x0020 <= cp <= 0x007E:
            continue
        # Persian & Arabic Unicode blocks
        if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0xFB50 <= cp <= 0xFDFF) or (0xFE70 <= cp <= 0xFEFF):
            continue
        # Digits
        if char.isdigit():
            continue
        # SentencePiece subword markers
        if char in [" ", " ", "#", "@", "<", ">", "[", "]", "_", "-", "/", ",", ".", ":", ";", "(", ")", "'", '"']:
            continue
        return False
    return True

def prune_vocab_and_weights(tokenizer, model: nn.Module) -> Tuple[List[str], List[int]]:
    """
    Prunes the HuggingFace tokenizer vocabulary and slices the model's embedding weight tensor
    to retain only Persian + English tokens.
    """
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    
    kept_tokens = []
    kept_indices = []
    
    for token, token_id in vocab.items():
        if token_id in special_ids or is_target_script(token):
            kept_tokens.append(token)
            kept_indices.append(token_id)
            
    # Sort indices to preserve monotonic order
    kept_indices_sorted = sorted(kept_indices)
    index_tensor = torch.tensor(kept_indices_sorted, dtype=torch.long)
    
    # Get original input embeddings
    orig_embeddings = model.encoder.get_input_embeddings()
    orig_weight = orig_embeddings.weight.data
    
    # Slice the embedding matrix directly
    new_weight = orig_weight[index_tensor].clone()
    
    # Resize model embeddings and copy weights
    model.encoder.resize_token_embeddings(len(kept_indices_sorted))
    model.encoder.get_input_embeddings().weight.data.copy_(new_weight)
    
    print(f"[Vocab Pruner] Original tokens: {len(vocab):,} -> Pruned tokens: {len(kept_indices_sorted):,} ({len(kept_indices_sorted)/len(vocab)*100:.1f}%)")
    return kept_tokens, kept_indices_sorted
