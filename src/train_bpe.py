import os
from tqdm import tqdm
from hindi_bpe import HindiBPE
import pickle

def load_data_in_chunks(file_path: str, chunk_size: int = 10000, max_lines: int = 1_000_000):
    """Load and process the Hindi dataset in chunks"""
    print(f"Loading data from {file_path}")
    chunk = []
    total_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading lines"):
            if total_lines >= max_lines:
                if chunk:  # Yield the last chunk
                    yield '\n'.join(chunk)
                break
                
            if line.strip():
                chunk.append(line.strip())
                if len(chunk) >= chunk_size:
                    yield '\n'.join(chunk)
                    chunk = []
            total_lines += 1
            
        if chunk:  # Don't forget the last chunk
            yield '\n'.join(chunk)

def main():
    # Initialize tokenizer
    tokenizer = HindiBPE(max_vocab_size=5000, target_compression=3.2)
    
    # Load and process data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data_file = os.path.join(data_dir, 'hi_processed.txt')
    
    if not os.path.exists(data_file):
        print("Error: Processed data file not found. Please run download_data.py first.")
        return
    
    print("\nTraining BPE tokenizer...")
    # Train on chunks of data
    best_compression = 0
    chunk_count = 0
    is_first_chunk = True
    
    for chunk in load_data_in_chunks(data_file):
        chunk_count += 1
        print(f"\nProcessing chunk {chunk_count}...")
        
        # Train on this chunk
        tokenizer.train_on_chunk(chunk, is_first_chunk=is_first_chunk)
        is_first_chunk = False
        
        # Calculate metrics
        current_ratio = tokenizer.get_compression_ratio(chunk)
        vocab_size = len(tokenizer.vocab)
        
        print(f"Current metrics:")
        print(f"- Vocabulary size: {vocab_size} tokens")
        print(f"- Compression ratio: {current_ratio:.2f}")
        
        # Check if we've met our goals
        if current_ratio >= 3.2 and vocab_size < 5000:
            if current_ratio > best_compression:
                best_compression = current_ratio
                # Save the best model
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                os.makedirs(model_path, exist_ok=True)
                
                state = {
                    'vocab': tokenizer.vocab,
                    'inverse_vocab': tokenizer.inverse_vocab,
                    'bpe_ranks': tokenizer.bpe_ranks
                }
                
                model_file = os.path.join(model_path, 'hindi_bpe_model.pkl')
                with open(model_file, 'wb') as f:
                    pickle.dump(state, f)
                print(f"New best model saved! (compression ratio: {current_ratio:.2f})")
        
        # Early stopping if we've met our goals
        if current_ratio >= 3.2 and vocab_size < 5000 and chunk_count >= 5:
            print("\nMet all requirements and processed enough chunks. Stopping early.")
            break
    
    # Test the final model
    test_cases = [
        "नमस्ते भारत",
        "मैं हिंदी सीख रहा हूं",
        "यह एक परीक्षण वाक्य है",
        "भारत एक विशाल देश है",
        "मुझे हिंदी भाषा बहुत पसंद है"
    ]
    
    print("\nTesting final model:")
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nOriginal: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Matches: {'✓' if text == decoded else '✗'}")
    
    # Print final metrics
    print("\nFinal Results:")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")
    print(f"Best compression ratio: {best_compression:.2f}")

if __name__ == "__main__":
    main() 