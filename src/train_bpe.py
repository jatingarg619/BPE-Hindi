import os
from hindi_bpe import HindiBPE
from tqdm import tqdm

def load_processed_data_in_chunks(file_path: str, max_sentences: int = 1_000_000) -> str:
    """Load data in chunks, up to max_sentences"""
    buffer = []
    sentence_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading sentences"):
            if sentence_count >= max_sentences:
                break
                
            line = line.strip()
            if not line:
                continue
                
            buffer.append(line)
            sentence_count += 1
            
            if len(buffer) >= 10000:  # Process in chunks of 10K sentences
                yield ' '.join(buffer)
                buffer = []
    
    if buffer:  # Don't forget the last chunk
        yield ' '.join(buffer)

def main():
    # Initialize paths
    data_dir = os.path.join("..", "data")
    processed_file = os.path.join(data_dir, "hi_processed.txt")
    
    # Check if processed data exists
    if not os.path.exists(processed_file):
        print("Processed data not found. Please run download_data.py first.")
        return
    
    # Initialize BPE
    print("Initializing BPE tokenizer...")
    print("Training Parameters:")
    print("1. Using first 1 million sentences")
    print("2. Vocabulary size must be < 5000 tokens")
    print("3. Compression ratio must be ≥ 3.2")
    bpe = HindiBPE()
    
    print("\nTraining BPE model...")
    is_first_chunk = True
    total_sentences = 0
    
    for chunk in load_processed_data_in_chunks(processed_file):
        if not chunk.strip():
            continue
            
        bpe.train_on_chunk(chunk, is_first_chunk=is_first_chunk)
        is_first_chunk = False
        
        # Check if we've met both requirements
        test_text = chunk[:10000]  # Use a sample of text
        compression_ratio = bpe.get_compression_ratio(test_text)
        vocab_size = len(bpe.vocab)
        
        print(f"\nCurrent status:")
        print(f"Vocabulary size: {vocab_size} tokens")
        print(f"Compression ratio: {compression_ratio:.2f}")
        
        if compression_ratio >= 3.2:
            if vocab_size < 5000:
                print("\nSuccess! Met all requirements:")
                print(f"1. Vocabulary size: {vocab_size} tokens (< 5000)")
                print(f"2. Compression ratio: {compression_ratio:.2f} (≥ 3.2)")
                break
            else:
                print("\nWarning: Need to reduce vocabulary size while maintaining compression ratio")
    
    print("\nFinal Results:")
    print(f"Vocabulary size: {len(bpe.vocab)} tokens")
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    # Test the model with various Hindi texts
    test_cases = [
        "नमस्ते भारत",
        "मैं हिंदी सीख रहा हूं",
        "यह एक परीक्षण वाक्य है",
        "भारत एक विशाल देश है",
        "मुझे हिंदी भाषा बहुत पसंद है"
    ]
    
    print("\nTesting encoding/decoding on multiple examples:")
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Original: {test_text}")
        encoded = bpe.encode(test_text)
        print(f"Encoded: {encoded}")
        decoded = bpe.decode(encoded)
        print(f"Decoded: {decoded}")
        print(f"Matches: {'✓' if decoded == test_text else '✗'}")

if __name__ == "__main__":
    main() 