import os
import requests
from tqdm import tqdm
import re

def download_file(url, filename):
    """
    Download file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def is_valid_sentence(text: str) -> bool:
    """Check if the text is a valid Hindi sentence"""
    # Must contain at least one Hindi character
    if not re.search(r'[\u0900-\u097F]', text):
        return False
    
    # Must end with a sentence ending punctuation or be of reasonable length
    if not re.search(r'[।?!\.]$', text) and len(text.split()) < 3:
        return False
        
    return True

def process_data(input_file, output_file, max_sentences=1_000_000):
    """
    Process the input file and take only first 1 million valid sentences
    """
    sentence_count = 0
    total_size = 0
    buffer = []

    print(f"Processing first {max_sentences:,} sentences...")
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        pbar = tqdm(total=max_sentences, desc="Processing sentences")
        
        for line in infile:
            # Split line into potential sentences
            # Split on common Hindi sentence endings and other punctuation
            potential_sentences = re.split(r'([।?!\.])\s*', line.strip())
            
            # Process each potential sentence
            i = 0
            while i < len(potential_sentences):
                # Combine sentence text with its punctuation
                if i + 1 < len(potential_sentences) and potential_sentences[i+1] in '।?!.':
                    sentence = potential_sentences[i] + potential_sentences[i+1]
                    i += 2
                else:
                    sentence = potential_sentences[i]
                    i += 1
                
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if it's a valid sentence
                if is_valid_sentence(sentence):
                    outfile.write(sentence + '\n')
                    total_size += len(sentence.encode('utf-8'))
                    sentence_count += 1
                    pbar.update(1)
                    
                    if sentence_count >= max_sentences:
                        break
            
            if sentence_count >= max_sentences:
                break
        
        pbar.close()
    
    print(f"Processed {sentence_count:,} sentences")
    print(f"Total size: {total_size/(1024*1024):.2f} MB")

def main():
    # Create data directory if it doesn't exist
    os.makedirs("../data", exist_ok=True)
    
    # URL of the Hindi dataset
    url = "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/v1-indiccorp/hi.txt"
    raw_file = "../data/hi_raw.txt"
    processed_file = "../data/hi_processed.txt"
    
    # Download if file doesn't exist
    if not os.path.exists(raw_file):
        print("Downloading Hindi dataset...")
        download_file(url, raw_file)
    
    # Process the file
    print("Processing the dataset...")
    process_data(raw_file, processed_file)

if __name__ == "__main__":
    main() 