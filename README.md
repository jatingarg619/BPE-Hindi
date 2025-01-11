# Hindi BPE Implementation

This project implements a Byte Pair Encoding (BPE) tokenizer specifically designed for Hindi text, with the following requirements:
- Vocabulary size: < 5000 tokens
- Target compression ratio: ≥ 3.2

## Features
- Custom BPE tokenizer optimized for Hindi characters and word combinations
- Proper handling of Hindi Unicode characters and combining marks
- Efficient compression while maintaining text integrity
- Special token handling (<PAD>, <UNK>, <BOS>, <EOS>)
- Word boundary preservation using special tokens

## Requirements
- Python 3.7+
- tqdm
- regex
- requests

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Download and process the Hindi dataset:
```bash
cd src
python download_data.py
```

2. Train the BPE tokenizer:
```bash
python train_bpe.py
```

## Test Cases

The implementation is tested with various Hindi text examples to ensure proper encoding and decoding:

```python
# Test Case 1: Basic greeting
Original: नमस्ते भारत
Expected: Complete preservation of both words

# Test Case 2: Complex sentence with special characters
Original: मैं हिंदी सीख रहा हूं
Expected: Proper handling of 'ैं' in मैं and 'ूं' in हूं

# Test Case 3: Multiple word sentence
Original: यह एक परीक्षण वाक्य है
Expected: Preservation of compound words like परीक्षण

# Test Case 4: Common Hindi sentence
Original: भारत एक विशाल देश है
Expected: Proper handling of all words including है

# Test Case 5: Longer sentence with various word types
Original: मुझे हिंदी भाषा बहुत पसंद है
Expected: Complete preservation of all words
```

## Implementation Details

### Key Components
1. **Tokenization**:
   - Character-level tokenization with Hindi combining marks
   - Word boundary preservation
   - Special handling of common Hindi words

2. **Training Process**:
   - Frequency-based pair selection
   - Vocabulary size control (< 5000 tokens)
   - Compression ratio optimization (≥ 3.2)

3. **Encoding/Decoding**:
   - Efficient token merging strategy
   - Proper handling of unknown tokens
   - Word boundary maintenance

### File Structure
```
Assignment_10/
├── data/                  # Data directory
│   ├── hi_raw.txt        # Raw Hindi corpus
│   └── hi_processed.txt  # Processed corpus
├── src/
│   ├── download_data.py  # Data processing
│   ├── hindi_bpe.py      # Core BPE implementation
│   └── train_bpe.py      # Training script
└── requirements.txt      # Dependencies
```

## Performance Metrics
- Vocabulary Size: 4950 tokens (< 5000 requirement)
- Compression Ratio: 3.66 (> 3.2 requirement)
- Accurate encoding/decoding of Hindi text
- Proper handling of special characters and word boundaries
