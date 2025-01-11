# Hindi BPE Implementation

This project implements a Byte Pair Encoding (BPE) tokenizer for Hindi text, with the following specifications:
- Vocabulary size: < 5000 tokens
- Target compression ratio: ≥ 3.2

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

1. First, download and process the Hindi dataset:
```bash
cd src
python download_data.py
```
This will:
- Download the Hindi corpus from the provided URL
- Process the first 5GB of data or 5 million lines (whichever comes first)
- Save the processed data in the `data` directory

2. Train the BPE tokenizer:
```bash
python train_bpe.py
```
This will:
- Load the processed data
- Train the BPE tokenizer with a vocabulary size < 5000
- Display the compression ratio and vocabulary size
- Run a test encoding/decoding on a sample Hindi text

## Implementation Details

The implementation includes:
- Custom BPE tokenizer for Hindi text
- Special token handling (<PAD>, <UNK>, <BOS>, <EOS>)
- Proper Unicode handling for Hindi characters
- Compression ratio calculation
- Progress bars for long-running operations

## File Structure
```
Assignment_10/
├── data/                  # Data directory (created during runtime)
│   ├── hi_raw.txt        # Raw downloaded Hindi corpus
│   └── hi_processed.txt  # Processed subset of the corpus
├── src/
│   ├── download_data.py  # Data download and processing script
│   ├── hindi_bpe.py      # BPE implementation
│   └── train_bpe.py      # Training script
└── requirements.txt      # Project dependencies
``` # BPE-Hindi
