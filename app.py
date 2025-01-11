import gradio as gr
from src.hindi_bpe import HindiBPE
import pickle
import os

# Initialize the tokenizer
tokenizer = HindiBPE(max_vocab_size=5000, target_compression=3.2)

# Load production model state
model_file = 'hindi_bpe_model.pkl'
if os.path.exists(model_file):
    print("Loading production model...")
    with open(model_file, 'rb') as f:
        state = pickle.load(f)
        tokenizer.vocab = state['vocab']
        tokenizer.inverse_vocab = state['inverse_vocab']
        tokenizer.bpe_ranks = state['bpe_ranks']
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")
else:
    raise FileNotFoundError("Production model not found! Please run train_bpe.py first and copy the model file.")

def process_text(text: str, mode: str) -> str:
    """Process text using the tokenizer"""
    if not text.strip():
        return "Please enter some text."
        
    if mode == "Encode":
        # Encode the text
        encoded = tokenizer.encode(text)
        return f"Encoded tokens: {encoded}"
    else:
        # First encode then decode to show the round trip
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        return f"Original: {text}\nDecoded: {decoded}\nMatches: {'✓' if text == decoded else '✗'}"

# Create the interface
iface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(label="Enter Hindi Text", placeholder="नमस्ते भारत"),
        gr.Radio(["Encode", "Encode & Decode"], label="Operation", value="Encode & Decode")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Hindi BPE Tokenizer (Production Model)",
    description="""This is a production-grade Byte Pair Encoding (BPE) tokenizer trained on 1 million Hindi sentences.
    Features:
    - Vocabulary size: < 5000 tokens
    - Compression ratio: ≥ 3.2
    - Trained on 1M sentences
    - Proper handling of Hindi Unicode characters and combining marks""",
    examples=[
        ["नमस्ते भारत", "Encode & Decode"],
        ["मैं हिंदी सीख रहा हूं", "Encode & Decode"],
        ["यह एक परीक्षण वाक्य है", "Encode & Decode"],
        ["भारत एक विशाल देश है", "Encode & Decode"],
        ["मुझे हिंदी भाषा बहुत पसंद है", "Encode & Decode"]
    ]
)

if __name__ == "__main__":
    iface.launch() 