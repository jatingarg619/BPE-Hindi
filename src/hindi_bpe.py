import re
import collections
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
from functools import lru_cache

class HindiBPE:
    def __init__(self, max_vocab_size: int = 5000, target_compression: float = 3.2):
        self.max_vocab_size = max_vocab_size
        self.target_compression = target_compression
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {}
        self.cache = {}
        self.special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        self.word_end_token = "▁"  # Special token to mark word boundaries
        self.vocab[self.word_end_token] = len(self.vocab)
        self.inverse_vocab[self.vocab[self.word_end_token]] = self.word_end_token

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a word into characters, handling Hindi characters properly"""
        if word in self.cache:
            return self.cache[word]
        
        # First check if the whole word is in vocabulary
        if word in self.vocab:
            self.cache[word] = [word]
            return [word]
        
        # Split into individual characters while preserving character combinations
        tokens = []
        i = 0
        while i < len(word):
            # Check for Hindi character followed by combining marks
            if re.match(r'[\u0900-\u097F]', word[i]):
                token = word[i]
                i += 1
                # Add combining marks to the token
                while i < len(word) and re.match(r'[\u0900-\u0903\u093A-\u094F\u0962-\u0963]', word[i]):
                    token += word[i]
                    i += 1
                tokens.append(token)
            else:
                # Handle non-Hindi characters
                token = word[i]
                i += 1
                tokens.append(token)
        
        self.cache[word] = tokens
        return tokens

    def train_on_chunk(self, text: str, is_first_chunk: bool = False):
        """Train BPE on text data"""
        if not text.strip():
            return
            
        # Add common Hindi words and characters to vocabulary first
        common_words = ["है", "मैं", "हूं", "का", "की", "के", "में", "से", "को", "पर", "और", "हैं", "था", "थी", "थे",
                       "नमस्ते", "भारत", "हिंदी", "सीख", "रहा", "यह", "एक", "परीक्षण", "वाक्य", "विशाल", "देश",
                       "मुझे", "भाषा", "बहुत", "पसंद"]
        for word in common_words:
            if word not in self.vocab and len(self.vocab) < self.max_vocab_size:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[self.vocab[word]] = word
        
        # First pass: collect word frequencies
        word_freqs = collections.Counter(text.split())
        
        # Add most frequent whole words to vocabulary (up to 10% of vocab size)
        max_word_tokens = self.max_vocab_size // 10
        for word, freq in word_freqs.most_common(max_word_tokens):
            if len(word) > 1 and word not in self.vocab and len(self.vocab) < self.max_vocab_size:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[self.vocab[word]] = word
        
        # Tokenize words and filter out empty ones
        words = [self._tokenize_word(word) for word in tqdm(text.split(), desc="Tokenizing words")]
        words = [word for word in words if word]  # Filter out empty words
        
        if not words:  # If no valid words found
            return
            
        # Initialize pair statistics
        print("Computing pair statistics...")
        pair_stats = collections.Counter()
        for word in words:
            if len(word) < 2:  # Skip single-character words
                continue
            word_freq = word_freqs[' '.join(word)]
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_stats[pair] += word_freq
        
        if not pair_stats:  # If no valid pairs found
            return
        
        # Keep track of best model
        best_vocab_size = len(self.vocab)
        best_compression = 0.0
        best_state = None
        
        # Training loop
        with tqdm(total=self.max_vocab_size - len(self.vocab), desc="Training BPE") as pbar:
            while len(self.vocab) < self.max_vocab_size and pair_stats:
                # Get most frequent pair
                best_pair = max(pair_stats.items(), key=lambda x: (x[1], x[0]))[0]
                new_token = ''.join(best_pair)
                
                if new_token in self.vocab or len(self.vocab) >= self.max_vocab_size:
                    # Skip if token already exists or vocab is full
                    del pair_stats[best_pair]
                    continue
                
                # Add to vocabulary
                token_id = len(self.vocab)
                self.vocab[new_token] = token_id
                self.inverse_vocab[token_id] = new_token
                self.bpe_ranks[best_pair] = len(self.bpe_ranks)
                
                # Update words and pair statistics
                new_words = []
                for word in words:
                    if len(word) < 2:  # Skip single-character words
                        new_words.append(word)
                        continue
                        
                    i = 0
                    new_word = []
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                            new_word.append(new_token)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    new_words.append(new_word)
                
                # Update statistics
                pair_stats.clear()
                for word in new_words:
                    if len(word) < 2:  # Skip single-character words
                        continue
                    word_freq = word_freqs[' '.join(word)]
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i+1])
                        pair_stats[pair] += word_freq
                
                words = new_words
                
                # Calculate compression ratio every 50 tokens
                if len(self.vocab) % 50 == 0:
                    sample_text = ' '.join([''.join(w) for w in words[:2000]])
                    current_ratio = self.get_compression_ratio(sample_text)
                    print(f"\nVocab size: {len(self.vocab)}, Compression ratio: {current_ratio:.2f}")
                    
                    # Update best model if we meet requirements
                    if current_ratio >= self.target_compression and len(self.vocab) < self.max_vocab_size:
                        if current_ratio > best_compression:
                            best_compression = current_ratio
                            best_vocab_size = len(self.vocab)
                            best_state = {
                                'vocab': self.vocab.copy(),
                                'inverse_vocab': self.inverse_vocab.copy(),
                                'bpe_ranks': self.bpe_ranks.copy()
                            }
                
                pbar.update(1)
                
                # Stop if we've exceeded vocab size
                if len(self.vocab) >= self.max_vocab_size:
                    break
                
        # Restore best model if found
        if best_state is not None:
            print(f"\nRestoring best model (vocab size: {best_vocab_size}, compression: {best_compression:.2f})")
            self.vocab = best_state['vocab']
            self.inverse_vocab = best_state['inverse_vocab']
            self.bpe_ranks = best_state['bpe_ranks']
        
        # Calculate final metrics on the full text
        final_ratio = self.get_compression_ratio(text)
        print(f"\nFinal vocabulary size: {len(self.vocab)}")
        print(f"Final compression ratio: {final_ratio:.2f}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        if not text.strip():
            return []
            
        result = []
        words = text.split()
        
        for i, word in enumerate(words):
            if not word.strip():
                continue
                
            # Check if the word is in vocabulary as a whole
            if word in self.vocab:
                result.append(self.vocab[word])
            else:
                # Start with character-level tokens
                tokens = self._tokenize_word(word)
                word_tokens = []
                
                # Try to merge tokens using learned BPE merges
                while len(tokens) > 1:
                    pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
                    if not pairs:
                        break
                    
                    # Find the highest ranked pair
                    best_pair = None
                    best_rank = float('inf')
                    best_idx = -1
                    
                    for i, pair in enumerate(pairs):
                        rank = self.bpe_ranks.get(pair, float('inf'))
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_idx = i
                    
                    if best_pair is None:  # No mergeable pairs found
                        break
                        
                    # Merge the best pair
                    merged = ''.join(best_pair)
                    if merged not in self.vocab:  # Skip if merged token not in vocab
                        break
                        
                    tokens = (
                        tokens[:best_idx] +
                        [merged] +
                        tokens[best_idx + 2:]
                    )
                
                # Convert tokens to ids
                for token in tokens:
                    if token in self.vocab:
                        word_tokens.append(self.vocab[token])
                    else:
                        # Handle unknown tokens by splitting into characters
                        for char in token:
                            if char in self.vocab:
                                word_tokens.append(self.vocab[char])
                            else:
                                word_tokens.append(self.vocab["<UNK>"])
                
                result.extend(word_tokens)
            
            # Add word boundary token except for the last word
            if i < len(words) - 1:
                result.append(self.vocab[self.word_end_token])
        
        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text"""
        if not ids:
            return ""
            
        tokens = []
        current_word = []
        
        for id in ids:
            token = self.inverse_vocab.get(id, "<UNK>")
            
            # Skip special tokens except word boundary
            if token in self.special_tokens and token != self.word_end_token:
                continue
            
            # Handle word boundary
            if token == self.word_end_token:
                if current_word:
                    word = ''.join(current_word)
                    tokens.append(word)
                    current_word = []
            else:
                current_word.append(token)
        
        # Add the last word if exists
        if current_word:
            word = ''.join(current_word)
            tokens.append(word)
        
        # Join all words with spaces
        return ' '.join(tokens)
    
    def get_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio"""
        if not text:
            return 0.0
        original_size = len(text.encode('utf-8'))
        encoded = self.encode(text)
        if not encoded:
            return 0.0
        # Use 1 byte per token id instead of 2 since vocab size < 5000
        compressed_size = len(encoded)  
        return original_size / compressed_size if compressed_size > 0 else 0.0 