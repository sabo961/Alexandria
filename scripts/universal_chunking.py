#!/usr/bin/env python3
"""
Alexandria Universal Semantic Chunker
=====================================

A robust, semantic-aware text splitter that groups sentences by meaning
using vector similarity. Replaces fixed-window and keyword-based strategies.

Principles:
1. Semantic Integrity: Break where the topic changes, not where the word count ends.
2. Context Buffering: Maintain a minimum chunk size to ensure the AI has enough context.
3. Domain Agnostic: Works equally well for Philosophy, Technical Manuals, and Fiction.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class UniversalChunker:
    def __init__(
        self, 
        embedding_model,
        threshold: float = 0.5, 
        min_chunk_size: int = 200, 
        max_chunk_size: int = 1500
    ):
        """
        Args:
            embedding_model: An instance of SentenceTransformer or a generator function.
            threshold: Similarity threshold (0.0 - 1.0). Lower = fewer breaks, Higher = more breaks.
            min_chunk_size: Minimum words per chunk (prevents atomic/useless chunks).
            max_chunk_size: Maximum words per chunk (safety cap for LLM context limits).
        """
        self.model = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using a robust regex."""
        # Split by punctuation followed by space, keeping the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 2]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Splits text into semantically cohesive chunks.
        
        Returns:
            List of Dicts with 'text' and metadata.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Generate embeddings for all sentences at once for efficiency
        # Handle both class instances and raw models
        if hasattr(self.model, 'generate_embeddings'):
            raw_embeddings = self.model.generate_embeddings(sentences)
            # Filter out None values (Ollama failures) and replace with zero vectors
            dim = len(raw_embeddings[0]) if raw_embeddings and raw_embeddings[0] is not None else 1024
            embeddings = np.array([
                e if e is not None else [0.0] * dim 
                for e in raw_embeddings
            ])
        else:
            embeddings = self.model.encode(sentences, show_progress_bar=False)

        chunks = []
        current_sentences = [sentences[0]]
        current_word_count = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            word_count = len(sentence.split())
            
            # Calculate similarity with the previous sentence
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1), 
                embeddings[i].reshape(1, -1)
            )[0][0]

            # Decision Logic:
            # 1. If similarity is low (topic change)
            # 2. AND we have enough content in current buffer (min_chunk_size)
            # 3. OR the current buffer is dangerously large (max_chunk_size)
            
            should_break = (similarity < self.threshold and current_word_count >= self.min_chunk_size)
            must_break = (current_word_count + word_count > self.max_chunk_size)

            if should_break or must_break:
                # Close current chunk
                chunks.append(self._create_chunk_dict(" ".join(current_sentences), len(chunks), metadata))
                current_sentences = [sentence]
                current_word_count = word_count
            else:
                # Add to current chunk
                current_sentences.append(sentence)
                current_word_count += word_count

        # Add the final buffer
        if current_sentences:
            chunks.append(self._create_chunk_dict(" ".join(current_sentences), len(chunks), metadata))

        logger.info(f"Universal Chunker: Created {len(chunks)} chunks from {len(sentences)} sentences.")
        return chunks

    def _create_chunk_dict(self, text: str, index: int, metadata: Optional[Dict]) -> Dict:
        chunk_data = {
            "text": text,
            "chunk_id": index,
            "word_count": len(text.split()),
            "strategy": "universal-semantic"
        }
        if metadata:
            chunk_data.update(metadata)
        return chunk_data

# Simple test runner
if __name__ == "__main__":
    from scripts.ingest_books import EmbeddingGenerator
    
    test_text = """
    Philosophy is the study of general and fundamental questions. 
    It is a critical and systematic approach. 
    In contrast, a hammer is a tool meant for driving nails into wood. 
    Carpentry requires physical skill and precision. 
    Nietzsche often wrote about the will to power. 
    He was a German philosopher who challenged traditional morality. 
    """
    
    embedder = EmbeddingGenerator()
    chunker = UniversalChunker(embedder, threshold=0.4, min_chunk_size=5)
    result = chunker.chunk(test_text)
    
    for c in result:
        print(f"\n[CHUNK {c['chunk_id']}] ({c['word_count']} words):")
        print(f"Content: {c['text']}")
