#!/usr/bin/env python3
"""
Alexandria Fixed-Size Chunker
=============================

Fast, simple chunking by word count with overlap.
No embeddings during chunking = 10-20x faster than semantic chunking.
"""

import re
import uuid
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class FixedChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: Target words per chunk
            overlap: Words to overlap between chunks
            min_chunk_size: Minimum words for a valid chunk
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 2]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into fixed-size chunks with overlap.
        
        Returns:
            List of dicts with 'text' and 'id'
        """
        words = text.split()
        if len(words) < self.min_chunk_size:
            # Too short, return as single chunk
            if len(words) < 10:
                return []
            return [{
                'id': str(uuid.uuid4()),
                'text': text.strip(),
                **(metadata or {})
            }]

        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Only add if meets minimum size (except last chunk)
            if len(chunk_words) >= self.min_chunk_size or end == len(words):
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'text': chunk_text,
                    **(metadata or {})
                })
            
            # Move start with overlap
            start = end - self.overlap if end < len(words) else end
            
            # Prevent infinite loop
            if start <= 0 or (end == len(words)):
                break

        logger.info(f"Fixed Chunker: Created {len(chunks)} chunks from {len(words)} words")
        return chunks
