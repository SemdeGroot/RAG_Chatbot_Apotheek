#!/usr/bin/env python3
"""
Shortcut zodat je ook 'python -m augmented_generation' kunt gebruiken.
Equivalent aan: python -m augmented_generation.rag_chat --interactive --db data/vectordb
"""
from .rag_chat import main as _main

if __name__ == "__main__":
    _main()

