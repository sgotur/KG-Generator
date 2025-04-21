#!/usr/bin/env python3

import os
import json
import math
import re
import warnings
import sys
from openai import OpenAI
import networkx as nx
import ipycytoscape
import ipywidgets
import pandas as pd
from neptune_graph import NeptuneGraphManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class KnowledgeGraphExtractor:
    def __init__(self, api_key=None, api_base=None, model_name=None, neptune_endpoint=None):
        """Initialize the Knowledge Graph Extractor with API credentials and model settings."""
        # API Configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.model_name = model_name or os.getenv("MODEL_NAME", "deepseek-chat")
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "API key is required. Please provide it either:\n"
                "1. As an argument: KnowledgeGraphExtractor(api_key='your-key')\n"
                "2. Through environment variable OPENAI_API_KEY in .env file"
            )
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # LLM Configuration
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", 4096))
        
        # Chunking Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 150))  # Number of words per chunk
        self.overlap = int(os.getenv("OVERLAP", 30))      # Number of words to overlap between chunks

        # Storage for results
        self.chunks = []
        self.all_extracted_triples = []
        self.failed_chunks = []

        # Initialize Neptune connection if endpoint is provided
        self.neptune_manager = None
        neptune_endpoint = neptune_endpoint or os.getenv("NEPTUNE_ENDPOINT")
        if neptune_endpoint:
            try:
                self.neptune_manager = NeptuneGraphManager(neptune_endpoint=neptune_endpoint)
                print("Neptune connection initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Neptune connection: {str(e)}")

    def validate_chunking_config(self):
        """Validate the chunking configuration."""
        if self.overlap >= self.chunk_size and self.chunk_size > 0:
            raise ValueError(f"Overlap ({self.overlap}) must be smaller than chunk size ({self.chunk_size}).")
        return True

    def split_text_into_chunks(self, text):
        """Split input text into overlapping chunks."""
        words = text.split()
        total_words = len(words)
        start_index = 0
        chunk_number = 1

        print(f"Starting chunking process...")

        while start_index < total_words:
            end_index = min(start_index + self.chunk_size, total_words)
            chunk_text = " ".join(words[start_index:end_index])
            self.chunks.append({
                "text": chunk_text,
                "chunk_number": chunk_number,
                "word_count": end_index - start_index
            })

            next_start_index = start_index + self.chunk_size - self.overlap

            if next_start_index <= start_index:
                if end_index == total_words:
                    break
                next_start_index = start_index + 1

            start_index = next_start_index
            chunk_number += 1

            if chunk_number > total_words:  # Safety check
                print("Warning: Chunking loop exceeded total word count, breaking.")
                break

        print(f"Text successfully split into {len(self.chunks)} chunks.")
        return self.chunks

    @property
    def extraction_system_prompt(self):
        """Get the system prompt for the LLM."""
        return """
        You are an AI expert specialized in knowledge graph extraction.
        Your task is to identify and extract factual Subject-Predicate-Object (SPO) triples from the given text.
        Focus on accuracy and adhere strictly to the JSON output format requested in the user prompt.
        Extract core entities and the most direct relationship.
        """

    @property
    def extraction_user_prompt_template(self):
        """Get the user prompt template for the LLM."""
        return """
        Please extract Subject-Predicate-Object (S-P-O) triples from the text below.

        **VERY IMPORTANT RULES:**
        1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object".
        2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
        3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
        4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
        5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
        6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
        7.  **Completeness:** Extract all distinct factual relationships mentioned.

        **Text to Process:**
        ```text
        {text_chunk}
        ```
        """

    def normalize_triple(self, triple):
        """Normalize a triple by trimming spaces and converting to lowercase."""
        return {
            'subject': triple['subject'].strip().lower(),
            'predicate': triple['predicate'].strip().lower(),
            'object': triple['object'].strip().lower(),
            'chunk': triple.get('chunk', 0)  # Preserve chunk information if present
        }

    def is_valid_triple(self, triple):
        """Check if a triple is valid (non-empty after normalization)."""
        return (triple['subject'] and 
                triple['predicate'] and 
                triple['object'])

    def deduplicate_triples(self, triples):
        """Remove duplicate triples while preserving the earliest chunk number."""
        # Create a dictionary to store unique triples
        unique_triples = {}
        
        for triple in triples:
            # Create a key from the normalized triple parts
            key = (triple['subject'], triple['predicate'], triple['object'])
            
            # If this is the first time we've seen this triple, or if it came from an earlier chunk
            if key not in unique_triples or triple['chunk'] < unique_triples[key]['chunk']:
                unique_triples[key] = triple
        
        # Convert back to list
        return list(unique_triples.values())

    def clean_triples(self, triples):
        """Clean and normalize a list of triples."""
        # Normalize all triples
        normalized = [self.normalize_triple(t) for t in triples]
        
        # Filter out invalid triples
        valid = [t for t in normalized if self.is_valid_triple(t)]
        
        # Remove duplicates
        unique = self.deduplicate_triples(valid)
        
        return unique

    def process_chunk(self, chunk):
        """Process a single chunk and extract triples."""
        prompt = self.extraction_user_prompt_template.format(text_chunk=chunk['text'])

        try:
            print(f"\nProcessing chunk {chunk['chunk_number']}...")
            print(f"Chunk text length: {len(chunk['text'])} characters")
            
            # Call LLM with system + user prompt
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.extraction_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens
                )
                print(f"API call successful for chunk {chunk['chunk_number']}")
            except Exception as e:
                print(f"API call failed for chunk {chunk['chunk_number']}: {str(e)}")
                raise
            
            raw = res.choices[0].message.content.strip()
            print(f"Raw response for chunk {chunk['chunk_number']}:")
            print("-" * 50)
            print(raw[:500] + "..." if len(raw) > 500 else raw)  # Print first 500 chars of response
            print("-" * 50)
            
            # Parse response
            try:
                # First try to parse as direct JSON array
                data = json.loads(raw)
                print(f"Successfully parsed direct JSON for chunk {chunk['chunk_number']}")
            except json.JSONDecodeError:
                try:
                    # Try to find JSON array in the text
                    match = re.search(r'(\[.*\])', raw, re.DOTALL)
                    if match:
                        data = json.loads(match.group(1))
                        print(f"Successfully extracted JSON array using regex for chunk {chunk['chunk_number']}")
                    else:
                        print(f"No JSON array found in response for chunk {chunk['chunk_number']}")
                        data = []
                except Exception as e:
                    print(f"Failed to parse JSON for chunk {chunk['chunk_number']}: {str(e)}")
                    data = []

            # Handle different response formats
            if isinstance(data, dict):
                # Try to find a list in the dictionary values
                triples_list = None
                for value in data.values():
                    if isinstance(value, list):
                        triples_list = value
                        break
                if triples_list is not None:
                    data = triples_list
                    print(f"Found list in dictionary response for chunk {chunk['chunk_number']}")
                else:
                    print(f"No list found in dictionary response for chunk {chunk['chunk_number']}")
                    data = []
            elif not isinstance(data, list):
                print(f"Unexpected response format for chunk {chunk['chunk_number']}")
                data = []

            # Validate and store triples
            triples = []
            for t in data:
                if (isinstance(t, dict) and 
                    all(k in t and isinstance(t[k], str) for k in ['subject', 'predicate', 'object'])):
                    triples.append(dict(t, chunk=chunk['chunk_number']))
                else:
                    print(f"Invalid triple format in chunk {chunk['chunk_number']}: {t}")

            print(f"Extracted {len(triples)} valid triples from chunk {chunk['chunk_number']}")

            if triples:
                # Clean and normalize the triples
                cleaned_triples = self.clean_triples(triples)
                print(f"After cleaning: {len(cleaned_triples)} unique triples from chunk {chunk['chunk_number']}")
                self.all_extracted_triples.extend(cleaned_triples)
                return True
            else:
                self.failed_chunks.append({
                    'chunk_number': chunk['chunk_number'],
                    'error': 'No valid triples',
                    'response': raw
                })
                print(f"No valid triples found in chunk {chunk['chunk_number']}")
                return False

        except Exception as e:
            error_msg = f"Error processing chunk {chunk['chunk_number']}: {str(e)}"
            print(error_msg)
            self.failed_chunks.append({
                'chunk_number': chunk['chunk_number'],
                'error': str(e),
                'response': ''
            })
            return False

    def process_text(self, text):
        """Process the entire text and extract knowledge graph triples."""
        # Validate configuration
        self.validate_chunking_config()
        
        # Split text into chunks
        self.split_text_into_chunks(text)
        
        print(f"\nStarting triple extraction from {len(self.chunks)} chunks using model '{self.model_name}'...")
        print(f"Chunk size: {self.chunk_size} words, Overlap: {self.overlap} words")
        
        # Process each chunk
        for i, chunk in enumerate(self.chunks, 1):
            print(f"\n{'='*50}")
            print(f"Processing chunk {i}/{len(self.chunks)}")
            print(f"Chunk number: {chunk['chunk_number']}")
            print(f"Word count: {chunk['word_count']}")
            print(f"{'='*50}")
            
            success = self.process_chunk(chunk)
            if success:
                print(f"Successfully processed chunk {chunk['chunk_number']}")
            else:
                print(f"Failed to process chunk {chunk['chunk_number']}")
        
        # Clean all extracted triples one final time to handle cross-chunk duplicates
        print("\nCleaning all extracted triples...")
        self.all_extracted_triples = self.clean_triples(self.all_extracted_triples)
        
        print(f"\nProcessing complete!")
        print(f"Total chunks processed: {len(self.chunks)}")
        print(f"Total triples extracted: {len(self.all_extracted_triples)}")
        print(f"Failed chunks: {len(self.failed_chunks)}")
        
        # Store triples in Neptune if connection is available
        if self.neptune_manager and self.all_extracted_triples:
            try:
                self.neptune_manager.store_triples(self.all_extracted_triples)
                print("\nTriples stored in Neptune successfully!")
                
                # Visualize the graph
                G = self.neptune_manager.get_graph_data()
                self.neptune_manager.visualize_graph(G, output_file='knowledge_graph.png')
                
                # Create interactive visualization
                interactive_graph = self.neptune_manager.visualize_interactive(G)
                display(interactive_graph)
                
            except Exception as e:
                print(f"Error storing/visualizing triples in Neptune: {str(e)}")
        
        return {
            'triples': self.all_extracted_triples,
            'failed_chunks': self.failed_chunks
        }

    def get_results_dataframe(self):
        """Convert extracted triples to a pandas DataFrame."""
        if not self.all_extracted_triples:
            return pd.DataFrame()
        return pd.DataFrame(self.all_extracted_triples)

    def close(self):
        """Close any open connections."""
        if self.neptune_manager:
            self.neptune_manager.close()

def main():
    try:
        # Set pandas display options to show full content
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # Example usage with Neptune integration
        neptune_endpoint = os.getenv('NEPTUNE_ENDPOINT')  # Get from environment variable
        extractor = KnowledgeGraphExtractor(neptune_endpoint=neptune_endpoint)
        
        # Sample text about Albert Einstein
        text = """
        Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who developed the theory of relativity, 
        one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. 
        He is best known to the general public for his mass–energy equivalence formula E = mc², which has been dubbed "the world's most famous equation".
        
        Einstein received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect".
        He published more than 300 scientific papers and more than 150 non-scientific works. His intellectual achievements and originality have made the word "Einstein" 
        synonymous with "genius".
        """
        
        print("Processing text about Albert Einstein...")
        results = extractor.process_text(text)
        
        # Display results
        print("\nExtracted Triples:")
        df = extractor.get_results_dataframe()
        if not df.empty:
            print("\n" + "="*80)
            print("Extracted Knowledge Graph Triples:")
            print("="*80)
            print(df.to_string())
            print("="*80)
            print(f"\nTotal triples extracted: {len(df)}")
        else:
            print("No triples were extracted.")
        
        if results['failed_chunks']:
            print("\nFailed Chunks:")
            for chunk in results['failed_chunks']:
                print(f"Chunk {chunk['chunk_number']}: {chunk['error']}")
    
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'extractor' in locals():
            extractor.close()

if __name__ == "__main__":
    main() 
