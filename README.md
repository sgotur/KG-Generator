# Knowledge Graph Extractor

A Python tool for extracting knowledge graphs from unstructured text using LLM-based triple extraction. This tool processes text into Subject-Predicate-Object (SPO) triples that can be used to build knowledge graphs.

## Features

- Text chunking with configurable overlap
- LLM-based triple extraction using OpenAI/DeepSeek API
- JSON-formatted output
- Automatic pronoun resolution
- Error handling and failed chunk tracking
- Results export to pandas DataFrame

## Requirements

- Python 3.8+
- OpenAI API key or DeepSeek API credentials
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledge-graph-extractor
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Set your API credentials either through environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-base-url"  # Optional, for DeepSeek or other providers
```

Or provide them directly when initializing the extractor:
```python
extractor = KnowledgeGraphExtractor(
    api_key="your-api-key",
    api_base="your-api-base-url"  # Optional
)
```

## Usage

### Basic Usage

```python
from extract import KnowledgeGraphExtractor

# Initialize the extractor
extractor = KnowledgeGraphExtractor()

# Process text
text = """
Your unstructured text here...
"""
results = extractor.process_text(text)

# Get results as DataFrame
df = extractor.get_results_dataframe()
print(df)
```

### Running the Example

The repository includes an example script that demonstrates the usage with a sample text about Albert Einstein:

```bash
python example.py
```

### Customizing Chunk Size and Overlap

```python
extractor = KnowledgeGraphExtractor()
extractor.chunk_size = 200  # Adjust chunk size
extractor.overlap = 40     # Adjust overlap
```

### Handling Results

```python
results = extractor.process_text(text)

# Access extracted triples
triples = results['triples']

# Check for failed chunks
failed_chunks = results['failed_chunks']
for chunk in failed_chunks:
    print(f"Chunk {chunk['chunk_number']} failed: {chunk['error']}")
```

## Testing

The repository includes unit tests to verify the functionality:

```bash
python -m unittest test_extract.py
```

The tests cover:
- Text chunking functionality
- Configuration validation
- Empty text handling
- Results DataFrame structure

## Output Format

The extracted triples are returned in the following format:

```python
{
    'triples': [
        {
            'subject': 'entity1',
            'predicate': 'relation',
            'object': 'entity2',
            'chunk': 1  # chunk number where this triple was found
        },
        # ... more triples
    ],
    'failed_chunks': [
        {
            'chunk_number': 2,
            'error': 'error message',
            'response': 'raw response'
        },
        # ... any failed chunks
    ]
}
```

## Project Structure

```
.
├── extract.py           # Main implementation
├── example.py          # Example usage
├── test_extract.py     # Unit tests
├── requirements.txt    # Dependencies
├── README.md          # Documentation
├── LICENSE            # MIT License
└── .gitignore         # Git ignore file
```

## Limitations

- API rate limits apply based on your OpenAI/DeepSeek account
- Processing large texts may take time due to API calls
- Quality of extraction depends on the LLM model used
- Text chunks are processed independently, which may miss cross-chunk relationships

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 