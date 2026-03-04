# Document Intelligence Refinery

A multi-stage agentic pipeline for enterprise document extraction. This system intelligently analyzes PDFs to determine the optimal extraction strategy, balancing cost, speed, and accuracy.

## Features

*   **Intelligent Triage:** Analyzes document characteristics (origin, layout complexity, domain) to select the best extraction strategy.
*   **Multi-Strategy Extraction:**
    *   **Strategy A (FastText):** High-speed extraction for native digital documents using `pdfplumber`. 
    *   **Strategy B (Layout):** Deep learning-based layout analysis (mocked integration with Docling/MinerU) for complex layouts and tables.
    *   **Strategy C (Vision):** VLM-based extraction (GPT-4o-mini) for scanned images or highly unstructured content, with budget guards.
*   **Escalation Guard:** Automatically upgrades from fast extraction to layout analysis if confidence scores are low.
*   **Comprehensive Logging:** detailed audit trail of all extraction attempts, costs, and decisions in `.refinery/extraction_ledger.jsonl`.
*   **Standardized Output:** All strategies output a unified `ExtractedDocument` Pydantic model.

## Prerequisites

*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (recommended) or Poetry/Pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Document-Intelligence-Refinery
    ```

2.  **Set up the environment:**
    Using `uv` (fastest):
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

    Using Poetry:
    ```bash
    poetry install
    ```

3.  **Set Environment Variables:**
    If you plan to use the Vision Strategy (Strategy C), set your OpenRouter or OpenAI API key:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

## Usage

### 1. Triage a Document
Analyze a document to see its classification and recommended strategy.

```bash
python src/agents/triage.py path/to/document.pdf
```

### 2. Run Extraction Pipeline
To run the full pipeline (Triage -> Route -> Extract -> Escalate if needed), you can use a simple script or the python shell:

```python
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

# 1. Analyze
triage = TriageAgent()
profile = triage.analyze("path/to/document.pdf")
print(f"Profile: {profile.origin_type}, Cost: {profile.estimated_extraction_cost}")

# 2. Extract
router = ExtractionRouter()
doc = router.extract("path/to/document.pdf", profile)

print(f"Extracted {len(doc.text_blocks)} text blocks.")
print(f"Metadata: {doc.metadata}")
```

## Running Tests

The project includes a suite of unit tests for the Triage Agent, FastText confidence scoring, and Router escalation logic.

```bash
pytest tests/
```

## Project Structure

```
.
├── pyproject.toml          # Project configuration and dependencies
├── rubric
│   └── extraction_rules.yaml # Configuration for thresholds and domains
├── src
│   ├── agents
│   │   ├── triage.py       # Triage Agent implementation
│   │   └── extractor.py    # Extraction Router & Escalation Logic
│   ├── models
│   │   └── __init__.py     # Pydantic Schemas (DocumentProfile, ExtractedDocument, etc.)
│   └── strategies
│       ├── base.py         # Abstract Base Class for strategies
│       ├── fast_text.py    # Strategy A: pdfplumber
│       ├── layout.py       # Strategy B: Mock Docling/MinerU
│       └── vision.py       # Strategy C: GPT-4o-mini VLM
└── tests                   # Unit tests
```

## Docker Support

You can build and run the application using Docker, ensuring a consistent environment.

1.  **Build the Docker Image:**
    ```bash
    docker build -t document-intelligence-refinery .
    ```

2.  **Run the Container:**
    ```bash
    docker run --rm -v $(pwd):/app/data document-intelligence-refinery pytest tests/
    ```

## Configuration

You can tune the classification thresholds and domain keywords in `rubric/extraction_rules.yaml`

