
import sys
try:
    import docling
    print(f"Docling version: {docling.__version__}")
    try:
        from docling.document_converter import DocumentConverter
        print("Import success")
    except ImportError as e:
        print(f"Import from docling.document_converter failed: {e}")
except ImportError as e:
    print(f"Import docling failed: {e}")
