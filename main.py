from pathlib import Path
from src.config import PDF_PATH, INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import PDFDocumentLoader
from src.chunker import TextSplitter
from src.vector_store import VectorStoreBuilder
from src.rag import SimpleRag


def build_index():
    if Path(INDEX_PATH).exists():
        print("Index already exists.")
        return

    print("Building index...")
    loader = PDFDocumentLoader(PDF_PATH)
    documents = loader.load()
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.chunk(documents)
    builder = VectorStoreBuilder()
    vector_store = builder.build(chunks)
    builder.save(vector_store, INDEX_PATH)
    print("Index saved.")


def main():
    build_index()

    rag = SimpleRag()
    result = rag.query("What is the core component of RAG?")

    print("\nAnswer:", result["answer"])
    print("\nContexts retrieved:", len(result["contexts"]))
    print("First context:", result["contexts"][0][:200])


if __name__ == "__main__":
    main()