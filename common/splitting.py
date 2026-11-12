"""
Suddivisione del testo in sezioni (parent) e pezzi (child), comune all'ingestione
degli archivi (ingestion/src/ingest.py) e a quella dei documenti di sessione.

Le dimensioni sono in token cl100k_base. Le sezioni seguono, se richiesto, le
intestazioni Markdown (metadati "Header 1".."Header 4"); i pezzi si ricavano da
ogni sezione e sono quelli che ricevono l'embedding.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
_markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)


def make_splitters(parent_chunk_size: int, chunk_size: int, chunk_overlap: int):
    """(splitter delle sezioni, splitter dei pezzi)."""
    parent = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=parent_chunk_size, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " "])
    child = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    return parent, child


def split_parents(text: str, parent_splitter, by_markdown_headers: bool):
    """Sezioni del testo (Document di langchain, con le intestazioni nei metadati)."""
    if by_markdown_headers:
        return parent_splitter.split_documents(_markdown_splitter.split_text(text))
    return parent_splitter.create_documents([text])


def split_children(parent_text: str, child_splitter):
    return child_splitter.create_documents([parent_text])
