"""
Stato condiviso del motore di risposta.

`qdrant_client` è creato dal worker DOPO il fork (worker.init_worker_process):
i moduli lo leggono sempre da qui (`state.qdrant_client`) e mai con
`from rag.state import qdrant_client`, che ne copierebbe il valore None di
partenza.
"""

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
CONCEPT_COLLECTION = "conceptual_dictionary"
BOILERPLATE_COLLECTION = "boilerplate_phrases"

qdrant_client = None
