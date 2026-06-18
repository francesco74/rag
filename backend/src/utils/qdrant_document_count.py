import os
from qdrant_client import QdrantClient

def count_qdrant_metrics():
    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    
    collection_name = "document_chunks"
    
    try:
        # Verifica se la collection esiste
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            print(f"❌ La collection '{collection_name}' non esiste.")
            return

        print("📊 CALCOLO METRICHE QDRANT IN CORSO...\n")

        # 1. Conteggio Vettori (Chunks) - Chiamata nativa ultraveloce
        total_chunks = client.count(collection_name=collection_name).count
        
        # 2. Conteggio Documenti Univoci (Sources) - Scroll del payload
        unique_sources = set()
        offset = None
        
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=500,               # Batch più grande per scorrere in fretta
                with_payload=["source"], # Chiediamo SOLO il campo source per minimizzare la RAM
                with_vectors=False,
                offset=offset
            )
            
            for record in records:
                source = record.payload.get("source")
                if source:
                    unique_sources.add(source)
            
            if offset is None:
                break

        print(f"🔹 Totale frammenti vettoriali (Chunks): {total_chunks}")
        print(f"📄 Totale documenti fisici (Sorgenti Uniche): {len(unique_sources)}")
        
        # Opzionale: decommenta per vedere i nomi dei file
        # print("\nLista Documenti:")
        # for doc in sorted(unique_sources):
        #     print(f" - {doc}")

    except Exception as e:
        print(f"🔥 Errore durante l'accesso a Qdrant: {e}")

if __name__ == "__main__":
    count_qdrant_metrics()