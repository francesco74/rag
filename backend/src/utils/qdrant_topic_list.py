import os
from qdrant_client import QdrantClient

# Configurazione
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

def list_qdrant_topics():
    print("Scansione di Qdrant in corso...\n")
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        unique_topics = set()
        offset = None
        
        # Paginazione veloce
        while True:
            records, offset = qdrant.scroll(
                collection_name="document_chunks",
                limit=1000,
                with_payload=["topic_id"], 
                with_vectors=False         
            )
            
            for r in records:
                # Estraiamo il topic e lo aggiungiamo al set (che ignora i duplicati in automatico O(1))
                topic = r.payload.get("topic_id")
                if topic:
                    unique_topics.add(topic)
            
            # Se offset è None, abbiamo letto tutta la collection
            if offset is None:
                break
                
        print(f"✓ Operazione completata. Trovati {len(unique_topics)} topic unici in 'document_chunks':")
        print("-" * 40)
        for t in sorted(unique_topics):
            print(f" • {t}")
        print("-" * 40)
            
    except Exception as e:
        print(f"❌ Errore durante la lettura da Qdrant: {e}")

if __name__ == "__main__":
    list_qdrant_topics()