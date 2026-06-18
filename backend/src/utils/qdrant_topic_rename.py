import os
from qdrant_client import QdrantClient, models

# Configurazione
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

OLD_TOPIC = "greenlees-carteggio"
NEW_TOPIC = "greenlees"

def rename_qdrant_topic():
    print(f"Inizio migrazione Qdrant da '{OLD_TOPIC}' a '{NEW_TOPIC}'...")
    
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # =========================================================
        # 1. AGGIORNA I CHUNK DEI DOCUMENTI (document_chunks)
        # =========================================================
        point_ids = []
        offset = None
        
        # Paginazione sicura per recuperare tutti gli ID
        while True:
            records, offset = qdrant.scroll(
                collection_name="document_chunks",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=OLD_TOPIC))
                ]),
                limit=500,
                with_payload=False # Zero overhead di rete, prendiamo solo gli ID
            )
            point_ids.extend([r.id for r in records])
            if offset is None:
                break
                
        if point_ids:
            # set_payload fa il merge. Modifica topic_id e lascia intatto "content", "source", ecc.
            qdrant.set_payload(
                collection_name="document_chunks",
                payload={"topic_id": NEW_TOPIC},
                points=point_ids
            )
            print(f"✓ Aggiornati {len(point_ids)} chunks in 'document_chunks'.")
        else:
            print("⚠ Nessun chunk trovato con questo topic_id.")

        # =========================================================
        # 2. ELIMINA LA VECCHIA CACHE (semantic_cache)
        # =========================================================
        qdrant.delete(
            collection_name="semantic_cache",
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="topic", match=models.MatchValue(value=OLD_TOPIC))
                ])
            )
        )
        print(f"✓ Cache semantica per '{OLD_TOPIC}' invalidata con successo.")
        
    except Exception as e:
        print(f"❌ Errore durante l'operazione su Qdrant: {e}")

if __name__ == "__main__":
    rename_qdrant_topic()