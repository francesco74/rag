import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def list_topics_from_qdrant():
    # Connessione a Qdrant usando le variabili d'ambiente o i default
    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", 6333))
    )
    
    collection_name = "document_chunks"
    
    try:
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            print(f"❌ La collection '{collection_name}' non esiste in Qdrant.")
            return

        print("🔍 Scansione dei payload in Qdrant in corso...\n")
        unique_pairs = set()
        offset = None
        
        # Usiamo 'scroll' per scorrere tutti i vettori in lotti da 100
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=["topic_id", "sub_topic_id"], # Chiediamo solo i campi necessari
                with_vectors=False,                        # Risparmiamo banda
                offset=offset
            )
            
            for record in records:
                topic = record.payload.get("topic_id", "SCONOSCIUTO")
                sub_topic = record.payload.get("sub_topic_id", "NON_ASSEGNATO")
                unique_pairs.add((topic, sub_topic))
            
            if offset is None:
                break

        if not unique_pairs:
            print("⚠️ Nessun dato trovato nella collection.")
            return

        # Raggruppamento per una visualizzazione ad albero
        tree = {}
        for topic, sub_topic in unique_pairs:
            if topic not in tree:
                tree[topic] = set()
            tree[topic].add(sub_topic)
            
        # Stampa ad albero
        print("📊 STRUTTURA ATTUALE IN QDRANT:")
        print("===============================")
        for topic, sub_topics in sorted(tree.items()):
            print(f"📁 Macroarea (topic_id): {topic}")
            for st in sorted(sub_topics):
                print(f"   ↳ 📂 Sottocategoria: {st}")
        print("===============================\n")

    except Exception as e:
        print(f"🔥 Errore durante l'accesso a Qdrant: {e}")

if __name__ == "__main__":
    list_topics_from_qdrant()