import os
from qdrant_client import QdrantClient

# Configurazione
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

def list_qdrant_collections():
    print("Connessione a Qdrant in corso...\n")
    try:
        # Inizializzazione del client
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Recupero nativo di tutte le collection
        response = qdrant.get_collections()
        
        # Estrazione dei nomi dalla risposta
        collection_names = [col.name for col in response.collections]
        
        if not collection_names:
            print("⚠️ Nessuna collection trovata nel database.")
            return
            
        print(f"✓ Operazione completata. Trovate {len(collection_names)} collection:")
        print("-" * 40)
        for name in sorted(collection_names):
            print(f" • {name}")
        print("-" * 40)
            
    except Exception as e:
        print(f"❌ Errore durante la connessione o la lettura da Qdrant: {e}")

if __name__ == "__main__":
    list_qdrant_collections()