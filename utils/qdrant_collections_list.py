import os
from qdrant_client import QdrantClient
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

# Configurazione


def list_qdrant_collections():
    print("Connessione a Qdrant in corso...\n")
    try:
        # Inizializzazione del client
        qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
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