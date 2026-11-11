"""
Elimina una collection di Qdrant, irreversibilmente. Per svuotare la cache delle
risposte c'è utils/qdrant_semantic_cache_reset, che la ricrea.

    python -m utils.qdrant_delete_collection document_chunks__vecchio --dry-run   # cosa eliminerebbe
    python -m utils.qdrant_delete_collection document_chunks__vecchio
"""
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from common.config import settings

def delete_qdrant_collection(collection_name, dry_run=False):
    print(f"Connessione a Qdrant ({settings.qdrant_host}:{settings.qdrant_port}) in corso...\n")
    try:
        qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        # 1. Verifica se la collection esiste
        try:
            qdrant.get_collection(collection_name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                print(f"⚠️ Operazione annullata: La collection '{collection_name}' non esiste nel database.")
                return
            else:
                raise e # Rilancia l'errore se è un problema diverso (es. auth o timeout)

        # 2. Con --dry-run si ferma qui
        if dry_run:
            punti = qdrant.count(collection_name, exact=True).count
            print(f"Verrebbe eliminata la collection '{collection_name}' ({punti} punti): nessuna modifica (--dry-run).")
            return

        # 3. Eliminazione
        print(f"Eliminazione di '{collection_name}' in corso...")
        qdrant.delete_collection(collection_name=collection_name)
        print(f"✓ Operazione completata. Collection '{collection_name}' eliminata con successo.")
            
    except Exception as e:
        print(f"❌ Errore durante la connessione o l'eliminazione: {e}")

if __name__ == "__main__":
    # Configurazione dei parametri da riga di comando
    parser = argparse.ArgumentParser(description="Elimina una specifica collection da Qdrant.")
    parser.add_argument(
        "collection", 
        type=str, 
        help="Il nome esatto della collection da eliminare"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mostra cosa eliminerebbe, senza eliminare"
    )
    
    args = parser.parse_args()
    
    delete_qdrant_collection(args.collection, args.dry_run)