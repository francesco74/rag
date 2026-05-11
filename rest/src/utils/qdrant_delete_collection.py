import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Configurazione
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

def delete_qdrant_collection(collection_name, force=False):
    print(f"Connessione a Qdrant ({QDRANT_HOST}:{QDRANT_PORT}) in corso...\n")
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 1. Verifica se la collection esiste
        try:
            qdrant.get_collection(collection_name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                print(f"⚠️ Operazione annullata: La collection '{collection_name}' non esiste nel database.")
                return
            else:
                raise e # Rilancia l'errore se è un problema diverso (es. auth o timeout)

        # 2. Conferma manuale (salvo se c'è il flag --force)
        if not force:
            print(f"⚠️  ATTENZIONE: Stai per eliminare IRREVERSIBILMENTE la collection '{collection_name}'.")
            confirm = input("Tutti i vettori e i payload andranno persi. Vuoi procedere? (y/n): ")
            if confirm.lower() != 'y':
                print("Operazione annullata dall'utente.")
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
        "-f", "--force", 
        action="store_true", 
        help="Salta la richiesta di conferma (utile negli script CI/CD)"
    )
    
    args = parser.parse_args()
    
    delete_qdrant_collection(args.collection, args.force)