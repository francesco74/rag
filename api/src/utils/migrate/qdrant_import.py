# import_qdrant.py
import os
import requests
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


# Configurazione dinamica tramite variabili d'ambiente
QDRANT_HTTP_URL = f"http://{settings.qdrant_host}:{settings.qdrant_port}"

# Cartella in cui hai depositato i file .snapshot (es. la cartella qdrant_snapshots creata prima)
SNAPSHOT_DIR = "./"  

def import_snapshots():
    # Identifica tutti i file .snapshot presenti nella cartella
    try:
        files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".snapshot")]
    except Exception as e:
        print(f"× Impossibile accedere alla cartella {SNAPSHOT_DIR}: {e}")
        return
        
    if not files:
        print(f"[-] Nessun file .snapshot trovato nella directory '{SNAPSHOT_DIR}'.")
        print("[*] Ricordati di spostare i file recuperati con 'kubectl cp' dentro questa cartella.")
        return

    print(f"[*] Inizio ripristino sul nuovo database Qdrant ({settings.qdrant_host}:{settings.qdrant_port})...")
    
    for file in files:
        # Estrae il nome della collection dal nome del file (es. document_chunks.snapshot -> document_chunks)
        col_name = file.replace(".snapshot", "")
        print(f"\n[*] Upload e ripristino di '{col_name}' dal file '{file}'...")
        
        file_path = os.path.join(SNAPSHOT_DIR, file)
        
        # Endpoint nativo REST di Qdrant per il recupero da file multipart
        recover_url = f"{QDRANT_HTTP_URL}/collections/{col_name}/snapshots/recover"
        
        try:
            with open(file_path, "rb") as f:
                files_payload = {"snapshot": f}
                
                # Impostiamo un timeout molto lungo (5 minuti) perché Qdrant, oltre a ricevere il file, 
                # deve decomprimerlo e ricostruire gli indici in RAM/Disco prima di rispondere.
                response = requests.post(recover_url, files=files_payload, timeout=300)
            
            if response.status_code == 200:
                print(f"✓ Collection '{col_name}' importata e ripristinata con successo!")
            else:
                print(f"× Errore durante il ripristino di '{col_name}' (Codice {response.status_code}):")
                print(f"  Dettaglio: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"× Errore: Timeout durante il ripristino di '{col_name}'. Il file potrebbe essere troppo grande o il server è saturo.")
        except Exception as e:
            print(f"× Errore imprevisto di I/O o di rete per '{col_name}': {e}")

if __name__ == "__main__":
    import_snapshots()