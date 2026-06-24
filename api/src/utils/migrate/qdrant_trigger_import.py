# trigger_recover.py
import os
import requests

# Configurazione endpoint Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant.rag.svc.cluster.local")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_HTTP_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

COLLECTIONS = ["document_chunks", "parent_documents"]

def trigger_local_restore():
    print(f"[*] Invio comandi di restore a Qdrant ({QDRANT_HTTP_URL})...")
    
    for col in COLLECTIONS:
        url = f"{QDRANT_HTTP_URL}/collections/{col}/snapshots/recover"
        
        # Questo dice a Qdrant di leggere il file che hai appena pusgato in /tmp via kubectl cp
        payload = {"location": f"file:///tmp/{col}.snapshot"}
        headers = {"Content-Type": "application/json"}
        
        print(f"\n[+] Innesco ripristino locale per la collection '{col}'...")
        try:
            # Chiamata PUT (esattamente come il comando curl mancante)
            response = requests.put(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                print(f"✓ Qdrant ha preso in carico il file: {response.json()}")
            else:
                print(f"× Errore ({response.status_code}): {response.text}")
                
        except Exception as e:
            print(f"× Errore di rete/connessione durante l'innesco di {col}: {e}")

if __name__ == "__main__":
    trigger_local_restore()