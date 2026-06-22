import os
import asyncio
import json
import argparse
from typing import Optional
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models

load_dotenv()

async def inspect_collection_payloads(collection_name: str, limit: int = 5, file_name: Optional[str] = None):
    """
    Esegue lo scroll di una collection per mostrare ID e metadati.
    Se viene passato `file_name`, filtra i risultati per quel file specifico.
    """
    client = AsyncQdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"), 
        port=int(os.environ.get("QDRANT_PORT", 6333)),
        timeout=60.0
    )
    
    # Configura il filtro SOLO se il parametro file_name è presente
    scroll_filter = None
    filter_info = ""
    if file_name:
        filter_info = f" (filtrato per file: '{file_name}')"
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file_name)
                )
            ]
        )

    print(f"\n=== SCROLL COLLECTION: {collection_name}{filter_info} (Limite: {limit} record) ===")
    
    try:
        records, next_page_offset = await client.scroll(
            collection_name=collection_name,
            limit=limit,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            print("Nessun record trovato con i criteri specificati.")
            return

        for record in records:
            print(f"\n🏷️ [ID]: {record.id}")
            formatted_payload = json.dumps(record.payload, indent=2, ensure_ascii=False)
            print(f"📄 [Metadati / Payload]:\n{formatted_payload}")
            print("-" * 50)
            
        if next_page_offset:
            print(f"⏩ Ci sono altri dati. Offset per la pagina successiva: {next_page_offset}")
        else:
            print("✅ Fine dei dati per questa ricerca.")

    except Exception as e:
        print(f"Errore durante lo scroll su {collection_name}: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    # Configurazione del parser con un help personalizzato e formattato
    help_description = """
=== Qdrant Collection Inspector ===
Ispetta i metadati (payload) e gli ID dei record salvati in una collection di Qdrant.

Esempi di utilizzo:
  1. Base:
     python qdrant_scroll.py --collection document_chunks

  2. Limita il numero di risultati (es. 5 record):
     python qdrant_scroll.py --collection document_chunks --limit 5

  3. Filtra per nome di un file specifico:
     python qdrant_scroll.py --collection document_chunks --file_name "report_2023.pdf"
"""

    parser = argparse.ArgumentParser(
        description=help_description,
        formatter_class=argparse.RawTextHelpFormatter # Permette di mantenere le andate a capo nella stringa sopra
    )
    
    parser.add_argument(
        "-c", "--collection", 
        type=str, 
        required=True, 
        help="[OBBLIGATORIO] Il nome della collection in Qdrant da ispezionare."
    )
    parser.add_argument(
        "-f", "--file_name", 
        type=str, 
        default=None, 
        help="[OPZIONALE] Filtra i record mostrando solo i chunk appartenenti a questo file."
    )
    parser.add_argument(
        "-l", "--limit", 
        type=int, 
        default=15, 
        help="[OPZIONALE] Il numero massimo di record da stampare a schermo. (Default: 15)"
    )
    
    args = parser.parse_args()

    # Avvio dell'asincrono passando i parametri catturati
    asyncio.run(inspect_collection_payloads(
        collection_name=args.collection, 
        limit=args.limit, 
        file_name=args.file_name
    ))