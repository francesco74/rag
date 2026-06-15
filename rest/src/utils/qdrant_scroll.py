import os
import asyncio
import json
import argparse
from typing import Optional
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models

load_dotenv()

QDRANT_COLLECTION = "document_chunks"

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
            print(f"\n🔹 [ID]: {record.id}")
            formatted_payload = json.dumps(record.payload, indent=2, ensure_ascii=False)
            print(f"🔹 [Metadati / Payload]:\n{formatted_payload}")
            print("-" * 50)
            
        if next_page_offset:
            print(f"📌 Ci sono altri dati. Offset per la pagina successiva: {next_page_offset}")
        else:
            print("🏁 Fine dei dati per questa ricerca.")

    except Exception as e:
        print(f"Errore durante lo scroll su {collection_name}: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    # Configurazione degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Ispetta i metadati dei record in Qdrant.")
    parser.add_argument(
        "--file_name", 
        type=str, 
        default=None, 
        help="Nome del file per cui filtrare i chunk (opzionale)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=15, 
        help="Numero massimo di record da mostrare (default: 5)"
    )
    
    args = parser.parse_args()

    # Avvio dell'asincrono passando i parametri catturati
    asyncio.run(inspect_collection_payloads(
        collection_name=QDRANT_COLLECTION, 
        limit=args.limit, 
        file_name=args.file_name
    ))