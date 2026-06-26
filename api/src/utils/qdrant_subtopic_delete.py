import os
import sys
import argparse
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

def delete_points_by_scope(topic_id: str, sub_topic_id: str):
    """
    Elimina da Qdrant tutti i chunk che corrispondono sia al topic_id che al sub_topic_id specificati.
    """
    # Configurazione connessione
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", 6333))
    
    client = QdrantClient(host=host, port=port)
    collection_name = "document_chunks" 

    print(f"Connesso a Qdrant su {host}:{port}")
    print(f"Avvio eliminazione mirata in '{collection_name}':")
    print(f"  ↳ topic_id     = '{topic_id}'")
    print(f"  ↳ sub_topic_id = '{sub_topic_id}'")

    try:
        # Esecuzione della cancellazione con doppia condizione in 'must'
        result = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="topic_id",
                            match=models.MatchValue(value=topic_id)
                        ),
                        models.FieldCondition(
                            key="sub_topic_id",
                            match=models.MatchValue(value=sub_topic_id)
                        )
                    ]
                )
            )
        )
        
        print("\n=== Operazione completata! ===")
        print(f"Risposta di Qdrant: {result}")
        
    except Exception as e:
        print(f"\n✗ Errore critico durante l'eliminazione su Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Configurazione del parser per i parametri nominali
    parser = argparse.ArgumentParser(
        description="Elimina in modo selettivo i chunk da Qdrant in base a topic_id e sub_topic_id."
    )
    
    # Aggiunta degli argomenti richiesti (--nome_parametro)
    parser.add_argument(
        "--topic_id", 
        type=str, 
        required=True, 
        help="L'ID del topic bersaglio (es. rag_system)"
    )
    parser.add_argument(
        "--sub_topic_id", 
        type=str, 
        required=True, 
        help="L'ID del sub-topic bersaglio (es. attiprovincia)"
    )
    
    # Esegue il parsing. Se mancano argomenti required, argparse interrompe 
    # automaticamente l'esecuzione e stampa un messaggio di errore chiaro.
    args = parser.parse_args()
    
    # Avvio della funzione con i parametri mappati correttamente
    delete_points_by_scope(topic_id=args.topic_id, sub_topic_id=args.sub_topic_id)