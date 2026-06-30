import uuid
import logging
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import genai
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("populate_concepts")

# Configurazione Costanti
CONCEPT_COLLECTION = "conceptual_dictionary"
INPUT_FILE_PATH = "concepts.txt"
EMBEDDING_MODEL = "gemini-embedding-001"

def init_services():
    """Inizializza le connessioni a Qdrant e Gemini."""
    # Configura Gemini
    api_llm_key = settings.api_llm_key
    if not api_llm_key:
        raise ValueError("API_LLM_KEY non trovata nelle variabili d'ambiente!")
    genai.configure(api_key=api_llm_key)

    # Configura Qdrant
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

def parse_and_clean_file(file_path):
    """Legge il file di testo, ignora commenti/righe vuote e pulisce i dati."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file di input '{file_path}' non esiste!")

    parsed_concepts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Salta righe vuote o commenti
            if not line or line.startswith("#"):
                continue
                
            if ":" not in line:
                log.warning(f"Riga {line_num} ignorata (formato non valido, manca il separatore ':'): '{line}'")
                continue
                
            concept_part, aliases_part = line.split(":", 1)
            
            # Pulizia profonda dei testi
            concept = concept_part.strip()
            aliases = [a.strip().lower() for a in aliases_part.split(",") if a.strip()]
            
            if not concept or not aliases:
                log.warning(f"Riga {line_num} ignorata (concetto o alias vuoti).")
                continue
                
            parsed_concepts.append({
                "concept": concept,
                "aliases": aliases
            })
            
    return parsed_concepts

def reset_qdrant_collection(client):
    """Cancella ed esegue il reset ex novo della collection e dei suoi indici."""
    log.info(f"Piazza pulita: rimozione della collection '{CONCEPT_COLLECTION}' se esistente...")
    if client.collection_exists(CONCEPT_COLLECTION):
        client.delete_collection(CONCEPT_COLLECTION)
        
    log.info(f"Creazione nuova collection '{CONCEPT_COLLECTION}'...")
    client.create_collection(
        collection_name=CONCEPT_COLLECTION,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )
    
    # Ricreazione immediata degli indici di payload per ottimizzare il retrieval
    log.info("Configurazione indici di payload...")
    client.create_payload_index(CONCEPT_COLLECTION, "concept", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(CONCEPT_COLLECTION, "aliases", models.PayloadSchemaType.KEYWORD)

def main():
    try:
        client = init_services()
        
        # 1. Parsing e pulizia del file di testo
        log.info(f"Lettura e pulizia del file: {INPUT_FILE_PATH}...")
        concepts_data = parse_and_clean_file(INPUT_FILE_PATH)
        log.info(f"Trovati {len(concepts_data)} concetti validi pronti per l'indicizzazione.")
        
        if not concepts_data:
            log.warning("Nessun dato valido trovato nel file. Processo interrotto.")
            return

        # 2. Reset totale del Vector DB (Ex Novo)
        reset_qdrant_collection(client)

        # 3. Generazione Embedding e Caricamento dei Punti
        points = []
        log.info("Generazione degli embedding semantici tramite Gemini...")
        
        for item in concepts_data:
            concept = item["concept"]
            aliases = item["aliases"]
            
            # Strategia di Embedding: fondiamo il concetto con i suoi alias per dare 
            # all'embedding la massima densità semantica possibile.
            text_to_embed = f"{concept}: {', '.join(aliases)}"
            
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text_to_embed,
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
            vector = result['embedding']
            
            # Creazione del punto Qdrant strutturato
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "concept": concept,
                    "aliases": aliases
                }
            )
            points.append(point)
            log.info(f"✓ Pronto: '{concept}' con {len(aliases)} alias correlati.")

        # 4. Upsert finale in blocco (Batch)
        log.info(f"Caricamento di {len(points)} punti su Qdrant...")
        client.upsert(collection_name=CONCEPT_COLLECTION, points=points)
        log.info("=== Processo completato con successo! Il dizionario semantico è aggiornato. ===")

    except Exception as e:
        log.error(f"✗ Errore critico durante il popolamento: {e}", exc_info=True)

if __name__ == "__main__":
    main()