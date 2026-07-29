import uuid
import logging
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from common.embedding import init_embedding, embed_for_semantic_query
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("populate_concepts")

# Configurazione Costanti
CONCEPT_COLLECTION = "conceptual_dictionary"
INPUT_FILE_PATH = "concepts.txt"

def init_services():
    """
    Inizializza Qdrant e il client di embedding.

    L'embedding è centralizzato in embedding.py (init_embedding +
    embed_for_semantic_query), lo stesso modulo usato a runtime dal worker.
    In questo modo il dizionario concettuale viene indicizzato con esattamente
    lo stesso SDK, modello e task_type con cui poi viene interrogato: un
    requisito non negoziabile, perché il meccanismo si basa interamente sulla
    comparabilità via cosine similarity fra i due lati.
    """
    if not settings.api_llm_key:
        raise ValueError("API_LLM_KEY non trovata nelle variabili d'ambiente!")
    # Client di embedding centralizzato in embedding.py (stesso SDK/modello/
    # parametri del worker): garantisce che i vettori del dizionario siano
    # confrontabili con quelli calcolati a query-time da embed_for_semantic_query.
    init_embedding()

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return qdrant

def parse_and_clean_file(file_path):
    """
    Legge il file di testo, ignora righe vuote e pulisce i dati.

    Le intestazioni di sezione (righe che iniziano con '#', es.
    "# --- DIZIONARIO DEI CONCETTI GEOGRAFICI ---") non vengono più scartate
    come puro commento: definiscono la CATEGORIA dei concetti che seguono,
    fino alla prossima intestazione. Questo permette a worker.py di
    interrogare il dizionario in modo scoped (es. "solo concetti geografici"),
    invece di un'unica ricerca su tutta la collection che mescola zone
    geografiche e concetti amministrativi/tecnici nello stesso spazio
    vettoriale.

    "geografico" è l'unica categoria con un trattamento speciale a valle (in
    worker.py, per evitare di ri-applicare una decomposizione per comune che
    il rewriter ha già fatto da sé): qualunque altra intestazione diventa
    semplicemente "amministrativo". Aggiungere nuove sezioni al file (es.
    "# --- CONCETTI TURISTICI ---") non richiede modifiche al codice: finché
    non contengono la parola "geografic", rientrano automaticamente nel
    bucket "amministrativo".
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file di input '{file_path}' non esiste!")

    parsed_concepts = []
    current_category = "amministrativo"  # default prudente se il file non ha intestazioni

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                header_text = line.strip("# -").strip().lower()
                if "geografic" in header_text:
                    current_category = "geografico"
                elif header_text:
                    current_category = "amministrativo"
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
                "aliases": aliases,
                "category": current_category,
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
    client.create_payload_index(CONCEPT_COLLECTION, "category", models.PayloadSchemaType.KEYWORD)

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
            category = item["category"]
            
            # Strategia di Embedding: fondiamo il concetto con i suoi alias per dare 
            # all'embedding la massima densità semantica possibile.
            text_to_embed = f"{concept}: {', '.join(aliases)}"

            # SEMANTIC_SIMILARITY, 768 dim: stessa funzione con cui worker.py
            # interroga il dizionario, così i vettori sono confrontabili.
            vector = embed_for_semantic_query(text_to_embed)
            
            # Creazione del punto Qdrant strutturato
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "concept": concept,
                    "aliases": aliases,
                    "category": category,
                }
            )
            points.append(point)
            log.info(f"✓ Pronto: '{concept}' ({category}) con {len(aliases)} alias correlati.")

        # 4. Upsert finale in blocco (Batch)
        log.info(f"Caricamento di {len(points)} punti su Qdrant...")
        client.upsert(collection_name=CONCEPT_COLLECTION, points=points)
        log.info("=== Processo completato con successo! Il dizionario semantico è aggiornato. ===")

    except Exception as e:
        log.error(f"✗ Errore critico durante il popolamento: {e}", exc_info=True)

if __name__ == "__main__":
    main()