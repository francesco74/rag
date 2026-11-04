"""Cache semantica delle risposte."""

import hashlib
import json
import logging
import time
import uuid

from qdrant_client import models

from rag import state
from rag.state import CACHE_COLLECTION

log = logging.getLogger("WORKER")


def generate_filters_key(metadata_filters):
    """Genera una chiave univoca basata sui filtri applicati per non inquinare la cache."""
    if not metadata_filters: return "no_filters"
    # Ordina le chiavi per garantire che lo stesso set generi sempre lo stesso hash
    filter_str = json.dumps(metadata_filters, sort_keys=True)
    return hashlib.md5(filter_str.encode()).hexdigest()


def check_semantic_cache(query_vector, topic_id, sub_topics_key, filters_key):
    """Check if similar query exists in cache."""
    if not state.qdrant_client:
        return None
    
    try:
        hits = state.qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=0.97,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="sub_topics_key", match=models.MatchValue(value=sub_topics_key)),
                    models.FieldCondition(key="filters_key", match=models.MatchValue(value=filters_key)) 
                ]
            )
        ).points
        
        if hits:
            log.info(f"✓ Cache HIT (similarity: {hits[0].score:.3f}) for topic '{topic_id}' & sub_topics '{sub_topics_key}'")
            return hits[0].payload
        
        log.debug("Cache MISS")
        return None
        
    except Exception as e:
        log.error(f"Cache check failed: {e}")
        return None


def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id, sub_topics_key, filters_key):
    """Salva la risposta in cache includendo la chiave di combinazione dei sub-topic."""
    if not state.qdrant_client:
        return
    
    try:
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "original_query": original_query,
                "answer": answer,
                "sources": sources,
                "topic_id": topic_id,
                "sub_topics_key": sub_topics_key, 
                "filters_key": filters_key,
                "timestamp": time.time()
            }
        )
        state.qdrant_client.upsert(collection_name=CACHE_COLLECTION, points=[point])
        log.debug("Cache entry saved.")
        
    except Exception as e:
        log.error(f"Cache save failed: {e}")


def generate_sub_topics_key(selected_sub_topics):
   return ",".join(sorted(selected_sub_topics))
