from worker import process_rag_query
r = process_rag_query.delay(
    "La tua domanda qui",   # query
    [],                     # history (vuota per una domanda singola)
    "IL_TUO_TOPIC_ID",      # topic_id: deve esistere nel DB
)
print("task id:", r.id)
print(r.get(timeout=180))   # blocca finché il worker non risponde
