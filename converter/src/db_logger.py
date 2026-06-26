import os
import socket
import logging
import sys  # <--- AGGIUNGI QUESTO IMPORT
from mysql.connector import pooling

_db_pool = None

def init_db_pool(host, port, user, password, database):
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = pooling.MySQLConnectionPool(
                pool_name="ingest_worker_pool",
                pool_size=10,  # FIX: Alza la dimensione a 10 per sostenere i 5 task concorrenti + i log simultanei
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=10
            )
        except Exception as e:
            print(f"CRITICAL: Impossibile inizializzare il DB Pool: {e}", file=sys.stderr)

def get_db_connection():
    global _db_pool
    if _db_pool is None:
        # Ritorna None senza sollevare eccezioni distruttive durante la fase di bootstrap
        return None
    try:
        return _db_pool.get_connection()
    except Exception as e:
        print(f"Database connection pool exhausted or failed: {e}", file=sys.stderr) 
        return None

class MySQLLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.pod_name = os.environ.get("HOSTNAME", socket.gethostname())

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            log_msg = self.format(record) 
            conn = None
            try:
                conn = get_db_connection()
                if not conn:
                    # Se il pool non è pronto, stampiamo su console e usciamo in sicurezza
                    print(f"⚠️ DB Logger passivo: Pool non ancora pronto per registrare: {record.levelname} - {log_msg}", file=sys.stderr)
                    return
                    
                with conn.cursor() as cursor:
                    query = """
                        INSERT INTO system_logs 
                        (log_level, message, file_name, line_no, pod_name) 
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(query, (
                        record.levelname, 
                        log_msg, 
                        record.filename, 
                        record.lineno, 
                        self.pod_name
                    ))
                    conn.commit()
            except Exception as e:
                print(f"CRITICAL: Failed to write log to database: {e}", file=sys.stderr)
            finally:
                if conn:
                    conn.close()