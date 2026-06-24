import os
import socket
import logging
from mysql.connector import pooling

# Pool globale mantenuto in memoria
_db_pool = None

def init_db_pool(host, port, user, password, database):
    """Inizializza il pool di connessioni una sola volta all'avvio dell'app."""
    global _db_pool
    if _db_pool is None:
        try:
            _db_pool = pooling.MySQLConnectionPool(
                pool_name="ingest_worker_pool",
                pool_size=5,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=10
            )
        except Exception as e:
            print(f"CRITICAL: Impossibile inizializzare il DB Pool: {e}")

def get_db_connection():
    """Restituisce una connessione dal pool."""
    global _db_pool
    if _db_pool is None:
        # Fail-fast se qualcuno chiama il DB prima dell'inizializzazione
        raise RuntimeError("DB Pool non inizializzato. Chiama init_db_pool() prima.")
    try:
        return _db_pool.get_connection()
    except Exception as e:
        print(f"Database connection pool exhausted or failed: {e}") 
        return None

class MySQLLogHandler(logging.Handler):
    """Custom logging handler to send ERROR and CRITICAL logs to MySQL."""
    def __init__(self):
        super().__init__()
        self.pod_name = os.environ.get("HOSTNAME", socket.gethostname())

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            log_msg = self.format(record) 
            try:
                conn = get_db_connection()
                if not conn:
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
                print(f"CRITICAL: Failed to write log to database: {e}")
            finally:
                if 'conn' in locals() and conn:
                    conn.close()