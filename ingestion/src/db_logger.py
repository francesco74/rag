import os
import socket
import logging
import mysql.connector

def get_db_connection():
    """Helper function to get a MySQL connection using environment variables."""
    db_config = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASS"),
        "database": os.environ.get("DB_NAME", "rag_system"),
        "connect_timeout": 10
    }
    try: 
        return mysql.connector.connect(**db_config)
    except Exception as e: 
        print(f"Database connection failed: {e}") 
        return None

class MySQLLogHandler(logging.Handler):
    """Custom logging handler to send ERROR and CRITICAL logs to MySQL."""
    def __init__(self):
        super().__init__()
        # In Kubernetes, HOSTNAME is usually the pod name
        self.pod_name = os.environ.get("HOSTNAME", socket.gethostname())

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            log_msg = self.format(record) 
            try:
                conn = get_db_connection()
                if conn:
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
                    conn.close()
            except Exception as e:
                print(f"CRITICAL: Failed to write log to database: {e}")