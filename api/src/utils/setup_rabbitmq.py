import logging
import os
import sys
import time
import requests
from requests.auth import HTTPBasicAuth
from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool

# ==============================================================================
# CONFIGURAZIONE LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SETUP_MQ - %(levelname)s - %(message)s'
)
log = logging.getLogger("setup_rabbitmq")


# ==============================================================================
# TOPOLOGIA — FONTE UNICA DI VERITÀ
# Modificare solo qui se si aggiungono code o exchange.
# ==============================================================================

TOPOLOGY = {
    "exchanges": [
        {
            "name": "rag_dlx",
            "type": "direct",
            "durable": True,
            "auto_delete": False,
            "arguments": {},
            "description": "Dead Letter Exchange — riceve i messaggi rejected"
        }
    ],
    "queues": [
        # --- Dead Letter Queues (Quorum Queues per Alta Affidabilità) ---
        {
            "name": "da-convertire.dlq",
            "durable": True,
            "auto_delete": False,
            "arguments": {
                "x-queue-type": "quorum"  # <-- Forza la replica sui 3 nodi K8s
            },
            "description": "DLQ Quorum per messaggi falliti di da-convertire"
        },
        {
            "name": "da-indicizzare.dlq",
            "durable": True,
            "auto_delete": False,
            "arguments": {
                "x-queue-type": "quorum"  # <-- Forza la replica sui 3 nodi K8s
            },
            "description": "DLQ Quorum per messaggi falliti di da-indicizzare"
        },
        # --- Code Operative (Quorum Queues + DLX) ---
        {
            "name": "da-convertire",
            "durable": True,
            "auto_delete": False,
            "arguments": {
                "x-dead-letter-exchange": "rag_dlx",
                "x-dead-letter-routing-key": "da-convertire",
                "x-queue-type": "quorum"  # <-- Trasforma la coda in Quorum Queue
            },
            "description": "Coda input converter — HA Quorum + Routing su DLQ"
        },
        {
            "name": "da-indicizzare",
            "durable": True,
            "auto_delete": False,
            "arguments": {
                "x-dead-letter-exchange": "rag_dlx",
                "x-dead-letter-routing-key": "da-indicizzare",
                "x-queue-type": "quorum"  # <-- Trasforma la coda in Quorum Queue
            },
            "description": "Coda input ingest — HA Quorum + Routing su DLQ"
        },
    ],
    "bindings": [
        {
            "source_exchange": "rag_dlx",
            "destination_queue": "da-convertire.dlq",
            "routing_key": "da-convertire",
            "description": "rag_dlx --[da-convertire]--> da-convertire.dlq"
        },
        {
            "source_exchange": "rag_dlx",
            "destination_queue": "da-indicizzare.dlq",
            "routing_key": "da-indicizzare",
            "description": "rag_dlx --[da-indicizzare]--> da-indicizzare.dlq"
        },
    ]
}


# ==============================================================================
# CLIENT API
# ==============================================================================

class RabbitMQClient:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.base_url = f"http://{host}:{port}/api"
        self.auth = HTTPBasicAuth(user, password)
        # Il vhost è impostato rigidamente su "/" ed encodato correttamente per le API (%2F)
        self.vhost_encoded = requests.utils.quote("/", safe="")
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({"Content-Type": "application/json"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def wait_until_ready(self, max_attempts: int = 30, delay: float = 3.0) -> bool:
        """Attende che RabbitMQ sia raggiungibile e pronto."""
        log.info("Attendo che RabbitMQ sia raggiungibile...")
        for attempt in range(1, max_attempts + 1):
            try:
                r = self.session.get(self._url("/overview"), timeout=5)
                if r.status_code == 200:
                    log.info("RabbitMQ raggiungibile.")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            log.warning("  Tentativo %d/%d... (retry tra %.0fs)", attempt, max_attempts, delay)
            time.sleep(delay)
        log.error("RabbitMQ non risponde dopo %d tentativi.", max_attempts)
        return False

    def declare_exchange(self, name: str, ex_type: str, durable: bool,
                         auto_delete: bool, arguments: dict) -> bool:
        url = self._url(f"/exchanges/{self.vhost_encoded}/{requests.utils.quote(name, safe='')}")
        payload = {
            "type": ex_type,
            "durable": durable,
            "auto_delete": auto_delete,
            "arguments": arguments
        }
        r = self.session.put(url, json=payload, timeout=10)
        return r.status_code in (201, 204)

    def declare_queue(self, name: str, durable: bool,
                      auto_delete: bool, arguments: dict) -> bool:
        url = self._url(f"/queues/{self.vhost_encoded}/{requests.utils.quote(name, safe='')}")
        payload = {
            "durable": durable,
            "auto_delete": auto_delete,
            "arguments": arguments
        }
        r = self.session.put(url, json=payload, timeout=10)
        return r.status_code in (201, 204)

    def declare_binding(self, source_exchange: str, destination_queue: str,
                        routing_key: str) -> bool:
        url = self._url(
            f"/bindings/{self.vhost_encoded}"
            f"/e/{requests.utils.quote(source_exchange, safe='')}"
            f"/q/{requests.utils.quote(destination_queue, safe='')}"
        )
        payload = {"routing_key": routing_key, "arguments": {}}
        r = self.session.post(url, json=payload, timeout=10)
        return r.status_code in (201, 204)

    def queue_exists(self, name: str) -> bool:
        url = self._url(f"/queues/{self.vhost_encoded}/{requests.utils.quote(name, safe='')}")
        r = self.session.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("name") == name
        return False

    def exchange_exists(self, name: str) -> bool:
        url = self._url(f"/exchanges/{self.vhost_encoded}/{requests.utils.quote(name, safe='')}")
        r = self.session.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("name") == name
        return False


# ==============================================================================
# OPERAZIONI PRINCIPALI
# ==============================================================================

def apply_topology(client: RabbitMQClient) -> bool:
    """Applica la topologia completa. Ritorna True se tutto è andato a buon fine."""
    success = True

    log.info("--- Exchange ---")
    for ex in TOPOLOGY["exchanges"]:
        ok = client.declare_exchange(
            name=ex["name"],
            ex_type=ex["type"],
            durable=ex["durable"],
            auto_delete=ex["auto_delete"],
            arguments=ex["arguments"]
        )
        if ok:
            log.info("  [OK]  %s", ex["description"])
        else:
            log.error("  [ERR] %s", ex["description"])
            success = False

    log.info("--- Code ---")
    for q in TOPOLOGY["queues"]:
        ok = client.declare_queue(
            name=q["name"],
            durable=q["durable"],
            auto_delete=q["auto_delete"],
            arguments=q["arguments"]
        )
        if ok:
            log.info("  [OK]  %s", q["description"])
        else:
            log.error("  [ERR] %s", q["description"])
            success = False

    log.info("--- Binding ---")
    for b in TOPOLOGY["bindings"]:
        ok = client.declare_binding(
            source_exchange=b["source_exchange"],
            destination_queue=b["destination_queue"],
            routing_key=b["routing_key"]
        )
        if ok:
            log.info("  [OK]  %s", b["description"])
        else:
            log.error("  [ERR] %s", b["description"])
            success = False

    return success


def verify_topology(client: RabbitMQClient) -> bool:
    """Verifica che tutti gli elementi della topologia esistano sul broker."""
    log.info("--- Verifica Topologia ---")
    success = True

    for ex in TOPOLOGY["exchanges"]:
        exists = client.exchange_exists(ex["name"])
        if exists:
            log.info("  [OK]  Exchange: %s", ex["name"])
        else:
            log.error("  [ERR] Exchange mancante: %s", ex["name"])
            success = False

    for q in TOPOLOGY["queues"]:
        exists = client.queue_exists(q["name"])
        if exists:
            log.info("  [OK]  Coda: %s", q["name"])
        else:
            log.error("  [ERR] Coda mancante: %s", q["name"])
            success = False

    return success


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    max_attempts = settings.broker_max_attempts

    log.info("======================================================")
    log.info(" RabbitMQ Topology Setup")
    log.info(" Host   : %s:%d", settings.broker_host, settings.broker_port)
    log.info(" VHost  : /")
    log.info("======================================================")

    
    client = RabbitMQClient(
        host=settings.broker_host,
        port=settings.broker_port,
        user=settings.broker_username,
        password=settings.broker_password
    )

    if not client.wait_until_ready(max_attempts=max_attempts):
        sys.exit(1)

    log.info("Applicazione topologia...")
    ok = apply_topology(client)
    if not ok:
        log.error("Errore durante l'applicazione della topologia.")
        sys.exit(1)

    log.info("Verifica finale...")
    ok = verify_topology(client)

    if ok:
        log.info("======================================================")
        log.info(" Topologia completata con successo.")
        log.info(" I servizi possono essere avviati in qualsiasi ordine.")
        log.info("======================================================")
        sys.exit(0)
    else:
        log.error("Verifica fallita. Controllare i log sopra.")
        sys.exit(1)


if __name__ == "__main__":
    main()
