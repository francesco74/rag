"""
Elenco delle collection di Qdrant, con punti, dimensione dei vettori e quantizzazione.

    python -m utils.qdrant_collections_list
"""

from qdrant_client import QdrantClient

from common.config import settings


def main():
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    names = sorted(c.name for c in client.get_collections().collections)
    if not names:
        print("Nessuna collection.")
        return
    print(f"{'collection':34} {'punti':>10} {'dimensione':>11}  vettori su disco  quantizzazione")
    for name in names:
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", "con nome")
        on_disk = "sì" if getattr(vectors, "on_disk", False) else "no"
        quant = "int8 in RAM" if info.config.quantization_config else "nessuna"
        print(f"{name:34} {info.points_count or 0:>10} {size!s:>11}  {on_disk:16}  {quant}")


if __name__ == "__main__":
    main()
