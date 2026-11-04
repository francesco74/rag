"""Tempi delle fasi di una richiesta della chat e riga di riepilogo nel log."""

import time


class PhaseTimer:
    """
    Misura le fasi del task e compone la riga [SUMMARY].

    Si usa avvolgendo le chiamate: `phases.run("ricerca", retrieve_chunks, ...)`.
    Le fasi ripetute nei tentativi successivi si sommano. `info` raccoglie i
    dati da riportare oltre ai tempi (esito, tentativi, parent, periodo).
    """

    # Ordine di stampa; eventuali altre fasi seguono in ordine alfabetico.
    ORDER = ("concetti", "riscrittura", "embedding", "cache", "ricerca", "generazione", "grader")

    def __init__(self):
        self.started = time.perf_counter()
        self.durations = {}
        self.retrieval = {"t_search_s": 0.0, "t_rerank_s": 0.0}
        self.info = {}

    def run(self, phase, fn, *args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.durations[phase] = self.durations.get(phase, 0.0) + time.perf_counter() - started

    def note_retrieval(self, stats):
        """Somma i tempi interni di retrieve_chunks (ricerca e rerank) e annota i parent."""
        stats = stats or {}
        for key in self.retrieval:
            self.retrieval[key] += float(stats.get(key) or 0.0)
        if stats.get("n_parents_final") is not None:
            self.info["parent"] = stats["n_parents_final"]

    def summary(self, outcome):
        total = time.perf_counter() - self.started
        names = [n for n in self.ORDER if n in self.durations]
        names += sorted(set(self.durations) - set(self.ORDER))
        parts = []
        for name in names:
            text = f"{name} {self.durations[name]:.1f}s"
            if name == "ricerca" and any(self.retrieval.values()):
                text += (f" (vettori+testo {self.retrieval['t_search_s']:.1f}s, "
                         f"rerank {self.retrieval['t_rerank_s']:.1f}s)")
            parts.append(text)
        other = total - sum(self.durations.values())
        if other >= 0.05:
            parts.append(f"altro {other:.1f}s")
        head = f"esito={outcome}"
        for key in ("tentativi", "parent", "periodo"):
            if self.info.get(key) is not None:
                head += f" {key}={self.info[key]}"
        return f"{head} totale={total:.1f}s | " + " · ".join(parts)
