from __future__ import annotations

import os
import html
import logging
import xml.etree.ElementTree as ET

from config import settings
from soap_ricerca_builder import SoapDocWSRicercheBuilder, DocWSRicercheClient
from ricerca_filtri import RicercaFiltri

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level_str, logging.INFO),
    format='%(asctime)s - PARSER - %(levelname)s - %(message)s',
    force=True
)
log = logging.getLogger(__name__)

class ParsedRicercaRisultati:
    """Rappresenta e analizza la risposta SOAP di Maggioli Sicr@Web."""
    def __init__(self, raw_xml_response: str = "", is_dry_run: bool = False):
        self.errore = None
        self.ids = []
        self.documenti_metadata = {} 
        
        if is_dry_run or not raw_xml_response:
            return
            
        self._parse(raw_xml_response)

    def _parse(self, xml_string: str):
        try:
            # Sbroglia l'XML escapato dal server
            clean_xml = html.unescape(xml_string)
            tree = ET.fromstring(clean_xml)
            
            # Namespace standard di tempuri.org del WS Maggioli
            namespaces = {'ns': 'http://tempuri.org/'}
            
            # Controllo errori applicativi
            errore_nodo = tree.find(".//ns:Errore", namespaces)
            if errore_nodo is not None and errore_nodo.text and errore_nodo.text.strip():
                self.errore = errore_nodo.text.strip()
                return

            # Navighiamo tutti i nodi <Documento>
            for doc in tree.findall(".//ns:Documento", namespaces):
                id_doc = doc.findtext(".//ns:IdDocumento", namespaces=namespaces)
                
                if id_doc:
                    id_str = str(id_doc).strip()
                    self.ids.append(id_str)
                    
                    # Estrazione posizionale dei nomi degli allegati dichiarati nell'atto
                    allegati_nomi = []
                    for allg_nodo in doc.findall(".//ns:Allegato", namespaces):
                        nome_allg = allg_nodo.findtext("ns:NomeAllegato", namespaces=namespaces)
                        if nome_allg:
                            allegati_nomi.append(nome_allg.strip())
                    
                    # Popoliamo il dizionario strutturato dei metadati
                    self.documenti_metadata[id_str] = {
                        "oggetto": doc.findtext(".//ns:Oggetto", namespaces=namespaces),
                        "trattamento_descrizione": doc.findtext(".//ns:Trattamento_Descrizione", namespaces=namespaces),
                        "proponente_descrizione": doc.findtext(".//ns:Proponente_Descrizione", namespaces=namespaces),
                        "dirigente_descrizione": doc.findtext(".//ns:Dirigente_Descrizione", namespaces=namespaces),
                        "data": doc.findtext(".//ns:Data", namespaces=namespaces),
                        "anno": doc.findtext(".//ns:Anno", namespaces=namespaces),
                        "numero": doc.findtext(".//ns:Numero", namespaces=namespaces),
                        "data_esecutivita": doc.findtext(".//ns:DataEsecutivita", namespaces=namespaces),
                        "data_pubblicazione": doc.findtext(".//ns:DataPubblicazione", namespaces=namespaces),
                        "giorni_pubblicazione": doc.findtext(".//ns:GiorniPubblicazione", namespaces=namespaces),
                        "allegati_nomi": allegati_nomi  # Collezione dei nomi mantenendo l'ordine SOAP
                    }
                    
            log.debug("Parsing completato con successo. Atti validati: %d", len(self.ids))
                    
        except Exception as e:
            self.errore = f"Errore nel parsing XML: {str(e)}"
            log.error("Fallimento durante l'esecuzione di _parse XML: %s", str(e), exc_info=True)

def run_search(filtri: RicercaFiltri, dry_run: bool = False) -> ParsedRicercaRisultati:
    """Esegue la ricerca passandogli l'oggetto python RicercaFiltri configurato."""
    builder = SoapDocWSRicercheBuilder(ricerca_filtri=filtri)
    soap_xml = builder.to_soap_envelope(
        codice_amministrazione=settings.docws_codice_amministrazione,
        codice_aoo=settings.docws_codice_aoo,
        pretty=True,
    )

    if dry_run:
        log.info("=== MODALITÀ DRY RUN ATTIVA ===")
        return ParsedRicercaRisultati(is_dry_run=True)

    log.debug("Generated SOAP Request:\n%s", soap_xml)
    
    try:
        client = DocWSRicercheClient(
            endpoint_url=settings.docws_ricerca_endpoint,
            cid=settings.docws_cid,
            timeout=settings.http_timeout_seconds,
            verify_tls=settings.verify_tls,
        )

        res = client.send_ricerca_documenti_string(soap_xml, soap_version=settings.soap_version)
        return ParsedRicercaRisultati(raw_xml_response=res.response_text)
        
    except Exception as e:
        log.error("Errore durante la richiesta SOAP: %s", str(e), exc_info=True)
        risultato_fallito = ParsedRicercaRisultati()
        risultato_fallito.errore = f"Errore SOAP: {str(e)}"
        return risultato_fallito