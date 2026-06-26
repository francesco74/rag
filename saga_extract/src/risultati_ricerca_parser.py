from __future__ import annotations
import html
import logging
import xml.etree.ElementTree as ET
import re

from config import settings
from soap_ricerca_builder import SoapDocWSRicercheBuilder, DocWSRicercheClient
from ricerca_filtri import RicercaFiltri

# Usa la validazione di livello globale ereditata dal config unificato
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
            import re
            
            # Sanitizzazione preventiva dei caratteri di controllo sull'intera risposta
            xml_string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', xml_string)
            outer_tree = ET.fromstring(xml_string)
            
            result_text = None
            for el in outer_tree.iter():
                if el.tag.endswith("RicercaDocumentiStringResult"):
                    result_text = el.text
                    break
            
            if result_text is None:
                tree = outer_tree
            else:
                if not result_text.strip():
                    return
                result_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result_text)
                tree = ET.fromstring(result_text)
            
            namespaces = {'ns': 'http://tempuri.org/'}
            
            # --- HELPER ROBUSTI: Cercano i tag ignorando il problema dei Namespace ---
            def get_node(parent, tag):
                return parent.find(f".//ns:{tag}", namespaces=namespaces) or parent.find(f".//{tag}")

            def get_nodes(parent, tag):
                nodes = parent.findall(f".//ns:{tag}", namespaces=namespaces)
                return nodes if nodes else parent.findall(f".//{tag}")

            def get_val(parent, tag):
                # FIX: Uso forzato della keyword 'namespaces' per bypassare l'argomento 'default'
                val = parent.findtext(f".//ns:{tag}", namespaces=namespaces)
                if val is None:
                    val = parent.findtext(f".//{tag}")
                return val
            # -------------------------------------------------------------------------
            
            # Controllo errori applicativi
            errore_nodo = get_node(tree, "Errore")
            if errore_nodo is not None and errore_nodo.text and errore_nodo.text.strip():
                self.errore = errore_nodo.text.strip()
                return

            # Navighiamo tutti i nodi in modo agnostico
            for doc in get_nodes(tree, "Documento"):
                id_doc = get_val(doc, "IdDocumento")
                
                if id_doc:
                    id_str = str(id_doc).strip()
                    self.ids.append(id_str)
                    
                    allegati_nomi = []
                    for allg_nodo in get_nodes(doc, "Allegato"):
                        nome_allg = get_val(allg_nodo, "NomeAllegato")
                        if nome_allg:
                            allegati_nomi.append(nome_allg.strip())
                    
                    # Popoliamo il dizionario sfruttando l'helper per codice pulito (DRY)
                    self.documenti_metadata[id_str] = {
                        "oggetto": get_val(doc, "Oggetto"),
                        "trattamento_descrizione": get_val(doc, "Trattamento_Descrizione"),
                        "proponente_descrizione": get_val(doc, "Proponente_Descrizione"),
                        "dirigente_descrizione": get_val(doc, "Dirigente_Descrizione"),
                        "data": get_val(doc, "Data"),
                        "anno": get_val(doc, "Anno"),
                        "numero": get_val(doc, "Numero"),
                        "data_esecutivita": get_val(doc, "DataEsecutivita"),
                        "data_pubblicazione": get_val(doc, "DataPubblicazione"),
                        "giorni_pubblicazione": get_val(doc, "GiorniPubblicazione"),
                        "allegati_nomi": allegati_nomi  
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
            cid=settings.ws_username,
            verify_tls=settings.verify_tls,
        )

        res = client.send_ricerca_documenti_string(soap_xml, soap_version=settings.soap_version)
        return ParsedRicercaRisultati(raw_xml_response=res.response_text)
        
    except Exception as e:
        log.error("Errore durante la richiesta SOAP: %s", str(e), exc_info=True)
        risultato_fallito = ParsedRicercaRisultati()
        risultato_fallito.errore = f"Errore SOAP: {str(e)}"
        return risultato_fallito