"""
Script diagnostico: dump COMPLETO della risposta di LeggiAttoPlus.

Il metodo WSAttiSoapClient.leggi_atto_plus() (estrazione_documenti.py) fa il
parsing dell'XML interno ma tiene SOLO tre campi:

    oggetto = inner_tree.findtext(".//Oggetto")
    numero_atto = atto_el.findtext(".//Numero")   # solo dentro <Determina> o <Delibera>
    data_atto = atto_el.findtext(".//Data")

Tutto il resto della struttura XML (compresi eventuali campi che potrebbero
distinguere organo/sotto-tipo dell'atto, stato dell'iter, ecc.) viene
scartato. Questo script fa la STESSA chiamata SOAP ma stampa/salva l'intero
albero XML restituito, così puoi vedere anche i campi attualmente ignorati.

NOTE:
- Riusa client._soap_envelope_leggi_atto_plus() e client._post_soap() (stessi
  metodi "privati" già usati da leggi_atto_plus()), per essere certi di
  mandare esattamente la stessa richiesta che funziona in produzione.
- Il contenuto <Image> (base64 dei PDF/allegati) viene troncato nell'output a
  schermo per leggibilità, ma il file XML completo viene comunque salvato su
  disco intatto, se ti serve ispezionarlo per intero.

Uso:
    python debug_leggi_atto_plus_raw.py <uid>

Esempio (il caso decreto presidenziale 1/2024):
    python debug_leggi_atto_plus_raw.py 2119188
"""
import sys
import re
import xml.etree.ElementTree as ET

from common.estrazione_documenti import build_client_from_env


def redact_images(inner_tree: ET.Element) -> ET.Element:
    """Sostituisce il contenuto di ogni <Image> con un segnaposto leggibile,
    per non intasare l'output a schermo con blob base64 enormi. Il file XML
    completo (non redatto) viene comunque salvato separatamente su disco."""
    for allegato in inner_tree.findall(".//Allegati/Allegato"):
        image_el = allegato.find(".//Image")
        if image_el is not None and image_el.text:
            n_chars = len(image_el.text.strip())
            image_el.text = f"[[BASE64 OMESSO A SCHERMO: {n_chars} caratteri — vedi file XML completo salvato su disco]]"
    return inner_tree


def main():
    if len(sys.argv) != 2:
        print("Uso: python debug_leggi_atto_plus_raw.py <uid>")
        sys.exit(1)

    uid = sys.argv[1]

    print(f"Costruzione client Sicr@Web da env...")
    client = build_client_from_env()

    # Replica esatta di ciò che fa client.leggi_atto_plus(), fino al punto in
    # cui normalmente vengono estratti solo 3 campi: qui invece stampiamo/
    # salviamo tutto.
    filtro_xml = f"""<FiltroAttoIn>
  <IdDocumento>{uid}</IdDocumento>
  <DownloadAllegati>S</DownloadAllegati>
  <DownloadAllegatiSenzaFirma>N</DownloadAllegatiSenzaFirma>
</FiltroAttoIn>"""

    soap = client._soap_envelope_leggi_atto_plus(filtro_xml)

    print(f"Invio richiesta LeggiAttoPlus per UID {uid}...")
    resp_xml = client._post_soap(soap)

    raw_outer_path = f"leggi_atto_plus_RAW_outer_{uid}.xml"
    with open(raw_outer_path, "w", encoding="utf-8") as f:
        f.write(resp_xml)
    print(f"[✓] Risposta SOAP grezza (outer envelope) salvata in: {raw_outer_path}")

    tree = ET.fromstring(resp_xml)
    result_el = None
    for el in tree.iter():
        if el.tag.endswith("LeggiAttoPlusResult"):
            result_el = el
            break

    if result_el is None or not (result_el.text or "").strip():
        print("[!] Tag LeggiAttoPlusResult vuoto o non trovato. Guarda il file outer salvato sopra.")
        sys.exit(1)

    clean_inner = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result_el.text.strip())

    raw_inner_path = f"leggi_atto_plus_RAW_inner_{uid}.xml"
    with open(raw_inner_path, "w", encoding="utf-8") as f:
        f.write(clean_inner)
    print(f"[✓] XML interno COMPLETO (non redatto, con basi64 originali) salvato in: {raw_inner_path}")

    inner_tree = ET.fromstring(clean_inner)

    # Versione "leggibile a schermo": stessa struttura ma con le <Image>
    # sostituite da un placeholder, altrimenti lo scroll diventa impossibile.
    inner_tree_redatto = redact_images(inner_tree)

    try:
        ET.indent(inner_tree_redatto, space="  ")
    except AttributeError:
        # ET.indent esiste solo da Python 3.9+: se non disponibile, si stampa
        # comunque, solo meno "carino" da leggere.
        pass

    pretty_xml = ET.tostring(inner_tree_redatto, encoding="unicode")

    print("\n" + "="*100)
    print(f"STRUTTURA XML COMPLETA restituita da LeggiAttoPlus per UID {uid}")
    print("(Image sostituite da placeholder solo qui a schermo — il file salvato sopra le contiene per intero)")
    print("="*100)
    print(pretty_xml)
    print("="*100)

    # Elenco secco di tutti i TAG unici presenti, utile per farsi un'idea
    # veloce di quali campi esistono in totale nella struttura, a colpo d'occhio.
    tag_set = sorted({el.tag for el in inner_tree.iter()})
    print(f"\nElenco di tutti i tag distinti trovati nella risposta ({len(tag_set)}):")
    for t in tag_set:
        print(f"  - {t}")


if __name__ == "__main__":
    main()