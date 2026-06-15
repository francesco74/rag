import os
import requests
import html
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Carica le variabili dal .env
load_dotenv()

ENDPOINT = os.environ.get("REPWSS_ENDPOINT")
J2EE_USER = os.environ.get("REPWSS_J2EE_USERNAME")
J2EE_PASS = os.environ.get("REPWSS_J2EE_PASSWORD")
WS_USER = os.environ.get("REPWSS_USERNAME")
WS_PASS = os.environ.get("REPWSS_PASSWORD")

LOGON_CREDENTIALS = f"""<logon_credentials 
    j2eeusername="{J2EE_USER}" 
    j2eepassword="{J2EE_PASS}" 
    username="{WS_USER}" 
    password="{WS_PASS}" />"""

SOAP_PAYLOAD = f"""<soapenv:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:RepWSSGateway">
   <soapenv:Header/>
   <soapenv:Body>
      <urn:getAvailableDocumentClasses soapenv:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
         <logonCredentials xsi:type="xsd:string"><![CDATA[{LOGON_CREDENTIALS}]]></logonCredentials>
      </urn:getAvailableDocumentClasses>
   </soapenv:Body>
</soapenv:Envelope>"""

def main():
    print("Connessione al server per estrarre le tipologie di documenti...")
    
    # AGGIUNTO: SOAPAction vuota (spesso vitale per JBoss)
    headers = {
        'Content-Type': 'text/xml; charset=utf-8',
        'SOAPAction': '' 
    }
    
    resp = requests.post(ENDPOINT, data=SOAP_PAYLOAD.encode('utf-8'), headers=headers)
    
    # MODIFICA: Invece di fermarsi, stampa il motivo del 500
    if resp.status_code != 200:
        print(f"Errore HTTP {resp.status_code}!")
        print(f"Dettaglio errore dal server:\n{resp.text}")
        return

    root = ET.fromstring(resp.content)
    result_node = root.find(".//getAvailableDocumentClassesReturn")
    
    if result_node is None or not result_node.text:
        print("Nessuna classe restituita dal server.")
        return

    xml_decoded = html.unescape(result_node.text)
    
    try:
        classes_tree = ET.fromstring(xml_decoded)
    except ET.ParseError as e:
        print(f"Errore di parsing della risposta: {e}")
        return

    print("\n=== TIPOLOGIE DOCUMENTI DISPONIBILI ===")
    count = 0
    for doc_class in classes_tree:
        class_name = doc_class.attrib.get('name', 'Sconosciuto')
        class_desc = doc_class.attrib.get('description', 'Nessuna descrizione')
        print(f"- Nome in codice (da usare in filters.toml): '{class_name}'")
        print(f"  Descrizione a video: {class_desc}\n")
        count += 1
        
    print(f"Totale tipologie trovate: {count}")

if __name__ == "__main__":
    main()