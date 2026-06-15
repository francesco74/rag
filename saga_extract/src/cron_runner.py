import subprocess
import json
import sys
from datetime import datetime, timedelta

def main():
    # Calcolo del range: oggi e 3 giorni fa
    oggi = datetime.now()
    tre_giorni_fa = oggi - timedelta(days=3)

    # Formattazione obbligatoria per il Web Service Maggioli: DD/MM/YYYY
    data_inizio = tre_giorni_fa.strftime("%d/%m/%Y")
    data_fine = oggi.strftime("%d/%m/%Y")

    # Generazione dinamica del JSON dei filtri
    filtri = {
        "determina": {
            "adozione_data": data_inizio,
            "adozione_data_a": data_fine
        }
    }
    
    json_str = json.dumps(filtri)
    
    # Costruzione del comando per l'estrattore esistente
    comando = ["python", "estrattore.py", "--json-filters", json_str]
    
    # Se il cronrunner riceve flag come --debug o --dry, li passa a cascata
    if "--debug" in sys.argv:
        comando.append("--debug")
    if "--dry" in sys.argv:
        comando.append("--dry")

    # Esecuzione atomica del processo
    subprocess.run(comando, check=True)

if __name__ == "__main__":
    main()