# Warehouse
ICON25-26

## Requisiti e Installazione

Segui questi passaggi per configurare l'ambiente di sviluppo locale. La guida assume che tu stia utilizzando un sistema basato su Linux.

### 1. Creazione e Attivazione del Virtual Environment

Per prima cosa, crea l'ambiente virtuale Python (venv) e installa le dipendenze necessarie a partire dal file `requirements.txt`.

```bash
# Crea il virtual environment (chiamato 'venv')
python3 -m venv venv

# Attiva il virtual environment
source venv/bin/activate

# Aggiorna pip e installa i requisiti
pip install --upgrade pip
pip install -r requirements.txt
```
#Installazione SWI-Prolog
```bash
# 1. Clonazione della repository ufficiale
git clone [https://github.com/SWI-Prolog/swipl-devel.git](https://github.com/SWI-Prolog/swipl-devel.git)
cd swipl-devel

# 2. Creazione della directory di build
mkdir build && cd build

# 3. Configurazione di CMake puntando al Virtual Env ed escludendo la documentazione
cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV -DINSTALL_DOCUMENTATION=OFF ..

# 4. Compilazione e installazione
make -j$(nproc)
make install
```

Aprire il file venv/bin/activate e copiare in fondo le seguenti righe

### --- CONFIGURAZIONE LOCALE SWI-PROLOG PER PYSWIP ---
export SWI_HOME_DIR="$VIRTUAL_ENV/lib/swipl"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/swipl/lib/x86_64-linux:$LD_LIBRARY_PATH"

Una volta terminato , disattiva e riattiva il venv
```bash
deactivate
source venv/bin/activate
```
