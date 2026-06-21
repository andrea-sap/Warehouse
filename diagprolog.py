from pyswip import Prolog

# Risoluzione BUG logico: Stati incompatibili
MUTUAMENTE_ESCLUSIVI = {
    "direzione_nord": "direzione_sud",
    "direzione_sud": "direzione_nord",
    "modo_automatico": "modo_manuale",
    "modo_manuale": "modo_automatico"
}

OSSERVAZIONI_DISPONIBILI = {
    "1": ("direzione_nord", "Il selettore di direzione è impostato su NORD"),
    "2": ("direzione_sud", "Il selettore di direzione è impostato su SUD"),
    "3": ("modo_automatico", "La chiave di manutenzione è girata su AUTOMATICO"),
    "4": ("modo_manuale", "La chiave di manutenzione è girata su MANUALE"),
    "5": ("comando_computer", "Il computer (PLC) sta inviando il segnale di attivazione"),
    "6": ("nastro_fermo", "ANOMALIA: Il nastro trasportatore è fermo"),
    "7": ("spingitore_bloccato", "ANOMALIA: Il pistone spingitore di pacchi non si muove")
}

# =================================================================
# MOTORE LOGICO SEMPLIFICATO
# =================================================================
def build_warehouse_engine():
    prolog = Prolog()
    list(prolog.query("dynamic(regola/2)"))
    list(prolog.query("dynamic(assumibile/1)"))
    list(prolog.query("dynamic(risolvi/2)"))
    list(prolog.query("dynamic(trova_conflitto/1)"))

    list(prolog.query("retractall(regola(_, _))"))
    list(prolog.query("retractall(assumibile(_))"))
    list(prolog.query("retractall(risolvi(_, _))"))
    list(prolog.query("retractall(trova_conflitto(_))"))

    prolog.assertz("regola(abilitazione_nastro, [])")
    prolog.assertz("regola(abilitazione_spingitore, [])")

    # Topologia del circuito
    prolog.assertz("regola(alimentazione_motore, [live_w0])")
    prolog.assertz("regola(live_w0, [modo_automatico, ok_chiave_manutenzione, live_w1])")
    prolog.assertz("regola(live_w0, [modo_manuale, ok_chiave_manutenzione, live_w2])")
    prolog.assertz("regola(live_w1, [direzione_nord, ok_deviatore_flusso, live_w3])")
    prolog.assertz("regola(live_w2, [direzione_sud, ok_deviatore_flusso, live_w3])")
    
    prolog.assertz("regola(alimentazione_spingitore, [live_w4])")
    prolog.assertz("regola(live_w4, [comando_computer, ok_interruttore_computer, live_w3])")
    
    prolog.assertz("regola(linea_imballaggio, [live_w6])")
    prolog.assertz("regola(live_w3, [live_w5, ok_int_quadro_A])")
    prolog.assertz("regola(live_w6, [live_w5, ok_int_quadro_B])")
    prolog.assertz("regola(live_w5, [rete_elettrica_stabilimento])")
    
    prolog.assertz("regola(nastro_attivo, [abilitazione_nastro, alimentazione_motore, ok_motore])")
    prolog.assertz("regola(spingitore_attivo, [abilitazione_spingitore, alimentazione_spingitore, ok_valvola_aria])")

    # Vincoli di integrità
    prolog.assertz("regola(contraddizione, [nastro_fermo, nastro_attivo])")
    prolog.assertz("regola(contraddizione, [spingitore_bloccato, spingitore_attivo])")

    componenti = ["ok_int_quadro_A", "ok_int_quadro_B", "ok_deviatore_flusso", 
                  "ok_chiave_manutenzione", "ok_interruttore_computer", 
                  "ok_motore", "ok_valvola_aria", "rete_elettrica_stabilimento"]
    for comp in componenti:
        prolog.assertz(f"assumibile({comp})")

    # Meta-interprete standard pulito
    prolog.assertz("risolvi([], [])")
    prolog.assertz("risolvi([X|Resto], [X|AssResto]) :- assumibile(X), !, risolvi(Resto, AssResto)")
    prolog.assertz("risolvi([Obiettivo|Resto], Assunzioni) :- regola(Obiettivo, Corpo), risolvi(Corpo, AssCorpo), risolvi(Resto, AssResto), append(AssCorpo, AssResto, Assunzioni)")
    prolog.assertz("trova_conflitto(Conflitto) :- risolvi([contraddizione], Conflitto)")
    
    return prolog

# =================================================================
# ESTRAZIONE PULITA LATO PYTHON
# =================================================================
def stampa_solo_sospetti(conflitti_raw):
    if not conflitti_raw:
        print("\n NESSUN COMPONENTE SOSPETTO")
        print("(Nota: Se c'è un blocco, assicurati di aver inserito anche i sensori di contesto come direzione e modalità)")
        return

    print("\n COMPONENTI SOSPETTI RILEVATI:")
    componenti_sospetti = set()

    for soluzione in conflitti_raw:
        for componente in soluzione["Conflitto"]:
            comp_str = str(componente).strip("'\"")
            if comp_str.startswith("b'") or comp_str.startswith('b"'):
                comp_str = comp_str[2:-1]
            componenti_sospetti.add(comp_str)
            
    for comp in sorted(componenti_sospetti):
        print(f" - {comp}")

# =================================================================
# INTERFACCIA MANUALE SEQUENZIALE
# =================================================================
def prologdiag():
    prolog = build_warehouse_engine()

    while True:
        print("\n" + "="*65)
        print("         PANNELLO DI DIAGNOSTICA LINEA DI SMISTAMENTO            ")
        print("=======================================================")
        print(" [1] Inserisci Letture Sensori Manuali")
        print(" [0] Spegni Pannello")
        
        scelta_opzione = input("\nSeleziona un'opzione: ").strip()

        if scelta_opzione == "0":
            print("\nPannello diagnostico spento.")
            break
        elif scelta_opzione != "1":
            print(" Opzione non valida.")
            continue

        osservazioni = []
        print("\n--- Stato dei Sensori (Scegli i numeri, premi '0' per calcolare) ---")
        for chiave, (codice, descrizione) in OSSERVAZIONI_DISPONIBILI.items():
            print(f" [{chiave}] {descrizione}")
        print(" [0] Avvia Analisi Diagnostica")

        while True:
            scelta = input("Inserisci numero stato sensore: ").strip()
            if scelta == '0':
                break
            
            if scelta in OSSERVAZIONI_DISPONIBILI:
                codice_scelto = OSSERVAZIONI_DISPONIBILI[scelta][0]
                
                if codice_scelto in osservazioni:
                    print("  -> Già inserito.")
                    continue

                if codice_scelto in MUTUAMENTE_ESCLUSIVI:
                    opposto = MUTUAMENTE_ESCLUSIVI[codice_scelto]
                    if opposto in osservazioni:
                        osservazioni.remove(opposto)
                        print(f"  [!] Rimosso automaticamente lo stato opposto: {opposto}")

                osservazioni.append(codice_scelto)
                print(f"  -> Sensori attivi nella sessione: {osservazioni}")
            else:
                print(" Codice sensore errato.")

        # Caricamento nel database logico
        for oss in osservazioni:
            prolog.assertz(f"regola({oss}, [])")

        print("\nElaborazione dati topologici...")
        risultati = list(prolog.query("trova_conflitto(Conflitto)"))
        
        stampa_solo_sospetti(risultati)

        # FIX APPLICATO: Forza lo svuotamento immediato della memoria usando list()
        for oss in osservazioni:
            list(prolog.query(f"retractall(regola({oss}, []))"))

if __name__ == "__main__":
    prologdiag()
