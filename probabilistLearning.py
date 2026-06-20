from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.ExactInference import VariableElimination
import pandas as pd
import joblib
import textwrap

if __name__ == "__main__":
    data = pd.read_csv('./datasets/Orderdef.csv')


    categorical_cols = ["mese", "città", "packType", "dim"]

    for col in categorical_cols:
        data[col] = data[col].astype(str).astype("category")









    #create Net


    G = DiscreteBayesianNetwork()

    G.add_node("mese")
    G.add_node("città")





    G.add_edge("mese","shipptime")
    G.add_edge("città","shipptime")


    G.add_edge("mese","shippcost")
    G.add_edge("dim","shippcost")
    G.add_edge("città","shippcost")
    G.add_edge("packType","shippcost")




    G.add_edge("packType","packCost")


    G.add_edge("mese","unitsSold")
    G.add_edge("unitPrice","unitsSold")
    G.add_edge("shippcost","unitsSold")
    G.add_edge("shipptime","unitsSold")


    G.add_edge("shipptime","rating")
    G.add_edge("shippcost","rating")
    G.add_edge("unitPrice","rating")



    G.add_edge("packCost","unitPrice")
    G.add_edge("shippcost","unitPrice")



    G.add_edge("mese","daysinstock")
    G.add_edge("unitPrice","daysinstock")
    G.add_edge("shipptime","daysinstock")
    G.add_edge("rating","daysinstock")


    #learn params



    state_names = {
    "mese": [
        "Gennaio", "Febbraio", "Marzo", "Aprile",
        "Maggio", "Giugno", "Luglio", "Agosto", 
        "Settembre", "Ottobre", "Novembre", "Dicembre"] 
    ,"daysinstock":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    ,"dim": ["p","m","g"]
    ,"packCost": [10.95, 20.9, 30.85, 40.8, 50.75, 60.7, 70.65, 80.6, 90.55, 100.5, 110.45, 120.4, 130.35, 140.3, 150.25, 160.2, 170.15,
                180.1, 190.05, 0.24, 0.27, 0.3, 0.34, 0.37, 0.41, 0.44, 0.48, 0.51, 0.55, 0.58, 0.62, 0.66, 0.69, 0.73, 0.76, 0.8, 0.83, 0.87]
    ,"shippcost":[5.35, 7.7, 10.05, 12.4, 14.75, 17.1, 19.45, 21.8, 24.15, 26.5, 28.85, 31.2, 33.55, 35.9, 38.25, 40.6, 42.95, 45.3, 47.65]
    ,"shipptime":[5.45, 9.9, 14.35, 18.8, 23.25, 27.7, 32.15, 36.6, 41.05, 45.5, 49.95, 54.4, 58.85, 63.3, 67.75, 72.2, 76.65, 81.1, 85.55]
    ,"unitsSold":[7.63, 14.27, 20.9, 27.53, 34.17, 40.8, 47.43, 54.07, 60.7, 67.33, 73.97, 80.6, 87.23, 93.87, 100.5, 107.13, 113.77, 120.4,
                127.03, 133.67, 140.3, 146.93, 153.57, 160.2, 166.83, 173.47, 180.1, 186.73, 193.37]
    ,"unitPrice": [12.1, 24.1, 36.09, 48.09, 60.09, 72.09, 84.09, 96.08, 108.08, 120.08, 132.08, 144.08, 156.07, 168.07, 180.07, 192.07, 204.07,
                216.06, 228.06, 240.06, 252.06, 264.06, 276.05, 288.05, 300.05, 312.05, 324.05, 336.04, 348.04, 360.04, 372.04, 384.04, 396.03,
                    408.03, 420.03, 432.03, 444.03, 456.02, 468.02, 480.02, 492.02, 504.02, 516.01, 528.01, 540.01, 552.01, 564.01, 576.0, 588.0, 0.14,
                    0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86]
    ,"rating": [1,2,3,4,5]
    ,"città":[
        "Roma", "Milano", "Firenze", "Napoli", "Venezia",
        "Tokyo", "Kyoto", "Osaka", "Hiroshima", "Sapporo",
        "Parigi", "Lione", "Marsiglia", "Bordeaux", "Nizza",
        "New York", "Los Angeles", "Chicago", "Miami", "San Francisco",
        "Madrid", "Barcellona", "Siviglia", "Valencia", "Bilbao"]
    ,"packType":["cartone","polistirolo","plastica","legno"] }




    G = G.fit(data,state_names=state_names)

    D = G.to_graphviz()

    # Plot the model.
    D.draw("Net.png", prog="dot")


    joblib.dump(G,"ReteBayes.pkl")






def pred(Net):

    state_names = {
        "mese": [
            "Gennaio", "Febbraio", "Marzo", "Aprile",
            "Maggio", "Giugno", "Luglio", "Agosto", 
            "Settembre", "Ottobre", "Novembre", "Dicembre"
        ],
        "daysinstock": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        "dim": ["p", "m", "g"],
        "packCost": [10.95, 20.9, 30.85, 40.8, 50.75, 60.7, 70.65, 80.6, 90.55, 100.5, 110.45, 120.4, 130.35, 140.3, 150.25, 160.2, 170.15,
                    180.1, 190.05, 0.24, 0.27, 0.3, 0.34, 0.37, 0.41, 0.44, 0.48, 0.51, 0.55, 0.58, 0.62, 0.66, 0.69, 0.73, 0.76, 0.8, 0.83, 0.87],
        "shippcost": [5.35, 7.7, 10.05, 12.4, 14.75, 17.1, 19.45, 21.8, 24.15, 26.5, 28.85, 31.2, 33.55, 35.9, 38.25, 40.6, 42.95, 45.3, 47.65],
        "shipptime": [5.45, 9.9, 14.35, 18.8, 23.25, 27.7, 32.15, 36.6, 41.05, 45.5, 49.95, 54.4, 58.85, 63.3, 67.75, 72.2, 76.65, 81.1, 85.55],
        "unitsSold": [7.63, 14.27, 20.9, 27.53, 34.17, 40.8, 47.43, 54.07, 60.7, 67.33, 73.97, 80.6, 87.23, 93.87, 100.5, 107.13, 113.77, 120.4,
                    127.03, 133.67, 140.3, 146.93, 153.57, 160.2, 166.83, 173.47, 180.1, 186.73, 193.37],
        "unitPrice": [12.1, 24.1, 36.09, 48.09, 60.09, 72.09, 84.09, 96.08, 108.08, 120.08, 132.08, 144.08, 156.07, 168.07, 180.07, 192.07, 204.07,
                    216.06, 228.06, 240.06, 252.06, 264.06, 276.05, 288.05, 300.05, 312.05, 324.05, 336.04, 348.04, 360.04, 372.04, 384.04, 396.03,
                    408.03, 420.03, 432.03, 444.03, 456.02, 468.02, 480.02, 492.02, 504.02, 516.01, 528.01, 540.01, 552.01, 564.01, 576.0, 588.0, 0.14,
                    0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86],
        "rating": [1, 2, 3, 4, 5],
        "città": [
            "Roma", "Milano", "Firenze", "Napoli", "Venezia",
            "Tokyo", "Kyoto", "Osaka", "Hiroshima", "Sapporo",
            "Parigi", "Lione", "Marsiglia", "Bordeaux", "Nizza",
            "New York", "Los Angeles", "Chicago", "Miami", "San Francisco",
            "Madrid", "Barcellona", "Siviglia", "Valencia", "Bilbao"
        ],
        "packType": ["cartone", "polistirolo", "plastica", "legno"]
    }

    features = [
        "mese", "dim", "packCost", "shippcost", "shipptime", 
        "unitsSold", "unitPrice", "rating", "città", "packType", "daysinstock"
    ]
    
    print("\n" + "="*80)
    print("--- Predizione Feature Mancante ---".center(80))
    print("Scegli la feature mancante che vuoi predire tra le seguenti:".center(80))
    print("="*80)
    
    for i, feat in enumerate(features, 1):
        valori = state_names.get(feat, [])
        valori_str = ", ".join(str(v) for v in valori)
        prefisso = f"{i:2}. {feat:<12} -> ["
        
        wrapper = textwrap.TextWrapper(
            width=80, 
            initial_indent=prefisso, 
            subsequent_indent=" " * len(prefisso)
        )
        print(wrapper.fill(valori_str) + "]")
        print("-" * 80)
        
    # Scelta della feature mancante (con loop di ri-richiesta in caso di errore)
    while True:
        try:
            scelta = int(input("\nInserisci il numero della feature da predire: "))
            if 1 <= scelta <= len(features):
                target_feature = features[scelta - 1]
                break
            print("[INPUT INVALIDO] Numero fuori range. Scegli un numero da 1 a 11.")
        except ValueError:
            print("[INPUT INVALIDO] Devi inserire un numero intero valido.")
            
    print(f"\nHai scelto di predire: '{target_feature}'")
    print("Adesso inserisci i valori per le altre feature.\n")
    
    # Raccolta dei dati per le altre feature
    input_data = {}
    for feat in features:
        if feat == target_feature:
            continue  # Salta la feature target
            
        valori_ammessi = state_names.get(feat, [])
        
        # Loop continuo: si interrompe SOLO quando l'input è valido ed è negli state_names
        while True:
            valore_raw = input(f"Inserisci il valore per '{feat}': ").strip()
            
            if not valore_raw:
                print(f"[INPUT INVALIDO] Il campo '{feat}' non può essere vuoto.")
                continue

            # Tentativo di conversione automatica del tipo per il matching corretto
            try:
                if "." in valore_raw:
                    valore_tipizzato = float(valore_raw)
                else:
                    valore_tipizzato = int(valore_raw)
            except ValueError:
                # Se fallisce la conversione numerica, viene trattato come stringa
                # Gestiamo il case-insensitive per mesi e città (es. "firenze" -> "Firenze")
                if feat in ["mese", "città"]:
                    valore_tipizzato = valore_raw.capitalize()
                else:
                    valore_tipizzato = valore_raw
            
            # Controllo di corrispondenza esatta
            if valore_tipizzato in valori_ammessi:
                input_data[feat] = valore_tipizzato
                break # Esce dal loop della feature corrente e passa alla successiva
            else:
                print(f"\n[INPUT INVALIDO] Il valore '{valore_raw}' non è consentito per la feature '{feat}'.")
                print(f"I valori accettati sono: {valori_ammessi}")
                print("Per favore, reinserisci il dato.\n")

    # Creazione del DataFrame
    pdata = pd.DataFrame([input_data])
    
    print("\nElaborazione della predizione in corso...")
    
    try:
        # Esecuzione della predizione usando VariableElimination
        risultato_df = Net.predict(pdata, stochastic=False, algo=VariableElimination)
        valore_predetto = risultato_df[target_feature].values[0]
        
        print("-" * 40)
        print(f"Il valore più probabile per '{target_feature}' è: {valore_predetto}")
        print("-" * 40)
        
        return valore_predetto
        
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        return None