from probabilistLearning import pred
from diagprolog import prologdiag
from consegnestar import pathstar
from supervisedLearning import profila
import joblib
import sys







# Warehouse AI ASCII Art
def main():
# Warehouse AI ASCII Art - Versione Corretta (Raw Strings)

    print(r" _    _                      _                      ___  _____ ")
    print(r"| |  | |                    | |                    / _ \|_   _|")
    print(r"| |  | | __ _ _ __ ___  __ _| |__   ___  _   _  ___| /_\ \ | |  ")
    print(r"| |/\| |/ _` | '__/ _ \/ _` | '_ \ / _ \| | | |/ __|  _  | | |  ")
    print(r"\  /\  / (_| | | |  __/ (_| | | | | (_) | |_| |\__ \ | | |_| |_ ")
    print(r" \/  \/ \__,_|_|  \___|\__,_|_| |_|\___/ \__,_||___/\_| |_/\___/ ")
    print(r"                                                                   ")
    print(r"                   _______________________                         ")
    print(r"                  /|_|_|_|_|_|_|_|_|_|_|_/|                        ")
    print(r"                 /_|_|_|_|_|_|_|_|_|_|_/_/|                        ")
    print(r"                /_|_|_|_|_|_|_|_|_|_|_/_/|                         ")
    print(r"               /_|_|_|_|_|_|_|_|_|_|_/_/|                          ")
    print(r"              /_|_|_|_|_|_|_|_|_|_|_/_/|                           ")
    print(r"             /_|_|_|_|_|_|_|_|_|_|_/_/|                            ")
    print(r"            /______________________/_/|                            ")
    print(r"            |______________________|/|                             ")
    print(r"            |      _____           |/|                             ")
    print(r"            |     / ____|   /\     |/|     [___]                   ")
    print(r"            |    | (___    /  \    |/|    /[___]\                  ")
    print(r"            |     \___ \  / /\ \   |/|   /_|___|_\                 ")
    print(r"            |     ____) |/ ____ \  |/|  |/_|   |_\                 ")
    print(r"            |    |_____//_/    \_\ |/|  |_________|                ")
    print(r"            |______________________|/                              ")
    print(r"                                                                   ")
    print("Caricamento Modelli IA in corso...")
    

    columns = ['age', 'income', 'spending_score', 'membership_years',
       'purchase_frequency', 'last_purchase_amount', 'gender_Female',
       'gender_Male', 'gender_Other', 'preferred_category_Clothing',
       'preferred_category_Electronics', 'preferred_category_Groceries',
       'preferred_category_Home & Garden', 'preferred_category_Sports']

    try:
        # Caricamento tramite pickle/joblib dei tuoi asset
        Net = joblib.load("ReteBayes.pkl")
        model = joblib.load("RandomForest.pkl")
        print("Modelli caricati con successo!\n")
    except FileNotFoundError as e:
        print(f"[ATTENZIONE] Uno o più file di modello non sono stati trovati: {e}")
        print("Assicurati di aver salvato 'modeltwo.pkl', 'ReteBayes.pkl', 'scaler.pkl' e 'pca.pkl' nella cartella di esecuzione.")
        sys.exit(1)



    run = True

    while(run):
        print("═"*40)
        print("     SISTEMA DI GESTIONE AZIENDALE IA     ")
        print("═"*40)
        print("1. Consulenza AI per ordini")
        print("2. Cerca percorso più breve consegna merce")
        print("3. Esegui Diagnostica magazzino")
        print("4. Profila Utente")
        print("5. Esci")
        print("─"*40)
        opt = int(input(f"\n\nSelezionare Task:"))
        if opt == 1:
            pred(Net)
        elif opt == 2:
            pathstar()
        elif opt == 3:
            prologdiag()
        elif opt == 4:
            profila(model,columns)
        elif opt == 5:
            run= False

        else:
            print("\n[Scelta non valida] Riprova.\n")
        

if __name__ == "__main__":
      main()