import math
from astar import AStar

# 1. Definizione delle coordinate delle città (Latitudine, Longitudine approssimate)
# Usate dall'euristica per calcolare la distanza euclidea
COORDINATE = {
    # Italia
    "Roma": (41.90, 12.49), "Milano": (45.46, 9.19), "Firenze": (43.77, 11.25),
    "Napoli": (40.85, 14.26), "Venezia": (45.44, 12.31), "Bologna": (44.49, 11.34), # Intermedia
    
    # Francia
    "Parigi": (48.85, 2.35), "Lione": (45.76, 4.83), "Marsiglia": (43.29, 5.37),
    "Bordeaux": (44.83, -0.57), "Nizza": (43.70, 7.26),
    
    # Spagna
    "Madrid": (40.41, -3.70), "Barcellona": (41.38, 2.17), "Siviglia": (37.38, -5.98),
    "Valencia": (39.46, -0.37), "Bilbao": (43.26, -2.93), "Saragozza": (41.64, -0.88), # Intermedia
    
    # USA
    "New York": (40.71, -74.00), "Chicago": (41.87, -87.62), "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24), "San Francisco": (37.77, -122.41),
    "Denver": (39.73, -104.99), # Intermedia
    
    # Giappone
    "Tokyo": (35.67, 139.65), "Kyoto": (35.01, 135.76), "Osaka": (34.69, 135.50),
    "Hiroshima": (34.38, 132.45), "Sapporo": (43.06, 141.35),
    "Nagoya": (35.18, 136.90) # Intermedia
}

# 2. Definizione del Grafo delle connessioni reali (Strade o Rotte Aeree principali)
# Ogni arco ha un costo associato (es. distanza in km o tempo di percorrenza)
GRAFO = {
    # Connessioni Italia
    "Napoli": {"Roma": 220},
    "Roma": {"Napoli": 220, "Firenze": 270},
    "Firenze": {"Roma": 270, "Bologna": 100},
    "Bologna": {"Firenze": 100, "Milano": 210, "Venezia": 155},
    "Venezia": {"Bologna": 155, "Milano": 260},
    "Milano": {"Bologna": 210, "Venezia": 260, "Lione": 440, "Parigi": 850, "Nizza": 320},

    # Connessioni Francia
    "Nizza": {"Milano": 320, "Marsiglia": 200},
    "Marsiglia": {"Nizza": 200, "Lione": 310, "Barcellona": 500},
    "Lione": {"Milano": 440, "Marsiglia": 310, "Parigi": 460},
    "Bordeaux": {"Parigi": 580, "Bilbao": 340},
    "Parigi": {"Milano": 850, "Lione": 460, "Bordeaux": 580, "New York": 5800, "Tokyo": 9700},

    # Connessioni Spagna
    "Barcellona": {"Marsiglia": 500, "Saragozza": 300, "Valencia": 350},
    "Bilbao": {"Bordeaux": 340, "Saragozza": 300, "Madrid": 400},
    "Saragozza": {"Barcellona": 300, "Bilbao": 300, "Madrid": 320},
    "Valencia": {"Barcellona": 350, "Madrid": 360, "Siviglia": 650},
    "Madrid": {"Bilbao": 400, "Saragozza": 320, "Valencia": 360, "Siviglia": 530, "New York": 5700},
    "Siviglia": {"Madrid": 530, "Valencia": 650},

    # Connessioni USA
    "New York": {"Parigi": 5800, "Madrid": 5700, "Chicago": 1100, "Miami": 1750},
    "Chicago": {"New York": 1100, "Denver": 1600, "Miami": 1900},
    "Miami": {"New York": 1750, "Chicago": 1900, "Los Angeles": 3700},
    "Denver": {"Chicago": 1600, "Los Angeles": 1300, "San Francisco": 1500},
    "Los Angeles": {"Miami": 3700, "Denver": 1300, "San Francisco": 600, "Tokyo": 8800},
    "San Francisco": {"Denver": 1500, "Los Angeles": 600, "Tokyo": 8200},

    # Connessioni Giappone
    "Tokyo": {"Parigi": 9700, "Los Angeles": 8800, "San Francisco": 8200, "Sapporo": 830, "Nagoya": 350},
    "Sapporo": {"Tokyo": 830},
    "Nagoya": {"Tokyo": 350, "Kyoto": 130},
    "Kyoto": {"Nagoya": 130, "Osaka": 55},
    "Osaka": {"Kyoto": 55, "Hiroshima": 330},
    "Hiroshima": {"Osaka": 330}
}

# 3. Implementazione della classe A* estendendo quella della libreria
class CorriereRouter(AStar):
    def __init__(self, grafo, coordinate):
        self.grafo = grafo
        self.coordinate = coordinate

    def heuristic_cost_estimate(self, n1, n2):
        """Calcola la distanza euclidea tra due città come stima euristica"""
        lat1, lon1 = self.coordinate[n1]
        lat2, lon2 = self.coordinate[n2]
        # Formula della distanza euclidea standard (moltiplicata per un fattore di scala approssimativo per i km)
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111

    def distance_between(self, n1, n2):
        """Ritorna il costo reale del collegamento tra n1 e n2"""
        return self.grafo[n1].get(n2, float('inf'))

    def neighbors(self, node):
        """Ritorna le città vicine raggiungibili da quella attuale"""
        return list(self.grafo[node].keys())

# 4. Esecuzione del test

citta = [
    "Roma", "Milano", "Firenze", "Napoli", "Venezia", "Bologna",
    "Parigi", "Lione", "Marsiglia", "Bordeaux", "Nizza",
    "Madrid", "Barcellona", "Siviglia", "Valencia", "Bilbao", "Saragozza",
    "New York", "Chicago", "Miami", "Los Angeles", "San Francisco", "Denver",
    "Tokyo", "Kyoto", "Osaka", "Hiroshima", "Sapporo", "Nagoya"
]

def pathstar():
    router = CorriereRouter(GRAFO, COORDINATE)

    # Definiamo la partenza e la destinazione del corriere
    valid = False
    print(f"\n\nPunti transito abilitati: {citta} \n\n")
    while(not valid):
        partenza= input(f"\nScegli luogo di partenza : \n").strip()
        destinazione = input(f"\nScegli luogo di destinazione : \n").strip()
        if partenza in citta and destinazione in citta:
            valid = True
        else:
            print(f"Partenza o destinazione errate \n")
    

    print(f"Calcolo del percorso più breve da {partenza} a {destinazione}...\n")

    # Eseguiamo l'algoritmo A*
    percorso_generato = router.astar(partenza, destinazione)

    if percorso_generato:
        percorso = list(percorso_generato)
        print("Percorso ottimale trovato:")
        print(" -> ".join(percorso))
        
        # Calcolo del costo totale del viaggio
        costo_totale = 0
        for i in range(len(percorso) - 1):
            costo_totale += GRAFO[percorso[i]][percorso[i+1]]
        print(f"\nDistanza totale stimata del viaggio: {costo_totale} km")
    else:
        print("Spiacente, nessun percorso disponibile tra le due città selezionate.")


if __name__ == "__main__":
    pathstar()