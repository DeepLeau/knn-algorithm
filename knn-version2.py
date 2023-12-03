import numpy as np
import random
from sklearn.metrics import confusion_matrix, precision_score


def lire_fichier_texte(chemin_fichier):
    with open(chemin_fichier, 'r') as f:
        lignes = f.readlines()
        tableau = [ligne.strip().split(';') for ligne in lignes]
    tableau.pop()
    return tableau

data = lire_fichier_texte("C:/Users/Thomas/Desktop/IA/data.txt")
# Mélanger les données pour éliminer tout biais
random.seed(0)
random.shuffle(data)
# Diviser les données en deux parties
split_idx = int(0.5 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]
# Extraire les classes de l'ensemble d'entraînement et de test
train_classe = [row[-1] for row in train_data]
test_classe = [row[-1] for row in test_data]


def proche_voisins(test_row,k,train_data, train_classe):
    distances = {}
    for i in range(len(train_data)):
        inter = sum([(float(test_row[j]) - float(train_data[i][j])) ** 2 for j in range(len(test_row))])
        d = np.sqrt(inter)
        distances[i] = d
    distances_trie = dict(sorted(distances.items(), key=lambda item: item[1]))
    for j in range(len(distances_trie)-k):
        distances_trie.popitem()
    Classe = {}
    for key,value in distances_trie.items() :
        if train_data[key][7] not in Classe :
            Classe[train_data[key][7]] = 1
        else : 
            Classe[train_data[key][7]] += 1
    valeur_max = max(Classe.values())
    classe_finale = [cle for cle,valeurs in Classe.items() if valeurs == valeur_max][0]
    return classe_finale



def validation_croisee(data, k):
    # Extraire les vraies classes de l'ensemble de données
    vrai_classes = [row[-1] for row in data]
    # Effectuer l'apprentissage et évaluer les performances
    predictions = []
    for row in data:
        pred = proche_voisins(row[:-1], k, train_data, train_classe)
        predictions.append(pred)
    score = sum([1 for i in range(len(predictions)) if predictions[i] == vrai_classes[i]]) / float(len(predictions))
    return score

def trouver_k(data):
    # On prend des k impairs car ainsi on a toujours une majorité
    k_values = list(range(1, 30, 2))
    scores = []
    # Tester chaque valeur de k et stocker les scores
    for k in k_values:
        score = validation_croisee(data, k)
        scores.append(score)
    # Trouver la valeur de k qui donne le meilleur score de précision
    meilleur_k = k_values[scores.index(max(scores))]
    # Afficher le k optimal
    print("Le k optimal est :", meilleur_k)

def test_proche_voisins(data,k,i):
    # On prend le premier point de test
    test_row = test_data[i][:7]
    # On stocke la vraie classe du point de test
    true_classe = test_data[i][7]
    # On entraîne le modèle sur les données d'entraînement
    train_classe = [row[7] for row in train_data]
    # On prédit la classe du point de test en utilisant proche_voisins
    predicted_classe = proche_voisins(test_row, k, train_data, train_classe)
    # On affiche la prédiction et la vraie classe pour évaluer la performance
    print("Prédiction :", predicted_classe)
    print("Vraie classe :", true_classe)

def test():
    L = []
    for test_row in test_data:
        L.append(proche_voisins(test_row,1,train_data,train_classe))
    cm = confusion_matrix(test_classe,L)
    print(cm)
    precision = precision_score(test_classe, L, average=None)
    print(precision)