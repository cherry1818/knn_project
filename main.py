import numpy as np
from collections import Counter

class MyKNN:
    def __init__(self, k=3): #k - liczba najblizszych sasiadów, na podstawie ktorej będzie podejmowana dezycja
        self.k = k

    def fit(self, X_train, y_train): #funckcja zapamiętuje dane treningowe x_train i odpowiadająće im etykiety y_train
        self.X_train = X_train
        self.y_train = y_train

    #Obliczenie odległości Euklidesowej między dwoma punktami.
    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    #Przewidywanie etykiety dla zestawu testowego.
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self._distance(x, x_train) for x_train in self.X_train] #Obliczamy odległości między danym punktem x a wszystkimi punktami w danych treningowych
            k_indices = np.argsort(distances)[:self.k] #Sortujemy te odległości i bierzemy indeksy k najbliższych punktów
            k_nearest_labels = [self.y_train[i] for i in k_indices] #Pobieramy etykiety tych punktów
            most_common = Counter(k_nearest_labels).most_common(1)[0][0] #Znajdujemy najczęściej występującą etykietę
            predictions.append(most_common)
        return np.array(predictions)
