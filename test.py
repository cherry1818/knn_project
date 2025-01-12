import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from main import MyKNN

#Testowanie algorytmu KNN na zbiorze Iris.
def test_my_knn():
    # Wczytanie zbioru danych Iris
    data = load_iris()
    X, y = data.data, data.target

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tworzenie i testowanie własnego KNN
    my_knn = MyKNN(k=3)
    my_knn.fit(X_train, y_train)
    y_pred = my_knn.predict(X_test)

    # Ocena dokładności
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność własnego algorytmu KNN: {accuracy:.2f}")

if __name__ == "__main__":
    test_my_knn()
