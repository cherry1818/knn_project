from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from main import MyKNN

def compare_knn():
    # Wczytanie zbioru danych Iris
    data = load_iris()
    X, y = data.data, data.target

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Własna implementacja KNN
    my_knn = MyKNN(k=3)
    my_knn.fit(X_train, y_train)
    y_pred_my_knn = my_knn.predict(X_test)
    accuracy_my_knn = accuracy_score(y_test, y_pred_my_knn)
    print(f"Dokładność własnego KNN: {accuracy_my_knn:.2f}")

    # Implementacja KNN z biblioteki scikit-learn
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Dokładność KNN z biblioteki scikit-learn: {accuracy_sklearn:.2f}")

    # Porównanie wyników
    if accuracy_my_knn == accuracy_sklearn:
        print("Implementacje mają identyczną dokładność.")
    else:
        print("Istnieją różnice w dokładności między implementacjami.")

if __name__ == "__main__":
    compare_knn()
