import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from main import MyKNN


def compare_knn_mnist():
    # Wczytanie zbioru danych
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Konwersja danych
    X = data.data.view(data.data.size(0), -1).numpy()  # Flatten the images
    y = data.targets.numpy()

    # Subsampling - Użyj mniejszej liczby próbek
    subset_size = 10000  # Zamiast 60,000 próbek użyj 10,000
    X = X[:subset_size]
    y = y[:subset_size]

    # Normalizacja danych
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Redukcja wymiarowości - IncrementalPCA zamiast PCA
    from sklearn.decomposition import IncrementalPCA
    pca = IncrementalPCA(n_components=20)  # Użycie IncrementalPCA z 20 komponentami
    X_reduced = pca.fit_transform(X)

    # Podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

    # Własna implementacja KNN
    my_knn = MyKNN(k=3)
    my_knn.fit(X_train, y_train)
    y_pred_my_knn = my_knn.predict(X_test)
    accuracy_my_knn = accuracy_score(y_test, y_pred_my_knn)
    print(f"Dokładność własnego KNN na zbiorze MNIST z normalizacją: {accuracy_my_knn:.2f}")

    # Implementacja KNN z biblioteki scikit-learn
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Dokładność KNN z biblioteki scikit-learn: {accuracy_sklearn:.2f}")


if __name__ == "__main__":
    compare_knn_mnist()
