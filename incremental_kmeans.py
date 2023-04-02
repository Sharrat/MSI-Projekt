import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

'''Poniżej znajduje się podsumowanie tego, co robi każda metoda w klasie IncrementalKmeans:

__init__: Inicjuje obiekt z parametrami, takimi jak liczba klastrów, 
maksymalna liczba iteracji, rozmiar partii, metoda inicjalizacji centroidów i stan losowy. 
Inicjuje również miejsca na centroidy, liczniki i klasy.

predict: Dla danego zestawu punktów danych przypisuje 
każdy punkt do najbliższego centrum klastra i zwraca indeksy klastrów.

fit: Dopasowuje model do danych, dzieląc dane na partie i wywołując metodę partial_fit 
na każdej partii danych. Przetasowuje dane po każdej iteracji.

partial_fit: Wykonuje jeden krok algorytmu k-średnich inkrementalnych na 
partii punktów danych. Inicjalizuje centroidy, jeśli nie zostały jeszcze zainicjalizowane, 
aktualizuje klasy, przypisuje punkty danych do najbliższych klastrów, a następnie aktualizuje centra klastrów.

_assign_clusters: Dla danego zestawu punktów danych i centroidów oblicza odległość od 
każdego punktu do każdego centroidu i przypisuje każdy punkt do najbliższego centroidu.

_update_centers: Aktualizuje centra klastrów na podstawie średniej punktów w 
każdym klastrze, uwzględniając etykiety docelowe y oraz wagi próbek sample_weight, jeśli są dostarczone.

_kmeans_plus_plus: Inicjalizuje centroidy klastrów za pomocą metody k-means++, 
która zapewnia lepsze początkowe przybliżenie centroidów niż losowe wybieranie punktów.

_shuffle: Przetasowuje dane inplace (w miejscu), aby dane były przetwarzane w losowej kolejności.
'''
class IncrementalKmeans(BaseEstimator, ClassifierMixin):

    def __init__(self, n_clusters=8, max_iter=100, batch_size=100, init='k-means++', random_state=None):
        self.n_clusters = n_clusters # Liczba klastrów
        self.max_iter = max_iter # Maksymalna liczba iteracji
        self.batch_size = batch_size # Rozmiar partii dla uczenia online
        self.init = init # Metoda inicjalizacji centroidów
        self.random_state = random_state # Stan losowy dla reprodukowalności
        self.centroids = None # Miejsce na centroidy
        self.counts = None # Miejsce na liczniki klastrów
        self.classes = None # Miejsce na klasy
    def predict(self, X):
        # Przypisz każdy punkt do najbliższego centrum klastra
        najblizszy_klaster = self._assign_clusters(X, self.centroids)
        return najblizszy_klaster

    def fit(self, X, y=None, classes=None, sample_weight=None):
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        # Przetasuj dane przed przetwarzaniem
        X_przetasowane, y_przetasowane, sample_weight_przetasowane = self._shuffle(X, y, sample_weight)

        for iteration in range(self.max_iter):
            for batch_idx in range(n_batches):
                # Pobierz aktualną partię danych
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_przetasowane[start:end]
                y_batch = y_przetasowane[start:end] if y_przetasowane is not None else None
                sample_weight_batch = sample_weight_przetasowane[
                                      start:end] if sample_weight_przetasowane is not None else None

                # Wykonaj częściowe dopasowanie przy użyciu aktualnej partii danych
                self.partial_fit(X_batch, y_batch, classes, sample_weight_batch)

            # Przetasuj dane po każdej iteracji
            X_przetasowane, y_przetasowane, sample_weight_przetasowane = self._shuffle(X_przetasowane, y_przetasowane,
                                                                                       sample_weight_przetasowane)

            return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # Jeśli centroidy nie zostały zainicjalizowane, zainicjalizuj je
        if self.centroids is None:
            if self.init == 'k-means++':
                self.centroids = self._kmeans_plus_plus(X, self.n_clusters, self.random_state)
            else:
                self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
            self.counts = np.zeros(self.n_clusters)
            self.classes = classes
        else:
            # Zaktualizuj klasy
            if classes is not None:
                self.classes = np.unique(np.concatenate([self.classes, classes]))

        # Zaktualizuj centra klastrów przy użyciu nowych danych
        najblizszy_klaster = self._assign_clusters(X, self.centroids)
        self._update_centers(X, najblizszy_klaster, y, sample_weight)

        return self

    def _assign_clusters(self, X, centroids):
        # Oblicz odległość od każdego punktu do każdego centrum klastra
        odleglosci = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        # Przypisz każdy punkt do najbliższego centrum klastra
        najblizszy_klaster = np.argmin(odleglosci, axis=1)
        return najblizszy_klaster

    def _update_centers(self, X, najblizszy_klaster, y=None, sample_weight=None):
        # Zaktualizuj centra klastrów jako średnią punktów w klastrze
        for k in range(self.n_clusters):
            mask = najblizszy_klaster == k
            if np.sum(mask) > 0:
                if sample_weight is not None:
                    weight = sample_weight[mask]
                else:
                    weight = None
                if y is not None:
                    class_indices = np.where(np.in1d(self.classes, np.unique(y[mask])))[0]
                    if len(class_indices) > 0:
                        class_weight = weight[
                            np.in1d(y[mask], self.classes[class_indices[0]])] if weight is not None else None
                    else:
                        class_weight = None
                else:
                    class_weight = None
                self.centroids[k] = np.average(X[mask], axis=0, weights=class_weight)
                self.counts[k] += np.sum(mask)
    def _kmeans_plus_plus(self, X, n_clusters, random_state):
        # Wybierz pierwsze centrum klastra jednolicie losowo z danych
        rng = np.random.default_rng(random_state)
        pierwszy_centroid_idx = rng.choice(X.shape[0])
        centroids = [X[pierwszy_centroid_idx]]

        # Wybierz pozostałe centroidy za pomocą algorytmu k-means++
        for _ in range(1, n_clusters):
            odleglosci = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
            # Oblicz minimalną odległość do centroidu dla każdego punktu
            min_odleglosc_do_centroidu = np.min(odleglosci, axis=1)
            # Wybierz kolejny centroid na podstawie kwadratowych prawdopodobieństw odległości
            next_centroid_idx = rng.choice(
                np.arange(X.shape[0]),
                p=min_odleglosc_do_centroidu ** 2 / np.sum(min_odleglosc_do_centroidu ** 2))
            centroids.append(X[next_centroid_idx])

        return np.array(centroids)

    def _shuffle(self, X, y=None, sample_weight=None):
        # Przetasuj dane inplace
        idx = np.random.permutation(X.shape[0])
        X_przetasowane = X[idx]

        if y is not None:
            y_przetasowane = y[idx]
        else:
            y_przetasowane = None

        if sample_weight is not None:
            sample_weight_przetasowane = sample_weight[idx]
        else:
            sample_weight_przetasowane = None

        return X_przetasowane, y_przetasowane, sample_weight_przetasowane


# Autorzy: Adam Sołtysiak, Miłosz Woźniak