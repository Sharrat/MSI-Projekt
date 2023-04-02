import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
class IncrementalKmeans(BaseEstimator, ClassifierMixin):
    def __init__(self, k=2,max_iter=100,init='k-means++', random_state=None, batch=100):
        self.k=k # liczba klastrów którą chcemy uzyskać
        self.max_iter = max_iter # Maksymalna liczba iteracji algorytmu
        self.batch = batch # Rozmiar jednej partii dla uczenia dla streamów
        self.init = init #metoda inicjalizacji centroidów
        self.random_state = random_state # Wybór stanu losowego w celu zapewnienia mechanizmu reprodukowalności wyników
        self.centroidy = None #Miejsce na centroidy
        self.liczniki = None #Miejsce na liczniki klastrów
        self.klasy = None #miejsce na klasy

    def predict(self, X):
        najblizszy_klaster = self._przypisz_klastry(X, self.centroidy)
        return najblizszy_klaster

    def partial_fit(self, X, y, classes=None):
        #inicjalizacja centroidów zgodnie z wybraną metodą
        if self.centroidy is None:
            if self.init != 'k-means++':
                self.centroidy = X[numpy.random.choice(X.shape[0], self.k, replace=False)]
            else:
                self.centroidy = self._inicjalizacja_kpp(X, self.k,self.random_state)
            self.liczniki = numpy.zeros(self.k)
            self.klasy = classes
        else:
            #aktualizuj klasy
            if classes is not None:
                self.klasy = numpy.unique(numpy.concatenate([self.klasy, classes]))

        #Aktualizuj centra klastrów z użyciem nowych danych
        najblizszy_klaster = self._przypisz_klastry(X, self.centroidy)
        self._aktualizuj_centra(X, najblizszy_klaster, y)

        return self

    def _przypisz_klastry(self,X,centroidy):
        #obliczamy najpierw odległości od każdego punktu do każdego centrum klastra
        odleglosci = numpy.linalg.norm(X[:,numpy.newaxis,:]-centroidy, axis=2)
        #przypisujemy każdy punkt do najbliższego centroida
        najblizszy_klaster = numpy.argmin(odleglosci, axis=1)
        return najblizszy_klaster

    def _aktualizuj_centra(self, X, najblizszy_klaster, y=None):
        # Zaktualizuj centra klastrów jako średnią punktów w klastrze
        for k in range(self.k):
            maska = najblizszy_klaster == k
            if numpy.sum(maska)>0:
                self.centroidy[k]=numpy.average(X[maska],axis=0)
                self.liczniki[k]+=numpy.sum(maska)

    def _inicjalizacja_kpp(self, X, k, random_state):
        #pierwsze centrum klastra jest wybierane losowo
        losuj = numpy.random.default_rng(random_state)
        pierwszy_centroid = losuj.choice(X.shape[0])
        centroidy = [X[pierwszy_centroid]]
        # wybranie pozostałych za pomocą algorytmu k-means++
        for x in range(1, k):
            #najpierw obliczamy minimalne odległości do centroidu dla każdego punktu
            odleglosci = numpy.linalg.norm(X[:, numpy.newaxis, :] - centroidy, axis=2)
            min_odleglosci = numpy.min(odleglosci, axis=1)
            #wartość p jest obliczana jako stosunek kwadratu odległości punktu od najbliższego centroidu do sumy kwadratów odległości wszystkich punktów od ich najbliższych centroidów.
            nastepny_centroid = losuj.choice(numpy.arange(X.shape[0]),p=min_odleglosci**2/numpy.sum(min_odleglosci**2))
            centroidy.append(X[nastepny_centroid])
        return numpy.array(centroidy)


# Autorzy: Adam Sołtysiak, Miłosz Woźniak


