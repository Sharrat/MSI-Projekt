from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from incremental_kmeans import IncrementalKmeans

strumien = StreamGenerator(random_state=42, n_chunks=100, chunk_size=100, n_classes=2, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=2)

metryki = [adjusted_rand_score]
# Inicjalizacja modelu Incremental KMeans
model = IncrementalKmeans(random_state=42, n_clusters=2, batch_size=100, max_iter=1000)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)
plt.figure(figsize=(6, 3))

for m, metryka in enumerate(metryki):
    plt.plot(ewaluator.scores[0, :, m], label=metryka.__name__)
plt.title("Algorytm Incremental KMeans")
plt.ylim(-0.5, 1)
plt.ylabel('Jakość')
plt.xlabel('Partia')
plt.legend()
plt.show()

exit(0)


# Autorzy: Adam Sołtysiak, Miłosz Woźniak