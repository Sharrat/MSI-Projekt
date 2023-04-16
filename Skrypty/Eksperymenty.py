from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from Inkrementalny_kmeans import IncrementalKmeans
from sklearn.metrics import normalized_mutual_info_score
from MiniBatchKmeans import MiniBatchKMeansx
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon

#--------------------------Incremental-kmeans------------------------------#
strumien = StreamGenerator(random_state=42, n_chunks=100, chunk_size=100, n_classes=3, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)

metryki = [adjusted_rand_score, normalized_mutual_info_score]
# Inicjalizacja modelu Incremental KMeans
model = IncrementalKmeans(k=3,random_state=42)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)
#--------------------------------------------------------------------------#
ScoresIncremental = ewaluator.scores
#------------------------------MiniBatch-kmeans-------------------------------#
strumien = StreamGenerator(random_state=42, n_chunks=100, chunk_size=100, n_classes=3, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)

metryki = [adjusted_rand_score, normalized_mutual_info_score]
# Inicjalizacja modelu Incremental KMeans
model = MiniBatchKMeansx(n_clusters=3,random_state=42)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)

#--------------------------------------------------------------------------#

ScoresMiniBatch = ewaluator.scores

#-----------------------------T-Test---------------------------------------#
print("------------------------------------------------------------------------------------------")
print("TEST T-STUDENTA:")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: t = {t}, p = {p}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#



#-----------------------------Wilcoxon Signed-Rank Test--------------------#
print("------------------------------------------------------------------------------------------")
print("TEST Wilcoxon Signed-Rank Test:")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: statistic = {statistic}, p-value = {pvalue}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#



#--------------------------------WYKRESY-----------------------------------#
plt.figure(figsize=(9, 4))
for m, metryka in enumerate(metryki):
    plt.plot(ScoresIncremental[0, :, m], label=metryka.__name__+" Incremental-kmeans")
    plt.plot(ScoresMiniBatch[0, :, m], label=metryka.__name__+" Mini-batch-kmeans")
plt.title("Algorytm Incremental KMeans")
plt.ylim(-0.5, 1)
plt.ylabel('Jakość')
plt.xlabel('Partia')
plt.legend()
plt.show()
#--------------------------------------------------------------------------#
exit(0)


# Autorzy: Adam Sołtysiak, Miłosz Woźniak