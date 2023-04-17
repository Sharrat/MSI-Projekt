from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from Inkrementalny_kmeans import IncrementalKmeans
from sklearn.metrics import normalized_mutual_info_score
from MiniBatchKmeans import MiniBatchKMeansx
from Birch import Birchx
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
import sys

#EKSPERYMENT 1 PARAMETRY n_chunks=40, chunk_size=40 random_state=42, 3 klastry

metryki = [adjusted_rand_score, normalized_mutual_info_score]
print("EKSPERYMENT PIERWSZY")
#--------------------------Incremental-kmeans------------------------------#
strumien = StreamGenerator(random_state=42, n_chunks=40, chunk_size=40, n_classes=3, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Incremental KMeans
model = IncrementalKmeans(k=3,random_state=42)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)
#--------------------------------------------------------------------------#
ScoresIncremental = ewaluator.scores
#------------------------------MiniBatch-kmeans----------------------------#
strumien = StreamGenerator(random_state=42, n_chunks=40, chunk_size=40, n_classes=3, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Incremental KMeans
model = MiniBatchKMeansx(n_clusters=3,random_state=42)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)

#--------------------------------------------------------------------------#

ScoresMiniBatch = ewaluator.scores

#------------------------------Birch-------------------------------#
strumien = StreamGenerator(random_state=42, n_chunks=40, chunk_size=40, n_classes=3, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Birch
model = Birchx(n_clusters=3, random_state=42)

# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)

#--------------------------------------------------------------------------#

ScoresBirch = ewaluator.scores

#Zapisanie wynikow do pliku
f = open("wyniki.txt", 'w')
sys.stdout = f

#-----------------------------T-Test---------------------------------------#
print("------------------------------------------------------------------------------------------")
print("TEST T-STUDENTA : Incremental - mini-batch")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: t = {t}, p = {p}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#

print("TEST T-STUDENTA : Incremental - Birch")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncremental[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: t = {t}, p = {p}")
print("------------------------------------------------------------------------------------------")


#-----------------------------Wilcoxon Signed-Rank Test--------------------#
print("------------------------------------------------------------------------------------------")
print("TEST Wilcoxon Signed-Rank Test: Incremental - mini-batch")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: statistic = {statistic}, p-value = {pvalue}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#

print("TEST Wilcoxon Signed-Rank Test: Incremental - Birch")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncremental[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: statistic = {statistic}, p-value = {pvalue}")
print("------------------------------------------------------------------------------------------")




#--------------------------------WYKRESY-----------------------------------#
plt.figure(figsize=(15, 6))
for m, metryka in enumerate(metryki):
    plt.plot(ScoresIncremental[0, :, m], label=metryka.__name__+" Incremental-kmeans")
    plt.plot(ScoresMiniBatch[0, :, m], label=metryka.__name__+" Mini-batch-kmeans")
    plt.plot(ScoresBirch[0, :, m], label=metryka.__name__ + " Birch")
plt.title("Algorytm Incremental KMeans")
plt.ylim(-0.5, 1)
plt.ylabel('Jakość')
plt.xlabel('Partia')
plt.legend()
plt.show()
#--------------------------------------------------------------------------#
f.close()

#Wyswietlenie wynikow z pliku w terminalu
sys.stdout = sys.__stdout__
with open('wyniki.txt', 'r') as x:
    print(x.read())

#EKSPERYMENT 2 PARAMETRY n_chunks=33, chunk_size=10 random_state=77, 4 klastry
print("EKSPERYMENT DRUGI")
metryki = [adjusted_rand_score, normalized_mutual_info_score]

#--------------------------Incremental-kmeans------------------------------#
strumien = StreamGenerator(random_state=77, n_chunks=33, chunk_size=10, n_classes=4, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Incremental KMeans
model = IncrementalKmeans(k=4,random_state=77)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)
#--------------------------------------------------------------------------#
ScoresIncremental = ewaluator.scores
#------------------------------MiniBatch-kmeans----------------------------#
strumien = StreamGenerator(random_state=77, n_chunks=33, chunk_size=10, n_classes=4, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Incremental KMeans
model = MiniBatchKMeansx(n_clusters=4,random_state=77)
# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)

#--------------------------------------------------------------------------#

ScoresMiniBatch = ewaluator.scores

#------------------------------Birch-------------------------------#
strumien = StreamGenerator(random_state=77, n_chunks=33, chunk_size=10, n_classes=4, n_features=10, n_informative=2,
n_redundant=0, n_clusters_per_class=1)
# Inicjalizacja modelu Birch
model = Birchx(n_clusters=4, random_state=77)

# Inicjalizacja ewaluatora i dopasowanie modelu do strumienia danych
ewaluator = TestThenTrain(metryki)
ewaluator.process(strumien, model)

#--------------------------------------------------------------------------#

ScoresBirch = ewaluator.scores

#Zapisanie wynikow do pliku
f = open("wyniki2.txt", 'w')
sys.stdout = f

#-----------------------------T-Test---------------------------------------#
print("------------------------------------------------------------------------------------------")
print("TEST T-STUDENTA : Incremental - mini-batch")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: t = {t}, p = {p}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#

print("TEST T-STUDENTA : Incremental - Birch")
for m, metryka in enumerate(metryki):
    t, p = ttest_ind(ScoresIncremental[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: t = {t}, p = {p}")
print("------------------------------------------------------------------------------------------")


#-----------------------------Wilcoxon Signed-Rank Test--------------------#
print("------------------------------------------------------------------------------------------")
print("TEST Wilcoxon Signed-Rank Test: Incremental - mini-batch")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncremental[0, :, m], ScoresMiniBatch[0, :, m])
    print(f"{metryka.__name__}: statistic = {statistic}, p-value = {pvalue}")
print("------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------#

print("TEST Wilcoxon Signed-Rank Test: Incremental - Birch")
for m, metryka in enumerate(metryki):
    statistic, pvalue = wilcoxon(ScoresIncremental[0, :, m], ScoresBirch[0, :, m])
    print(f"{metryka.__name__}: statistic = {statistic}, p-value = {pvalue}")
print("------------------------------------------------------------------------------------------")




#--------------------------------WYKRESY-----------------------------------#
plt.figure(figsize=(15, 6))
for m, metryka in enumerate(metryki):
    plt.plot(ScoresIncremental[0, :, m], label=metryka.__name__+" Incremental-kmeans")
    plt.plot(ScoresMiniBatch[0, :, m], label=metryka.__name__+" Mini-batch-kmeans")
    plt.plot(ScoresBirch[0, :, m], label=metryka.__name__ + " Birch")
plt.title("Algorytm Incremental KMeans")
plt.ylim(-0.5, 1)
plt.ylabel('Jakość')
plt.xlabel('Partia')
plt.legend()
plt.show()
#--------------------------------------------------------------------------#
f.close()

#Wyswietlenie wynikow z pliku w terminalu
sys.stdout = sys.__stdout__
with open('wyniki2.txt', 'r') as x:
    print(x.read())



exit(0)


# Autorzy: Adam Sołtysiak, Miłosz Woźniak