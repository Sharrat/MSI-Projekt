# MSI-Projekt

W pliku incremental_kmeans.py znajduję się implementacja algorytmu inkrementalnego kmeans która jest kompatybilna z API sci-learn (dziedziczy BaseEstimator i ClassifierMixin)

W pliku Eksperymenty.py znajduję się implementacja wstępnego eksperymentu pokazującego działanie algorytmu przedstawionego wyżej, wykorzystuje się bibliotekę stream-learn w celu wygenerowania danych strumieniowych i następnie przeprowadza się ewaluację wykorzystując wstępnie w celach testu metrykę 'adjusted_rand_score' wykorzystując klasę z biblioteki stream-learn: TestThenTrain która implementuję również metodę walidacji krzyżowej o tej samej nazwie.
