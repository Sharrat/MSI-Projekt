# MSI-Projekt

<h2>Zawartość plików</h2>

W pliku Inkrementalny_kmeans.py znajduję się implementacja algorytmu inkrementalnego kmeans która jest kompatybilna z API sci-learn (dziedziczy BaseEstimator i ClassifierMixin)

W pliku Birch.py znajduje się zimportowany algorytm birch lekko zmieniony aby był kompatybilny z biblioteką stream-learn

W pliku MiniBatchKmeans.py znajduje się zimportowany algorytm Mini-Batch Kmeans lekko zmieniony aby był kompatybilny z biblioteką stream-learn

W pliku Eksperymenty.py znajduję się implementacja eksperymentów pokazujych działanie algorytmów przedstawionych wyżej, wykorzystuje się bibliotekę stream-learn w celu wygenerowania danych strumieniowych i następnie przeprowadza się ewaluację wykorzystując wstępnie w celach testu metryki adjusted_rand_score,normalized_mutual_info_score wykorzystując klasę z biblioteki stream-learn: TestThenTrain która implementuję również metodę walidacji krzyżowej o tej samej nazwie.

ewaluacja przeprowadzana jest dwukrotnie na różnych parametrach

wyniki poszczególnych eksperymentów wyświetlane są na wykresie jak i wyniki testów statystycznych zapisywane są w plikach (odpowiednio dla eksperymentu 1 i 2) wyniki.txt oraz wyniki2.txt

<h2>Uruchomienie eksperymentu </h2>

Uruchomienie następuję poprzez uruchomienie skryptu Eksperymenty.py.
Skrypty klasyfikatorów muszą być w tym samym folderze co skrypt eksperymenty.py
Skrypt zapisuje wyniki do wyżej wymienionych plików w tym samym folderze.
