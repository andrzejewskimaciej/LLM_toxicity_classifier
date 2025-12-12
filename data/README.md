# Zbiory danych

### [BAN-PL: Definicja Granicy przez Moderację Profesjonalną](https://github.com/ZILiAT-NASK/BAN-PL)

Zbiór **BAN-PL** (2024) to przełomowy zasób, który zmienia definicję "prawdy" w detekcji toksyczności. Zamiast opierać się na ankieterach, wykorzystuje on decyzje profesjonalnych moderatorów serwisu Wykop.pl.   

#### Struktura i Definicja Graniczności

Zbiór składa się z par postów: usuniętych (zbanowanych) oraz neutralnych. Kluczowym aspektem dla definicji "graniczności" jest fakt, że klasa neutralna zawiera również treści, które były **zgłoszone przez użytkowników** (oflagowane), ale po weryfikacji przez moderatorów zostały **pozostawione**.

- **Treści Graniczne w BAN-PL:** To właśnie te posty – zgłoszone przez społeczność jako szkodliwe, ale obronione przez moderatorów – stanowią idealną definicję treści granicznych w polskim internecie. Są one kontrowersyjne, mogą budzić sprzeciw, ale mieszczą się w granicach regulaminu serwisu.

- **Mechanizm Decyzyjny:** Decyzje o banowaniu podejmowane były zazwyczaj w głosowaniu 5 moderatorów. Dokumentacja zbioru  wspomina o przypadkach braku jednomyślności (np. głosowania 3:2), co potwierdza istnienie szarej strefy, choć publicznie udostępniony zbiór zawiera etykiety końcowe (`harmful`/`neutral`).   

#### Specyfika Lingwistyczna

BAN-PL jest bogaty w przykłady typowo polskiej inwencji w omijaniu filtrów:

- **Maskowanie wulgaryzmów:** Użycie znaków specjalnych w środku polskich słów (np. "k***a"), co jest trudne dla modeli tokenizujących podsłowa.

- **Fleksja i Neologizmy:** Tworzenie obraźliwych neologizmów poprzez dodawanie polskich końcówek do obcych rdzeni lub nazwisk.   

- **Substytucja fonetyczna:** Zapisywanie słów tak, jak się je słyszy, ale z błędami (np. "duq" zamiast "duck" w kontekście anglicyzmów, lub polskie odpowiedniki).

### [PolEval 2019 Task 6: Hierarchia Szkodliwości](https://huggingface.co/datasets/poleval/poleval2019_cyberbullying)

Zbiór przygotowany na potrzeby konkursu PolEval 2019 jest fundamentalnym zasobem do badania gradacji szkodliwości w polskim Twitterze.   

#### Skala Trójstopniowa (Zadanie 6-2)

Organizatorzy wprowadzili podział, który pozwala na rozróżnienie ciężaru gatunkowego agresji:

1. **Klasa 0 (Non-harmful):** Tweety neutralne.

2. **Klasa 1 (Cyberbullying):** Cyberprzemoc. Kategoria ta obejmuje ataki personalne, nękanie, wyśmiewanie, plotki. Często są to treści **graniczne** z punktu widzenia prawa (mogą nie być ścigane z urzędu), ale są wysoce szkodliwe dla jednostki.

3. **Klasa 2 (Hate-speech):** Mowa nienawiści. Ataki na grupy chronione (rasa, religia, orientacja). Jest to kategoria "najcięższa".

Taka taksonomia pozwala na trenowanie modeli, które nie tylko wykrywają "coś złego", ale potrafią ocenić, czy mamy do czynienia z kłótnią (cyberbullying), czy z ideologiczną nienawiścią (hate speech).   

#### Proces Adnotacji

Zbiór ten był podwójnie weryfikowany: najpierw przez wolontariuszy, a następnie przez ekspertów od cyberprzemocy. Wprowadzenie "super-annotatora" w przypadkach spornych wskazuje na dbałość o jakość danych w strefie granicznej.

### [DynaHate: Adwersarskie Generowanie Trudności](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)

**DynaHate** (Dynamic Hate) to zbiór powstały w procesie iteracyjnym, w którym ludzie próbowali "oszukać" model. Składa się z czterech rund.   

#### Ewolucja Trudności

- **Runda 1:** Przykłady proste, często zawierające wyraźne słowa kluczowe.

- **Rundy 2-4:** Annotatorzy, widząc, że model wyłapuje proste ataki, zaczęli tworzyć treści coraz bardziej subtelne, ironiczne, oparte na stereotypach, ale bez wulgaryzmów.

Przykłady z późniejszych rund DynaHate stanowią wzorzec **toksyczności ukrytej** (*implicit toxicity*). Zbiór ten jest idealny do trenowania modeli na wykrywanie treści, które są "bezpieczne" leksykalnie, ale "toksyczne" semantycznie.

### [Civil Comments (Jigsaw): Złoty Standard Etykietowania Ułamkowego](https://www.tensorflow.org/datasets/catalog/civil_comments)

Zbiór **Civil Comments**, znany również jako *Jigsaw Unintended Bias in Toxicity Classification*, jest obecnie najbardziej kompleksowym zasobem publicznym, który bezpośrednio odpowiada na potrzebę posiadania danych ze "skalą" toksyczności. Pochodzi on z archiwum platformy Civil Comments, która obsługiwała sekcje komentarzy dla niezależnych wydawców w USA w latach 2015–2017.   

#### Struktura Danych i Definicja Skali

Fundamentem tego zbioru jest odejście od binarnej decyzji pojedynczego moderatora. Każdy z blisko 2 milionów komentarzy został poddany ocenie przez grupę crowdworkerów (do 10 osób). Kluczowa kolumna `toxicity` zawiera wartość zmiennoprzecinkową (`float`) z zakresu od 0.0 do 1.0. Wartość ta reprezentuje frakcję annotatorów, którzy uznali komentarz za toksyczny.

- **Skala Toksyczności:**
  
  - `0.0` – Pełny konsensus co do bezpieczeństwa treści.
  
  - `0.2` – Treść w większości bezpieczna, ale budząca zastrzeżenia u mniejszości (tzw. *low-severity toxicity*).
  
  - `0.5` – Idealny przykład treści **granicznej**. Komentarz podzielił oceniających na dwie równe grupy. W praktyce są to często wypowiedzi sarkastyczne, ostre opinie polityczne lub teksty używające kontrowersyjnego słownictwa w neutralnym kontekście.
  
  - `0.8 - 1.0` – Treści jednoznacznie toksyczne (konsensus).

Taka konstrukcja pozwala na dowolne definiowanie progu odcięcia (np. dla surowej moderacji próg >0.8, dla systemów ostrzegawczych próg >0.4) oraz trenowanie modeli regresyjnych przewidujących stopień oburzenia społecznego.   

#### Podwymiary i Atrybuty

Oprócz głównej etykiety `toxicity`, zbiór dostarcza analogiczne ułamkowe oceny dla podkategorii, co pozwala na wielowymiarową analizę profilu szkodliwości:

- `severe_toxicity` (skrajna toksyczność)

- `obscene` (wulgarność/sprośność)

- `threat` (groźba)

- `insult` (zniewaga)

- `identity_attack` (atak na tożsamość)

- `sexual_explicit` (treści seksualne)

Szczególnie wartościowe dla analizy treści granicznych są podzbiory **CivilCommentsIdentities** oraz **CivilCommentsCovert**. Ten drugi zawiera specjalistyczne adnotacje dotyczące **niejawnej agresji** (*covert offensiveness*), w tym sarkazmu i mikroagresji, które stanowią najtrudniejszą część spektrum toksyczności. Zbiór ten bezpośrednio adresuje problem, gdzie tekst jest technicznie poprawny, ale intencjonalnie szkodliwy.   

#### Dostępność i Licencja

Zbiór jest dostępny na licencji **CC0 (Public Domain)**, co czyni go w pełni bezpiecznym do zastosowań komercyjnych i badawczych. Jest hostowany na platformach takich jak Kaggle, TensorFlow Datasets oraz HuggingFace.

### [ETHOS: Wykrywanie Granic poprzez Aktywne Uczenie](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)

Zbiór **ETHOS** (OnlinE haTe speecH detectiOn dataSet) powstał z wykorzystaniem protokołu **aktywnego uczenia się** (*active learning*), zaprojektowanego specjalnie w celu identyfikacji i adnotacji przykładów leżących na granicy decyzyjnej modelu.   

#### Metodologia Uncertainty Sampling

Autorzy ETHOS wykorzystali algorytm, który wybierał do ludzkiej weryfikacji te komentarze (z YouTube i Reddit), dla których wstępny model był najmniej pewny – tj. zwracał prawdopodobieństwo toksyczności w przedziale `[0.4, 0.6]`. W rezultacie, ETHOS jest zbiorem nasyconym treściami **granicznymi**, które są trudne do klasyfikacji zarówno dla maszyn, jak i ludzi.

#### Wersja Binarna i Multi-Label

Zbiór występuje w dwóch wariantach:

1. **ETHOS Binary:** 998 komentarzy z etykietą binarną. Ze względu na metodę doboru, nawet te binarne etykiety są wynikiem rozstrzygania trudnych dylematów klasyfikacyjnych.

2. **ETHOS Multi-Label:** 433 przykłady mowy nienawiści, opisane szczegółowo w 8 wymiarach (np. czy nawołuje do przemocy, czy jest atakiem skierowanym w osobę czy grupę). Te dodatkowe wymiary pozwalają na stopniowanie wagi przewinienia – atak nawołujący do przemocy jest na szczycie skali toksyczności, podczas gdy generalizacja bez przemocy może być niżej.


