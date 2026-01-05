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

Kołos, A., Okulska, I., Głąbińska, K., Karlińska, A., Wiśnios, E., Ellerik, P., Prałat, A. [BAN-PL: a Novel Polish Dataset of Banned Harmful and Offensive Content from Wykop.pl web service](https://arxiv.org/abs/2308.10592). 2023. arXiv:2308.1059.

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

Borkan D., Dixon L., Sorensen J., Thain N., Vasserman L. [Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification](https://arxiv.org/abs/1903.04561). 2019. arXiv:1903.04561.

