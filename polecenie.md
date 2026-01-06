# Wyjaśnialny klasyfikator toksyczności wypowiedzi (LLM-based)
## Cel:
Stworzyć model oparty na LLM, który klasyfikuje toksyczność krótkich wypowiedzi (np. komentarzy) i pokazuje, które fragmenty tekstu były problematyczne.

## Zakres (minimalny):

* Przygotować zbiór komentarzy z etykietami (toksyczny / nietoksyczny / graniczny).

* Zaimplementować klasyfikację LLM.

## Warstwa wyjaśnialności:

* podświetlanie w tekście słów/zwrotów odpowiedzialnych za klasyfikację,

* analiza błędnych decyzji (np. ironia, żart).

# Wspólne założenia dla wszystkich projektów
W każdym projekcie zespół powinien:

1. Zdefiniować zadanie i dane

* Przygotować / dobrać min. 50–100 przykładów (promptów) z ręcznie nadanymi etykietami / oczekiwanymi odpowiedziami lub ocenami.

* Zwięźle opisać domenę (co model ma robić i dla kogo).

2. Zbudować pipeline z LLM

* Wybrać 1–2 modele (API lub open-source).

* Zaprojektować prompt(y), ew. prosty RAG (retrieval-augmented generation).

3. Przeprowadzić ewaluację

* Zdefiniować 1–2 proste metryki jakości (accuracy, F1, zgodność z etykietą, ocena ekspercka itp.).

* Porównać co najmniej dwa warianty (np. inne prompty, inny model, dodatkowy kontekst).

4. Dodać warstwę wyjaśnialności (co najmniej jedna z opcji):

  * podświetlanie ważnych fragmentów tekstu,

  * wizualizacja użytych dokumentów (dla RAG),

  * chain-of-thought (rozumowanie krok po kroku),

  * self-critique / auto-ocena odpowiedzi,

  * LLM jako generator opisowych wyjaśnień dla użytkownika.

5. Przygotować raport i krótkie demo

* Opis zadania, metody, wyników, ograniczeń.

* 5–10 ciekawych przykładów: „success case” + „failure case” z komentarzem.
