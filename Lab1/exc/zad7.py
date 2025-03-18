import numpy as np


def gram_schmidt(A):
    """Implementacja ortogonalizacji Grama-Schmidta dla macierzy A."""
    m, n = A.shape          # Pobranie wartosci macierzy A (m - wiersze, n - kolumny)
    Q = np.zeros((m, n))    # Tworzymy pusta macierz Q
    R = np.zeros((n, n))    # Tworzymy pusta macierz kwadratowa

    for i in range(n):
        # Pobieramy i-tą kolumne macierzy
        v = A[:, i]

        for j in range(i):
            # Obliczamy rzut wektora a_i na poprzednie q_j
            R[j, i] = np.dot(Q[:, j], A[:, i])   # Iloczyn skalarny q_j i a_i
            v = v - R[j, i] * Q[:, j]            # Usuwamy składową w kierunku q_j

        # Normalizacja wektora
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R


# Definiujemy macierz A
A = np.array([
    [-2, 1, 2, 1],
    [2, -1, 2, 1],
    [2, 3, -4, 5],
    [2, 3, 0, -1]
], dtype=float)

# Wyznaczamy faktoryzację QR
Q, R = gram_schmidt(A)

# Zaokrąglnie wyniku:
Q = np.round(Q, 3)  # Zaokraglenie do 3 miejsc po przecinku
R = np.round(R, 3)  # Zaokraglenie macierzy R

# Normalizacja znaku macierzy R
for i in range(R.shape[0]):
    if R[i, i] < 0:     # Jesli wartość na diagonali jest ujemna
        Q[:, i] *= -1   # Zamiana znaku kolumny Q
        R[i, :] *= -1   # Zamiana znaku wiersza R

# Wyswietlenie wynikow
print("Macierz Q:")
print(Q)

print("\nMacierz R:")
print(R)

# Sprawdzamy poprawnosc iloczynu A = Q * R
print("\nMacierz A ≈ Q * R:")
print(np.round(Q @ R, 0))  # Porownanie z pierwotną macierzą A (zaokraglenie do liczb calkowitych)
