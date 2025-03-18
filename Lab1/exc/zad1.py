import numpy as np

def gauss_elimination(A, B):
    n = len(B)

    # Tworzymy rozszerzoną macierz [A | B]
    augmented_matrix = np.hstack((A, B.reshape(-1, 1)))

    # Eliminacja Gaussa
    for i in range(n):
        # Wybór elementu podstawowego (pivot)
        pivot = augmented_matrix[i, i]
        if pivot == 0:
            raise ValueError("Znaleziono zerowy pivot, konieczna zamiana wierszy.")

        # Normalizacja wiersza
        augmented_matrix[i] /= pivot

        # Eliminacja w pozostałych wierszach
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    # Podstawianie wsteczne
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i+1:n] * x[i+1:n])

    return x

# Definicja macierzy A i wektora B
A = np.array([[2, -1,  0,  0],
              [-1, 2, -1,  0],
              [0, -1, 2, -1],
              [0,  0, -1, 2]], dtype=float)

B = np.array([0, 0, 0, 5], dtype=float)

# Obliczanie rozwiązania
solution = gauss_elimination(A, B)

# Wyświetlenie wyników
print("Rozwiązanie układu równań metodą eliminacji Gaussa:", solution)
