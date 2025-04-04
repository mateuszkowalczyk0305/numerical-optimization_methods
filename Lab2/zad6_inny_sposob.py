import numpy as np


def algorytm_svd(A):
    # Obliczenie macierzy A^T * A
    ATA = np.dot(A.T, A)

    # Wyznaczenie wartości własnych i wektorów własnych macierzy A^T A
    eigvals, V = np.linalg.eig(ATA)

    # Sortowanie wartości własnych malejąco
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    V = V[:, sorted_indices]

    # Wyznaczenie sigma - macierzy diagonalnej z pierwiastkami z wartości własnych
    sigma = np.sqrt(np.maximum(eigvals, 0))

    # Tworzenie macierzy diagonalnej Sigma
    m, n = A.shape
    Sigma = np.zeros((m, n))
    for i in range(len(sigma)):
        if i < min(m, n):
            Sigma[i, i] = sigma[i]

    # Wyznaczenie U jako A * V * inv(sigma)
    Sigma_inv = np.linalg.pinv(Sigma[:n, :n])
    U = np.dot(A, np.dot(V, Sigma_inv))

    return U, Sigma, V


# ======================================
# Definiowanie macierzy z treści zadania
# ======================================

A1 = np.array([
    [1, 1],
    [0, 1],
    [1, 0]
])

A2 = np.array([
    [-1, 2, 2]
])

A3 = np.array([
    [2, 2, 2],
    [17 / 10, 17 / 10, 17 / 10],
    [3 / 5, 3 / 5, 3 / 5],
    [9 / 5, 9 / 5, 9 / 5],
    [-9 / 5, -9 / 5, -9 / 5]
])

# ======================================
# Wyznaczanie SVD dla wszystkich macierzy
# ======================================

U1, Sigma1, V1 = algorytm_svd(A1)
U2, Sigma2, V2 = algorytm_svd(A2)
U3, Sigma3, V3 = algorytm_svd(A3)

# ======================================
# Wyświetlanie wyników
# ======================================

np.set_printoptions(precision=4, suppress=True)  # Ustawienie zaokrąglania dla czytelnych wyników

print("SVD dla macierzy A1:")
print("U1 =\n", U1)
print("Sigma1 =\n", Sigma1)
print("V1 =\n", V1)
print("\n")

print("SVD dla macierzy A2:")
print("U2 =\n", U2)
print("Sigma2 =\n", Sigma2)
print("V2 =\n", V2)
print("\n")

print("SVD dla macierzy A3:")
print("U3 =\n", U3)
print("Sigma3 =\n", Sigma3)
print("V3 =\n", V3)
print("\n")
