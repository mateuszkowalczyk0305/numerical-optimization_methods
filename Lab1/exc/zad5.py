import numpy as np

def lu_decomposition(A):
    """
    Faktoryzacja LU macierzy A, zwraca macierze L i U
    """
    n = len(A)
    L = np.eye(n)           # utworzenie macierzy jednostkowej dolnotrójkątne L
    U = A.astype(float)     # przypisanie wartości macierzy A do macierzy U (docelowo górnotrójkątnej)

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i,i]       # obliczanie mnożnika
            L[j,i] = factor                 # zapamiętanie mnożnika w macierzy L
            for k in range(i, n):
                U[j,k] -= factor * U[i, k]  # eliminacja dolnej części macierzy

    return L, U

def forward_substitution(L, b):
    """
    Rozwiązanie układu Ly = b (podstawianie do przodu).
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        sum_Ly = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_Ly) / L[i, i]

    return y


def backward_substitution(U, y):
    """
    Rozwiązanie układu Ux = y (podstawianie wstecz).
    """
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_Ux = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_Ux) / U[i, i]

    return x


# Nasze dane z zadania:
A = np.array([[1, 2, 3, 4],
              [-1, 1, 2, 1],
              [0, 2, 1, 3],
              [0, 0, 1, 1]])

b = np.array([1, 1, 1, 1])

# Faktoryzacja:
L, U = lu_decomposition(A)

# Rozwiązanie układu równań:
y = forward_substitution(L, b)
x = backward_substitution(U, y)

# Wyznacznik macierzy:
det_A = np.prod(np.diag(U))

# Wyniki:
print("Macierz L:\n", L, "\n")
print("Macierz U:\n", U, "\n")
print(f"Wyznacznik macierzy A: {det_A}\n")
print(f"Rozwiązanie układu Ax = b: x = {[np.round(x,3).tolist()]}\n") # zaokrąglenie do 3 liczb znaczących

