import numpy as np
import time
import scipy.linalg as la
import pandas as pd

# Definicja parametrow
M = 200   # Liczba wierszy macierzy C
N = 30   # Rozmiar macierzy jednostkowej
L = 20   # Liczba kolumn macierzy C

# Generowanie macierzy
I_N = np.eye(N)  # Macierz jednostkowa N x N
C = np.random.rand(M, L)  # Macierz losowa M x L
A = np.kron(I_N, C.T @ C)  # Macierz A = I_N \otimes (C^T C)

# Generowanie wektora x i obliczenie prawej strony b = Ax
x = np.random.randn(N * L)  # Wektor losowy o rozkladzie normalnym
b = A @ x  # Wektor prawej strony

# Slownik do przechowywania wynikow
results = {}

# 1. Eliminacja Gaussa
start_time = time.time()
x_gauss = np.linalg.solve(A, b)
gauss_time = time.time() - start_time
gauss_error = np.linalg.norm(x_gauss - x)
results["Gauss"] = (round(gauss_time, 4), format(gauss_error, ".10e"))

# 2. Faktoryzacja LU
start_time = time.time()
lu, piv = la.lu_factor(A)
x_lu = la.lu_solve((lu, piv), b)
lu_time = time.time() - start_time
lu_error = np.linalg.norm(x_lu - x)
results["LU"] = (round(lu_time, 4), format(lu_error, ".10e"))

# 3. Faktoryzacja QR
start_time = time.time()
Q, R = np.linalg.qr(A)
x_qr = la.solve_triangular(R, Q.T @ b)
qr_time = time.time() - start_time
qr_error = np.linalg.norm(x_qr - x)
results["QR"] = (round(qr_time, 4), format(qr_error, ".10e"))

# Tworzenie tabeli wynikow
df_results = pd.DataFrame(results, index=["Czas wykonania (s)", "Blad aproksymacji"])

# Wyswietlenie wynikow
print("\nPorownanie metod rozwiazania ukladu rownan:")
print(df_results)