import numpy as np
import matplotlib.pyplot as plt

# Dyskretyzacja przedziału [0, π] na 20 punktów
x = np.linspace(0, np.pi, 20)
y = np.pi ** 2 - x ** 2  # Funkcja f(x) = π² - x²

# Wykres danych oryginalnych i aproksymacji
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(x, y, '.b', markersize=10, label='Oryginalna funkcja: f(x) = π² - x²')

errors = []  # Lista do przechowywania błędów dopasowania

# Pętla po różnych liczbach składników szeregu kosinusowego (n = 1 to 10)
for n in range(1, 11):
    # Budowanie macierzy A: kolumny to 1, cos(x), cos(2x), ..., cos(nx)
    A = np.ones((len(x), 1))
    for i in range(1, n + 1):
        A = np.hstack((A, np.cos(i * x).reshape(-1, 1)))

    # Rozwiązanie równania najmniejszych kwadratów: a = (A^T A)^-1 A^T b
    a = np.linalg.lstsq(A, y, rcond=None)[0]

    # Obliczenie wartości funkcji przybliżonej g(x) = a0 + a1*cos(x) + ...
    g = np.zeros_like(x)
    for i in range(n + 1):
        g += a[i] * np.cos(i * x)

    # Dodanie linii aproksymacji do wykresu
    plt.plot(x, g, label=f'n = {n}')

    # Obliczenie błędu ||A·a - y||₂
    err = np.linalg.norm(A @ a - y)
    errors.append(err)

plt.title('Dopasowanie szeregu kosinusowego do funkcji f(x)')
plt.xlabel('x');
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

# Wykres błędu dopasowania dla każdego n
plt.subplot(2, 1, 2)
plt.plot(range(1, 11), errors, marker='o')
plt.title('Błąd dopasowania ||A·a - y||₂ w zależności od n')
plt.xlabel('n')
plt.ylabel('Błąd')
plt.grid(True)

plt.tight_layout()
plt.show()
