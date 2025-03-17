import numpy as np

# Eliminacja Gaussa:
def gaussian_elimination(A, b):
    """
    Oblicza układ równań metodą eliminacji Gaussa.
    """
    n = len(b)           # dla równań liniowych jest to spełnione
    A = A.astype(float)  # Konwersja na float
    b = b.astype(float)

    # Eliminacja wierszy
    for i in range(n):                      # przechodzimy przez każdy wiersz macierzy
        for j in range(i + 1, n):           # Iterujemy przez wiersze poniżej aktualnego pivotu
            factor = A[j, i] / A[i, i]
            for k in range(i, n):
                A[j, k] -= factor * A[i, k] # odejmujemy wielokrotność pivotu pomnożoną przez factor aby uzyskać zero
            b[j] -= factor * b[i]           # analogicznie dla wyrazów wolnych

    # Podstawianie wsteczne:
    x = np.zeros(n)                         # pusty wektor do trzymania wyników
    for i in range(n - 1, -1, -1):          # iterujemy od ostatniego wiersza do pierwszego (Podst. wsteczne)
        sum_ax = 0
        for j in range(i + 1, n):           # obliczamy sume iloczynów
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]    # rozwiązujemy równanie

    return x

# Norma macierzy:
def matrix_norm_1(A):
    """
    Oblicza normę macierzy ||A||_1 jako maksimum sumy wartości bezwzględnych w kolumnach.
    """
    return max(np.sum(np.abs(A), axis=0))  # Suma po wierszach dla każdej kolumny, a następnie maksimum (axis = 0 - wertykalnie (kolumnowo))

# Wskaźnik uwarunkowania macierzy:
def condition_number_1(A):
    """
    Oblicza wskaźnik uwarunkowania macierzy jako ||A||_1 * ||A^(-1)||_1.
    """
    A_inv = np.linalg.inv(A)  # Obliczenie macierzy odwrotnej
    norm_A = matrix_norm_1(A)  # Norma 1 macierzy A
    norm_A_inv = matrix_norm_1(A_inv)  # Norma 1 macierzy odwrotnej

    return norm_A * norm_A_inv

# Macierz A i wektor b
A = np.array([[0.835, 0.667],
              [0.333, 0.266]])
# Deklaracja wyrazów wolnych w obu przypadkach
b_original = np.array([0.168, 0.067])
b_perturbed = np.array([0.168, 0.066])

# Rozwiązanie układu dla pierwotnego b
x_original = gaussian_elimination(A, b_original)

# Rozwiązanie układu dla zaburzonego b
x_perturbed = gaussian_elimination(A, b_perturbed)

# Wyznaczenie wskaźnika uwarunkowania macierzy:
cond_num = condition_number_1(A)

# Wyniki:
print(f"Dla b2 = 0.067: x1 = {x_original[0]:.3f}, x2 = {x_original[1]:.3f}")
print(f"Dla b2 = 0.066: x1 = {x_perturbed[0]:.3f}, x2 = {x_perturbed[1]:.3f}\n")
print("Wskaznik uwarunkowania macierzy: ", cond_num, "\n")

# Wniosek:
print("Macierz A jest żle uwarunkowana (wartość odbiega od 1), co oznacza, że układ równań jest bardzo wrażliwy na błędy zaokrągleń i drobne zaburzenia danych.\nTo prowadzi do niestabilności w rozwiązaniach, co widzimy na przykłądzie - minimalna zmiana b2 spowodowała ogromny skok w wartościach x1 i x2.")
