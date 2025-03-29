import numpy as np

#Pomocnicza funkcja do wyświetlania macierzy z zaokrągleniem do 4 miejsc po przecinku.
def print_matrix(name, matrix):
    np.set_printoptions(precision=4, suppress=True) # ustawienie precyzji do 4 miejsc po przecinku
    print(f"{name} =\n{matrix}\n")



# Definiowanie macierzy z zadania
A1 = np.array([[1, 1], [0, 1], [1, 0]])
A2 = np.array([[-1, 2, 2]])  # Macierz jednowierszowa
A3 = np.array([
    [2, 2, 2],
    [17/10, 17/10, 17/10],
    [3/5, 3/5, 3/5]
])

# Obliczanie rozkładów SVD za pomocą numpy.linalg.svd
U1, S1, Vt1 = np.linalg.svd(A1)
U2, S2, Vt2 = np.linalg.svd(A2)
U3, S3, Vt3 = np.linalg.svd(A3)

# Konwersja wektorów wartości osobliwych do macierzy diagonalnych dla zgodności z Matlabem
S1_matrix = np.zeros((A1.shape[0], A1.shape[1]))
np.fill_diagonal(S1_matrix, S1)

S2_matrix = np.zeros((A2.shape[0], A2.shape[1]))
np.fill_diagonal(S2_matrix, S2)

S3_matrix = np.zeros((A3.shape[0], A3.shape[1]))
np.fill_diagonal(S3_matrix, S3)

# Wyświetlanie wyników z zaokrągleniem do 4 miejsc po przecinku
print("Rozkład SVD dla macierzy A1:")
print_matrix("U1", U1)
print_matrix("S1", S1_matrix)
print_matrix("Vt1", Vt1)

print("Rozkład SVD dla macierzy A2:")
print_matrix("U2", U2)
print_matrix("S2", S2_matrix)
print_matrix("Vt2", Vt2)

print("Rozkład SVD dla macierzy A3:")
print_matrix("U3", U3)
print_matrix("S3", S3_matrix)
print_matrix("Vt3", Vt3)
