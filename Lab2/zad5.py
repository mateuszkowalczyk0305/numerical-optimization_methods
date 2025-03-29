import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ======================================
# Funkcja do wykreślania dysków Gerszgorina
# ======================================
def plot_gershgorin_disks(A, ax, title, centers, radii):
    """
    Rysuje dyski Gerszgorina dla zadanej macierzy A, korzystając z ręcznie policzonych środków i promieni.

    Parametry:
    -----------
    A : numpy.ndarray
        Macierz kwadratowa, dla której rysowane są dyski Gerszgorina.
    ax : matplotlib.axes.Axes
        Obiekt osi wykresu, na którym rysowane są dyski.
    title : str
        Tytuł wykresu.
    centers : list
        Lista środków dysków wyznaczonych ręcznie.
    radii : list
        Lista promieni dysków wyznaczonych ręcznie.
    """

    # Obliczanie wartości własnych za pomocą wbudowanej funkcji numpy
    eigenvalues = np.linalg.eigvals(A)

    # Ustawienia wykresu - tytuł, etykiety osi, siatka, osie symetrii
    ax.set_title(title)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.grid(True)
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)

    # Inicjalizacja maksymalnych wartości dla skalowania wykresu
    max_radius = max(radii)
    max_center = max(abs(center) for center in centers)

    # Rysowanie każdego dysku Gerszgorina
    for i, (center, radius) in enumerate(zip(centers, radii)):
        # Tworzenie dysku jako obiekt klasy Circle
        circle = Circle((center, 0), radius, color='blue', fill=True, alpha=0.2)

        # Dodawanie dysku do wykresu
        ax.add_patch(circle)

        # Rysowanie środka dysku (punktu) na wykresie
        ax.plot(center, 0, 'bo', label=f'Dysk {i + 1}: środek={center}, promień={radius}')

    # Rysowanie wartości własnych na wykresie jako czerwone krzyżyki
    ax.plot(np.real(eigenvalues), np.imag(eigenvalues), 'rx', markersize=10, label="Wartości własne")

    # Dodawanie legendy do wykresu
    ax.legend()

    # Dynamiczne ustawianie zakresów osi w zależności od największego dysku
    limit = max(max_center + max_radius, 6)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])


# ======================================
# Definiowanie macierzy, dla których będą rysowane dyski Gerszgorina
# ======================================
A1 = np.array([[-2, -1, 0], [2, 0, 0], [0, 0, 2]])
A2 = np.array([[5, 1, 1], [0, 6, 1], [0, 0, -5]])
A3 = np.array([[5.2, 0.6, 2.2], [0.6, 6.4, 0.5], [2.2, 0.5, 4.7]])

# Twoje ręcznie obliczone środki i promienie
centers_A1 = [-2, 0, 2]
radii_A1 = [1, 2, 0]

centers_A2 = [5, 6, -5]
radii_A2 = [2, 1, 0]

centers_A3 = [5.2, 6.4, 4.7]
radii_A3 = [2.8, 1.1, 2.7]

# ======================================
# Tworzenie układu wykresów (trzy wykresy w jednym wierszu)
# ======================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Rysowanie dysków dla każdej macierzy
plot_gershgorin_disks(A1, axs[0], r"Dyski Gerszgorina dla $A_1$", centers_A1, radii_A1)
plot_gershgorin_disks(A2, axs[1], r"Dyski Gerszgorina dla $A_2$", centers_A2, radii_A2)
plot_gershgorin_disks(A3, axs[2], r"Dyski Gerszgorina dla $A_3$", centers_A3, radii_A3)

# Automatyczne dostosowanie układu wykresów do okna
plt.tight_layout()

# Wyświetlenie wykresów
plt.show()

