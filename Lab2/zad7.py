import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Wczytanie i skalowanie obrazu do rozmiaru 640x400
image_path = "chill_guy.jpg"  # Ścieżka do pliku obrazu
img_rgb = Image.open(image_path).resize((640, 400))
img_gray = img_rgb.convert("L")  # Konwersja do odcieni szarości
img_matrix = np.array(img_gray) / 255.0  # Normalizacja do zakresu [0, 1]

# Rozkład SVD obrazu
U, S, Vt = np.linalg.svd(img_matrix, full_matrices=False)

# Funkcja do rekonstrukcji obrazu przy użyciu k wartości osobliwych
def reconstruct_image(U, S, Vt, k):
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# Rekonstrukcja obrazów
img_10 = reconstruct_image(U, S, Vt, 10)
img_20 = reconstruct_image(U, S, Vt, 20)
img_40 = reconstruct_image(U, S, Vt, 40)

# Wizualizacja wyników
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Oryginalny obraz")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_matrix, cmap='gray')
plt.title("Obraz w skali szarości")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_10, cmap='gray')
plt.title("Rekonstrukcja (10)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_20, cmap='gray')
plt.title("Rekonstrukcja (20)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_40, cmap='gray')
plt.title("Rekonstrukcja (40)")
plt.axis('off')

plt.tight_layout()
plt.show()
