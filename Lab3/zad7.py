import numpy as np
from scipy.linalg import toeplitz, svd
from numpy.linalg import norm, matrix_rank, cond
import matplotlib.pyplot as plt

# === Funkcje pomocnicze ===
def generate_data(N, x_type='A'):
    c = np.arange(N)
    r = np.concatenate([[0], -np.arange(1, N)])
    A = toeplitz(c, r)
    if x_type == 'A':
        x_star = np.arange(1, N + 1)
    elif x_type == 'B':
        x_star = np.random.normal(0, 1, N)
    b = A @ x_star
    return A, x_star, b

# === Tikhonow i TSVD regularyzacja ===
def tikhonov_solution(A, b, alpha):
    n = A.shape[1]
    return np.linalg.inv(A.T @ A + alpha * np.eye(n)) @ A.T @ b

def tsvd_solution(U, S, V, b, alpha):
    n = len(S)
    S_inv = np.diag([1 / (S[i] + alpha**2) for i in range(n)])
    return V @ S_inv @ U.T @ b

# === Analiza błędów ===
def analyze_errors(N=10):
    alpha_vals = np.linspace(0.0001, 1, 100)
    I = np.eye(N)

    for x_type in ['A', 'B']:
        A, x_star, b = generate_data(N, x_type)
        U, s, Vh = svd(A)
        V = Vh.T

        error_resi_tikh = []
        error_x_tikh = []
        error_resi_tsvd = []
        error_x_tsvd = []

        for alpha in alpha_vals:
            x_ls = tikhonov_solution(A, b, alpha)
            error_resi_tikh.append(norm(A @ x_ls - b))
            error_x_tikh.append(norm(x_ls))

            x_tsvd = tsvd_solution(U, s, V, b, alpha)
            error_resi_tsvd.append(norm(A @ x_tsvd - b))
            error_x_tsvd.append(norm(x_tsvd))

        # Znalezienie optymalnego alpha
        min_index_tikh = np.argmin(error_x_tikh)
        min_index_tsvd = np.argmin(error_x_tsvd)

        print(f"\nTyp danych ({x_type}):")
        print(f"Optymalna alfa (Tikhonow): {alpha_vals[min_index_tikh]:.5e}, Błąd rozwiązania: {error_x_tikh[min_index_tikh]:.5e}")
        print(f"Optymalna alfa (TSVD): {alpha_vals[min_index_tsvd]:.5e}, Błąd rozwiązania: {error_x_tsvd[min_index_tsvd]:.5e}")

        # Wykresy
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.semilogx(alpha_vals, error_x_tikh); plt.grid()
        plt.title(f'Błąd rozwiązania (Tikhonow) dla ({x_type})')

        plt.subplot(2, 2, 2)
        plt.semilogx(alpha_vals, error_resi_tikh); plt.grid()
        plt.title(f'Błąd residualny (Tikhonow) dla ({x_type})')

        plt.subplot(2, 2, 3)
        plt.semilogx(alpha_vals, error_x_tsvd); plt.grid()
        plt.title(f'Błąd rozwiązania (TSVD) dla ({x_type})')

        plt.subplot(2, 2, 4)
        plt.semilogx(alpha_vals, error_resi_tsvd); plt.grid()
        plt.title(f'Błąd residualny (TSVD) dla ({x_type})')

        plt.tight_layout()
        plt.show()

# === Rank i Cond(A) ===
def analyze_rank_cond():
    Ns = [5, 10, 50, 100]
    for N in Ns:
        A, _, _ = generate_data(N)
        print(f"N = {N}, Rank(A) = {matrix_rank(A)}, Cond(A) = {cond(A):.2e}")

# === Optymalne lambda vs N ===
def optimal_alpha_vs_N():
    Ns = [5, 10, 50, 100]
    lambda_values = np.logspace(-5, 5, 100)
    opt_A = []
    opt_B = []

    for N in Ns:
        c = np.arange(N)
        r = np.concatenate([[0], -np.arange(1, N)])
        A = toeplitz(c, r)

        x_star_A = np.arange(1, N + 1)
        b_A = A @ x_star_A

        x_star_B = np.random.randn(N)
        b_B = A @ x_star_B

        err_A = []
        err_B = []

        for lam in lambda_values:
            xA = np.linalg.inv(A.T @ A + lam * np.eye(N)) @ A.T @ b_A
            xB = np.linalg.inv(A.T @ A + lam * np.eye(N)) @ A.T @ b_B
            err_A.append(norm(xA - x_star_A))
            err_B.append(norm(xB - x_star_B))

        opt_A.append(lambda_values[np.argmin(err_A)])
        opt_B.append(lambda_values[np.argmin(err_B)])

    plt.figure()
    plt.semilogy(Ns, opt_A, '-o', label='(A) Deterministyczne')
    plt.semilogy(Ns, opt_B, '-s', label='(B) Rozkład normalny')
    plt.title('Optymalny parametr regularyzacji w funkcji wymiaru N')
    plt.xlabel('Wymiar N')
    plt.ylabel('Optymalny parametr regularyzacji')
    plt.legend()
    plt.grid(True)
    plt.show()

# === Uruchomienie wszystkich analiz ===
analyze_errors()
analyze_rank_cond()
optimal_alpha_vs_N()