import numpy as np
import matplotlib.pyplot as plt

eps = 1e-6
kmax = 20
N = 400

def newton_method(F, J, x0, roots):
    x = x0.copy()
    for k in range(kmax):
        try:
            delta = np.linalg.solve(J(x[0], x[1]), F(x[0], x[1]))
        except np.linalg.LinAlgError:
            break
        x = x - delta
        for i, r in enumerate(roots):
            if np.linalg.norm(x - r) < eps:
                return i
    return len(roots)

def plot_convergence(F, J, roots, region, title):
    x_vals = np.linspace(region[0], region[1], N)
    y_vals = np.linspace(region[2], region[3], N)
    colors = np.zeros((N, N))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            idx = newton_method(F, J, np.array([x, y], dtype=float), roots)
            colors[j, i] = idx

    plt.figure(figsize=(6, 5))
    plt.imshow(colors, extent=region, origin='lower', cmap='tab10')
    plt.title(f"Mapa de Convergência - {title}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Raiz atingida")
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()


# Sistema 1
def F1(x, y):
    return np.array([x**2 - y**2 - 1, 2*x*y])

def J1(x, y):
    return np.array([[2*x, -2*y], [2*y, 2*x]])

roots1 = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
region1 = (-1, 1, -1, 1)
plot_convergence(F1, J1, roots1, region1, "Sistema 1")

# Sistema 2
def F2(x, y):
    return np.array([x**3 - 3*x*y**2 - 1, 3*x**2*y - y**3])

def J2(x, y):
    return np.array([[3*x**2 - 3*y**2, -6*x*y],
                     [6*x*y, 3*x**2 - 3*y**2]])

roots2 = [
    np.array([1.0, 0.0]),
    np.array([-0.5, np.sqrt(3)/2]),
    np.array([-0.5, -np.sqrt(3)/2])
]
region2 = (-1.5, 1.5, -1.5, 1.5)
plot_convergence(F2, J2, roots2, region2, "Sistema 2")

# Sistema 3
def F3(x, y):
    return np.array([x**2 - y**2 - 1,
                     (x**2 + y**2 - 1)*(x**2 + y**2 - 2)])

def J3(x, y):
    return np.array([
        [2*x, -2*y],
        [4*x*(x**2 + y**2) - 6*x, 4*y*(x**2 + y**2) - 6*y]
    ])

roots3 = [
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([1.58, 1.22]),
    np.array([1.58, -1.22]),
    np.array([-1.58, 1.22]),
    np.array([-1.58, -1.22])
]
region3 = (-2, 2, -2, 2)
plot_convergence(F3, J3, roots3, region3, "Sistema 3")
