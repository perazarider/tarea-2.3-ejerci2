import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) - x / 2

def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iterations = []
    errors = []
    for _ in range(max_iter):
        c = (a + b) / 2
        iterations.append(c)
        errors.append(abs(func(c)))
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, iterations, errors
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iterations, errors


x0, x1, x2 = 0.0, 1.0, 2.0  
x_points = np.array([x0, x1, x2])
y_points = f(x_points)


x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]


root, iterations, errors = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)


plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = sin(x) - x/2", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.6f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices_sin.png")  
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(len(iterations)), iterations, marker='o', linestyle='-', color='purple', label='Aproximación de la raíz')
plt.xlabel("Iteraciones")
plt.ylabel("Valor de la raíz")
plt.title("Convergencia del método de bisección")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='orange', label='Error absoluto')
plt.xlabel("Iteraciones")
plt.ylabel("Error absoluto")
plt.title("Evolución del error en la búsqueda de la raíz")
plt.legend()
plt.grid(True)
plt.show()


print(f"Raíz aproximada usando interpolación: {root:.6f}")
for i, (x_val, err) in enumerate(zip(iterations, errors)):
    print(f"Iteración {i+1}: x = {x_val:.6f}, Error = {err:.6f}")
