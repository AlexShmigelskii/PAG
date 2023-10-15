import numpy as np
import matplotlib.pyplot as plt


def plot_result(X_table, Y_table, X_range, Y_approximated, degree, x_label, y_label):
    """Визуализация результатов"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_table, Y_table, label='Исходные данные', color='red')
    plt.plot(X_range, Y_approximated, label=f'Аппроксимация (степень {degree})', color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_results_close(X_table, Y_table, X_range, Y_approximated, poly_degree, y_label):
    """Визуализация результатов поближе"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X_table[(X_table >= 700) & (X_table <= 8000)], Y_table[(X_table >= 700) & (X_table <= 8000)],
                label='Исходные данные', color='red')
    plt.plot(X_range, Y_approximated, label=f'Аппроксимация (степень {poly_degree})', color='blue')
    plt.xlabel('Высота (м)')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_error(X_range, Y_approximated, data_dict):
    """Считает ошибки аппроксимированых и табличных данных"""
    results = [(x, y) for x, y in zip(X_range, Y_approximated)]

    # Посчитаем ошибки определенных ускорений и известных из исходных данных
    print('Значения ошибок:')
    for x, y in results:
        if x in np.arange(1000, 8000 + 1, 1000):
            print(f'Ошибка для высоты {x}: {data_dict[x] - y}')

            def least_squares_fit(x, y, degree, num_points=None):
                if num_points is None:
                    num_points = len(x)  # Используем все точки по умолчанию

                # Выбираем только первые num_points точек для аппроксимации
                x_subset = x[:num_points]
                y_subset = y[:num_points]

                # Создаем матрицу X со степенями x_subset
                X = np.vander(x_subset, degree + 1)

                # Решаем уравнение X^T * X * coeffs = X^T * y_subset для coeffs
                # Решение этой системы линейных уравнений производится для получения коэффициентов coeffs.
                coeffs = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_subset))

                return coeffs


def least_squares_fit(x, y, degree, num_points=None):
    if num_points is None:
        num_points = len(x)  # Используем все точки по умолчанию

    # Выбираем только первые num_points точек для аппроксимации
    x_subset = x[:num_points]
    y_subset = y[:num_points]

    # Создаем матрицу X со степенями x_subset
    X = np.vander(x_subset, degree + 1)

    # Решаем уравнение X^T * X * coeffs = X^T * y_subset для coeffs
    # Решение этой системы линейных уравнений производится для получения коэффициентов coeffs.
    coeffs = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_subset))

    return coeffs
