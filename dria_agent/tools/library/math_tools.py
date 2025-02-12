from dria_agent.tools.tool import tool

try:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVC
except ImportError:
    raise ImportError("Please install 'dria-agents[math_tools]")


@tool
def compute_pca(data: np.ndarray, n_components: int) -> tuple:
    """
    PCA on data.
    :param data: 2D array.
    :param n_components: Number of components.
    :return: (transformed data, explained variance ratio)
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return transformed, pca.explained_variance_ratio_


@tool
def eigen_decomposition(matrix: np.ndarray) -> tuple:
    """
    Eigen decomposition.
    :param matrix: Square array.
    :return: (eigenvalues, eigenvectors)
    """
    return np.linalg.eig(matrix)


@tool
def linear_regression_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """
    Fit linear regression and predict.
    :param X_train: Training features.
    :param y_train: Training targets.
    :param X_test: Test features.
    :return: Predictions.
    """
    model = LinearRegression().fit(X_train, y_train)
    return model.predict(X_test)


@tool
def vectorized_simpson_integration(f: callable, a: float, b: float, n: int) -> float:
    """
    Simpson's rule integration.
    :param f: Function to integrate.
    :param a: Lower limit.
    :param b: Upper limit.
    :param n: Even number of subintervals.
    :return: Integral approximation.
    """
    if n % 2:
        raise ValueError("n must be even.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return S * h / 3


@tool
def fast_fourier_transform(signal: np.ndarray) -> np.ndarray:
    """
    Compute FFT of a signal.
    :param signal: 1D array.
    :return: FFT result.
    """
    return np.fft.fft(signal)


@tool
def svm_classification(X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    SVM classification.
    :param X: Training features.
    :param y: Training labels.
    :param X_test: Test features.
    :return: Predicted labels.
    """
    model = SVC().fit(X, y)
    return model.predict(X_test)


@tool
def calculate_triangle_area(a: int, b: int, c: int) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    :param a: Length of triangle side.
    :param b: Length of triangle side.
    :param c: Length of triangle side.
    :return: The area of the triangle.
    """
    s = (a + b + c) / 2
    area = s * (s - a) * (s - b) * (s - c)
    return (area + 1e-5) ** (1 / 2)


@tool
def calculate_triangle_area(a: int, b: int, c: int) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    :param a: Length of triangle side.
    :param b: Length of triangle side.
    :param c: Length of triangle side.
    :return: The area of the triangle.
    """
    s = (a + b + c) / 2
    area = s * (s - a) * (s - b) * (s - c)
    return (area + 1e-5) ** 0.5


@tool
def solve_quadratic(a: float, b: float, c: float) -> tuple:
    """
    Solve quadratic equation a*x^2 + b*x + c = 0.

    :param a: Coefficient of x^2.
    :param b: Coefficient of x.
    :param c: Constant term.
    :return: Tuple of two solutions (complex if necessary).
    """
    import cmath

    disc = b * b - 4 * a * c
    sqrt_disc = cmath.sqrt(disc)
    return ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a))


@tool
def compute_determinant(matrix: list) -> float:
    """
    Compute the determinant of a square matrix recursively.

    :param matrix: Square matrix as a list of lists.
    :return: Determinant of the matrix.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    det = 0
    for c in range(len(matrix)):
        submatrix = [row[:c] + row[c + 1 :] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * compute_determinant(submatrix)
    return det


@tool
def simpson_integration(f: callable, a: float, b: float, n: int) -> float:
    """
    Numerically integrate f from a to b using Simpson's rule with n subintervals.

    :param f: Function to integrate.
    :param a: Lower limit.
    :param b: Upper limit.
    :param n: Number of subintervals (must be even).
    :return: Approximated integral of f.
    """
    if n % 2 != 0:
        raise ValueError("n must be even.")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


MATH_TOOLS = [
    compute_pca,
    eigen_decomposition,
    linear_regression_predict,
    svm_classification,
    calculate_triangle_area,
    solve_quadratic,
    compute_determinant,
    simpson_integration,
]
