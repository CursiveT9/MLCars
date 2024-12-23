import numpy as np
import matplotlib.pyplot as plt

# 1. Нормализация по среднему и стандартному отклонению (z-score normalization)
def z_score_normalization(X):
    mean, std = manual_mean_and_std(X)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# 2. Мин-макс нормализация
def min_max_normalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm, X_min, X_max

# 3. L1-нормализацию
def l1_normalization(X):
    norm = np.sum(np.abs(X), axis=0)  # Сумма абсолютных значений по каждому столбцу
    X_norm = X / norm  # Делим элементы на норму
    return X_norm, norm

def manual_mean_and_std(X):
    n = len(X)  # Количество элементов
    mean = np.sum(X) / n  # Среднее
    std_dev = np.sqrt(np.sum((X - mean) ** 2) / n) # Стандартная
    return mean, std_dev

def plot_normalization(X_original, X_norm1, X_norm2, X_norm3):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].scatter(X_original[:, 0], X_original[:, 1], c='blue')
    axs[0, 0].set_title('Исходные признаки')

    axs[0, 1].scatter(X_norm1[:, 0], X_norm1[:, 1], c='green')
    axs[0, 1].set_title('Z-score нормализация')

    axs[1, 0].scatter(X_norm2[:, 0], X_norm2[:, 1], c='red')
    axs[1, 0].set_title('Мин-Макс нормализация')

    axs[1, 1].scatter(X_norm3[:, 0], X_norm3[:, 1], c='purple')
    axs[1, 1].set_title('L1 нормализация')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]  # Переменные
    y = data[:, 2]    # Целевая переменная
    m = len(y)

    X_zscore, mean_z, std_z = z_score_normalization(X)
    X_minmax, X_min, X_max = min_max_normalization(X)
    X_vector, norm_vector = l1_normalization(X)

    plot_normalization(X, X_zscore, X_minmax, X_vector)

    # Подсчет среднего и стандартного отклонения явно
    mean_feature, std_feature = manual_mean_and_std(X[:, 0])

    print(f"Среднее значение: {mean_feature}, Среднеквадратичное отклонение: {std_feature}")
