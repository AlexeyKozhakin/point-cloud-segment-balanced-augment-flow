import os
import laspy
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from concurrent.futures import ProcessPoolExecutor
import time


# Функция Farthest Point Sampling (FPS)
def farthest_point_sampling(points, n_samples):
    num_points = points.shape[0]
    sampled_idx = np.zeros(n_samples, dtype=np.int32)
    sampled_idx[0] = np.random.randint(num_points)
    distances = pairwise_distances(points, points[[sampled_idx[0]]], metric='euclidean').squeeze()

    for i in range(1, n_samples):
        farthest_idx = np.argmax(distances)
        sampled_idx[i] = farthest_idx
        new_distances = pairwise_distances(points, points[[farthest_idx]], metric='euclidean').squeeze()
        distances = np.minimum(distances, new_distances)

    return points[sampled_idx], sampled_idx


# Функция для разбиения точек по сетке и выборки с FPS
def process_points_in_grid(las_data, grid_size=1, num_samples=1024):
    x = las_data.x
    y = las_data.y
    z = las_data.z
    classes = las_data.classification  # Классы точек в LAS файле

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    x_bins = np.arange(min_x, max_x + grid_size, grid_size)
    y_bins = np.arange(min_y, max_y + grid_size, grid_size)

    reduced_datasets = []
    class_datasets = []

    # Разбиваем точки по прямоугольникам
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (x >= x_bins[i]) & (x < x_bins[i + 1]) & (y >= y_bins[j]) & (y < y_bins[j + 1])
            selected_points = np.vstack([x[mask], y[mask], z[mask]]).T  # Формируем массив (N, 3)
            selected_classes = classes[mask]  # Классы точек

            if selected_points.shape[0] > 0:
                # 1. Вычисление центра масс
                center_of_mass = np.mean(selected_points, axis=0)

                # 2. Смещение точек
                selected_points -= center_of_mass

                if selected_points.shape[0] > num_samples:
                    # Если точек больше, применяем FPS
                    sampled_points, sampled_idx = farthest_point_sampling(selected_points, num_samples)
                    sampled_classes = selected_classes[sampled_idx]  # Выбираем соответствующие классы
                else:
                    # Если точек меньше 1024, нужно добавить точки
                    sampled_points = selected_points
                    sampled_classes = selected_classes

                    # Если точек меньше 1024, дополним недостающие точки
                    if selected_points.shape[0] < num_samples:
                        additional_points, _ = farthest_point_sampling(selected_points,
                                                                       num_samples - selected_points.shape[0])
                        sampled_points = np.vstack([selected_points, additional_points])

                        # K-ближайшие соседи для дополнения классов
                        knn = KNeighborsClassifier(n_neighbors=3)  # Количество ближайших соседей
                        knn.fit(selected_points, selected_classes)
                        additional_classes = knn.predict(additional_points)

                        sampled_classes = np.hstack([selected_classes, additional_classes])

                # Сохраняем выбранные точки и их классы
                reduced_datasets.append(sampled_points)
                class_datasets.append(sampled_classes[:, np.newaxis])  # Добавляем ось для (N, 1024, 1)

    return np.array(reduced_datasets), np.array(class_datasets)


# Функция обработки одного файла LAS
def process_single_las_file(file_path, output_dir, grid_size=1, num_samples=1024):
    las = laspy.read(file_path)
    reduced_datasets, class_datasets = process_points_in_grid(las, grid_size, num_samples)

    # Создаем имя файла на основе исходного пути
    file_name = os.path.basename(file_path).replace('.las', '')

    # Сохраняем точки и классы в отдельные numpy файлы
    np.save(os.path.join(output_dir, f'{file_name}_x.npy'), reduced_datasets)
    np.save(os.path.join(output_dir, f'{file_name}_y.npy'), class_datasets)

    print(f'Датасеты для {file_name} сохранены. Размеры:')
    print(f'x: {reduced_datasets.shape}, y: {class_datasets.shape}')


# Основная функция для обработки всех файлов LAS в каталоге
def process_all_las_files(input_dir, output_dir, grid_size=1, num_samples=1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Получаем список всех файлов .las в указанном каталоге
    las_files = [f for f in os.listdir(input_dir) if f.endswith('.las')]
    full_file_paths = [os.path.join(input_dir, f) for f in las_files]

    # Используем ProcessPoolExecutor для параллельной обработки файлов
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_las_file, file_path, output_dir, grid_size, num_samples) for file_path
                   in full_file_paths]

        # Ждем завершения всех задач
        for future in futures:
            future.result()


if __name__ == '__main__':
    # Пример использования
    input_directory = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
                       r'\data\las_org\data_las_stpls3d\LosAngeles_test_parallel')  # Путь к каталогу с файлами LAS
    output_directory = (r'C:\Users\alexe\PycharmProjects\lidar-to-cnn-preprocessing'
                        r'\data\las_org\data_las_stpls3d\Processed')  # Путь к каталогу для сохранения результатов
    grid_size = 5  # Размер квадрата
    num_samples = 1024  # Количество точек в каждом прямоугольнике

    start = time.time()
    process_all_las_files(input_directory, output_directory, grid_size, num_samples)
    end = time.time()
    print(f'Обработка завершена. Время: {end - start:.2f} секунд')