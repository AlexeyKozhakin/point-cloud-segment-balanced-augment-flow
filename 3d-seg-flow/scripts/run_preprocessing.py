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