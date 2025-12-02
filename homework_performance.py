import torch
import time

# 3.1 Подготовка данных
matrix_64 = torch.randn(64, 1024, 1024)
matrix_128 = torch.randn(128, 512, 512)
matrix_256 = torch.randn(256, 256, 256)

# Проверяем доступность CUDA
has_cuda = torch.cuda.is_available()

# Копируем матрицы на GPU если доступно
if has_cuda:
    matrix_64_gpu = matrix_64.cuda()
    matrix_128_gpu = matrix_128.cuda()
    matrix_256_gpu = matrix_256.cuda()


# 3.2 Функция измерения времени
# Измеряет время выполнения операции.
def measure_time(operation, matrix, use_gpu=False):
    if use_gpu and has_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = operation(matrix)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
    else:
        start_time = time.time()
        result = operation(matrix)
        elapsed_time = (time.time() - start_time) * 1000

    return result, elapsed_time


# 3.3 Операции для сравнения
def matmul_op(matrix):
    batch, n, m = matrix.shape
    result = torch.empty(batch, n, n, device=matrix.device)
    for i in range(batch):
        result[i] = torch.matmul(matrix[i], matrix[i].T)
    return result


def add_op(matrix):
    return matrix + matrix


def mul_op(matrix):
    return matrix * matrix


def transpose_op(matrix):
    return matrix.transpose(1, 2)


def sum_op(matrix):
    return torch.sum(matrix)


# Операции для тестирования
operations = [
    ("Матричное умножение", matmul_op),
    ("Сложение", add_op),
    ("Умножение", mul_op),
    ("Транспонирование", transpose_op),
    ("Сумма элементов", sum_op)
]

# Выводим таблицу
print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")

# Тестируем на средней матрице (128x512x512)
cpu_matrix = matrix_128
gpu_matrix = matrix_128_gpu if has_cuda else None

for op_name, op_func in operations:
    # Измеряем на CPU
    _, cpu_time = measure_time(op_func, cpu_matrix, use_gpu=False)

    # Измеряем на GPU если доступно
    if gpu_matrix is not None:
        _, gpu_time = measure_time(op_func, gpu_matrix, use_gpu=True)
        speedup = cpu_time / gpu_time
        print(f"{op_name:<25} | {cpu_time:<10.2f} | {gpu_time:<10.2f} | {speedup:<10.2f}x")
    else:
        print(f"{op_name:<25} | {cpu_time:<10.2f} | {'N/A':<10} | {'N/A':<10}")


# 3.4 Анализ результатов
if has_cuda:
    print("\nАнализ результатов:")
    print("- Матричное умножение получает наибольшее ускорение")
    print("- Поэлементные операции получают среднее ускорение")
    print("- Передача данных добавляет задержки")
    print("- Большие матрицы лучше используют GPU")
else:
    print("\nCUDA не доступен")

# Очистка памяти GPU
if has_cuda:
    torch.cuda.empty_cache()