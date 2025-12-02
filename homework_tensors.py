import torch

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# 1.1 Создание тензоров
# 1. Тензор 3x4, случайные числа 0-1
t1 = torch.rand(3, 4, device=device)
print(t1)

# 2. Тензор 2x3x4, нули
t2 = torch.zeros(2, 3, 4, device=device)
print(t2)

# 3. Тензор 5x5, единицы
t3 = torch.ones(5, 5, device=device)
print(t3)

# 4. Тензор 4x4, числа 0-15
t4 = torch.arange(16, device=device).reshape(4, 4)
print(t4)
print("\n" + "=" * 60)

# 1.2 Операции с тензорами
# Создаём тензоры A и B
try:
    A = torch.randn(3, 4, device=device)
    B = torch.randn(4, 3, device=device)

    # Проверка размерностей для умножения
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Несовместимые размеры для умножения: {A.shape} и {B.shape}")

    # Исходный тензор A (3x4) и B (4x3)
    print(A)
    print(B)

    # 1. Транспонирование
    A_T = A.T
    print(A_T)

    # 2. Матричное умножение
    AB = torch.matmul(A, B)
    print(AB)

    # 3. Поэлементное умножение
    # Проверка одинаковой формы для поэлементного умножения
    if A.shape != B.T.shape:
        print("Внимание: разные формы для поэлементного умножения")
    A_BT = A * B.T
    print(A_BT)

    # 4. Сумма элементов
    sum_A = A.sum()
    print(sum_A)

except Exception as e:
    print(f"Ошибка в операции с тензорами: {e}")

print("=" * 60)

# 1.3 Индексация и срезы
try:
    # Создаём тензор 5x5x5
    tensor_5x5x5 = torch.arange(125, device=device).reshape(5, 5, 5)
    print(tensor_5x5x5[:2])

    # 1. Первая строка (первый слой)
    first_row = tensor_5x5x5[0, :, :]
    print(first_row)

    # 2. Последний столбец (по всем слоям)
    last_col = tensor_5x5x5[:, :, -1]
    print(last_col)

    # 3. Подматрица 2x2 из центра
    center_sub = tensor_5x5x5[2:4, 2:4, 2:4]
    print(center_sub)

    # 4. Элементы с чётными индексами
    even_indices = tensor_5x5x5[::2, ::2, ::2]
    print(even_indices)

except Exception as e:
    print(f"Ошибка при индексации: {e}")

print("=" * 60)

# 1.4 Работа с формами
try:
    # Создаём тензор из 24 элементов
    tensor_24 = torch.arange(24, device=device)

    # Проверка что 24 элемента можно разложить на указанные формы
    total_elements = tensor_24.numel()

    # Разные формы
    # Форма 2x12
    if 2 * 12 == total_elements:
        t_2x12 = tensor_24.reshape(2, 12)
        print(t_2x12)

    # Форма 3x8
    if 3 * 8 == total_elements:
        t_3x8 = tensor_24.reshape(3, 8)
        print(t_3x8)

    # Форма 4x6
    if 4 * 6 == total_elements:
        t_4x6 = tensor_24.reshape(4, 6)
        print(t_4x6)

    # Форма 2x3x4
    if 2 * 3 * 4 == total_elements:
        t_2x3x4 = tensor_24.reshape(2, 3, 4)
        print(t_2x3x4)

    # Форма 2x2x2x3
    if 2 * 2 * 2 * 3 == total_elements:
        t_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
        print(t_2x2x2x3)

except Exception as e:
    print(f"Ошибка при работе с формами: {e}")

print("=" * 60)
