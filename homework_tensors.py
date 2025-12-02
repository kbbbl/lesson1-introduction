import torch


# 1.1 Создание тензоров
# 1. Тензор 3x4, случайные числа 0-1
t1 = torch.rand(3, 4)
print(t1)

# 2. Тензор 2x3x4, нули
t2 = torch.zeros(2, 3, 4)
print(t2)

# 3. Тензор 5x5, единицы
t3 = torch.ones(5, 5)
print(t3)

# 4. Тензор 4x4, числа 0-15
t4 = torch.arange(16).reshape(4, 4)
print(t4)
print("\n" + "=" * 60)


# 1.2 Операции с тензорами
# Создаём тензоры A и B
A = torch.randn(3, 4)
B = torch.randn(4, 3)

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
A_BT = A * B.T
print(A_BT)

# 4. Сумма элементов
sum_A = A.sum()
print(sum_A)
print("=" * 60)


# 1.3 Индексация и срезы
# Создаём тензор 5x5x5
tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)
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
print("=" * 60)


# 1.4 Работа с формами
# Создаём тензор из 24 элементов
tensor_24 = torch.arange(24)

# Разные формы
# Форма 2x12
t_2x12 = tensor_24.reshape(2, 12)
print(t_2x12)

# Форма 2x12
t_3x8 = tensor_24.reshape(3, 8)
print(t_3x8)

# Форма 4x6
t_4x6 = tensor_24.reshape(4, 6)
print(t_4x6)

# Форма 2x3x4
t_2x3x4 = tensor_24.reshape(2, 3, 4)
print(t_2x3x4)

# Форма 2x2x2x3
t_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
print(t_2x2x2x3)
print("=" * 60)