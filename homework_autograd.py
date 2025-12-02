import torch


# 2.1 Простые вычисления с градиентами
# Создаём тензоры с requires_grad=True
x = torch.tensor([[4.0, 6.0], [8.0, 10.0]], requires_grad=True)
y = torch.tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
z = torch.tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

# Вычисляем функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Вычисляем градиенты
f.sum().backward()

print("x.grad =", x.grad)
print("y.grad =", y.grad)
print("z.grad =", z.grad)

# Аналитическая проверка
print("2*x + 2*y*z =", 2*x + 2*y*z)
print("2*y + 2*x*z =", 2*y + 2*x*z)
print("2*z + 2*x*y =", 2*z + 2*x*y)


# 2.2 Градиент функции потерь (MSE)
# Синтетические данные
n, m = 5, 3
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0],
                  [13.0, 14.0, 15.0]])

y_true = torch.tensor([[2.0], [5.0], [8.0], [11.0], [14.0]])

# Параметры модели
w = torch.tensor([[0.5], [1.0], [1.5]], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# Прямой проход
y_pred = X @ w + b

# Функция потерь MSE
mse_loss = torch.mean((y_pred - y_true) ** 2)

# Обратное распространение
mse_loss.backward()

print("\nw.grad =", w.grad)
print("b.grad =", b.grad)

# Аналитическая проверка
print("(2/n) * X.T @ (X @ w + b - y_true) =", (2/n) * X.T @ (X @ w + b - y_true))
print("(2/n) * torch.sum(X @ w + b - y_true) =", (2/n) * torch.sum(X @ w + b - y_true))


# 2.3 Цепное правило
# Создаём тензор
x_chain = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]], requires_grad=True)

# Составная функция: f(x) = sin(x^2 + 1)
f_chain = torch.sin(x_chain**2 + 1)

# Вычисляем градиент с retain_graph=True
f_chain.sum().backward(retain_graph=True)

print("\nx.grad =", x_chain.grad)

# Проверка с помощью torch.autograd.grad
grad_check = torch.autograd.grad(outputs=f_chain.sum(), inputs=x_chain)[0]
print("torch.autograd.grad =", grad_check)