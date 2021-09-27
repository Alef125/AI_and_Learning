import numpy as np
import matplotlib.pyplot as plt


# ***************************************** Function Implementation *****************************************
# Rastrigin Function
def rastrigin(x1, x2):
    f = 20 + np.power(x1, 2) + np.power(x2, 2) - 10 * np.cos(2 * np.pi * x1) - 10 * np.cos(2 * np.pi * x1)
    return f


# Ackley Function
def ackley(x1, x2):
    f = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(0.5 * (np.power(x1, 2) + np.power(x2, 2)))) \
        - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return f


# Levi Function
def levi(x1, x2):
    f = np.power(np.sin(3 * np.pi * x1), 2) + np.power(x1 - 1, 2) * (1 + np.power(np.sin(3 * np.pi * x2), 2)) \
        + np.power(x2 - 1, 2) * (1 + np.power(np.sin(2 * np.pi * x2), 2))
    return f


# Bukin Function
def bukin(x1, x2):
    f = 100 * np.sqrt(np.abs(x2 - 0.01 * np.power(x1, 2)))
    return f
# ***********************************************************************************************************


# ***************************************** Gradient Implementation *****************************************
# Rastrigin Gradient
def rastrigin_grad(x1, x2):
    g1 = 2 * x1 + 20 * np.pi * np.sin(2 * np.pi * x1)
    g2 = 2 * x2 + 20 * np.pi * np.sin(2 * np.pi * x2)
    return g1, g2


# Ackley Gradient
def ackley_grad(x1, x2):
    r = np.sqrt(0.5 * (np.power(x1, 2) + np.power(x2, 2)))
    e_cos = np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    g1 = 2 * x1 * np.exp(-0.2 * r) / r + np.pi * np.sin(2 * np.pi * x1) * e_cos
    g2 = 2 * x2 * np.exp(-0.2 * r) / r + np.pi * np.sin(2 * np.pi * x2) * e_cos
    return g1, g2


# Levi Gradient
def levi_grad(x1, x2):
    g1 = 6 * np.pi * np.sin(3 * np.pi * x1) * np.cos(3 * np.pi * x1) \
         + 2 * (x1 - 1) * (1 + np.power(np.sin(3 * np.pi * x2), 2))
    g2 = 6 * np.pi * np.power((x1 - 1), 2) * np.sin(3 * np.pi * x1) * np.cos(3 * np.pi * x1) \
        + 2 * (x2 - 1) * (1 + np.power(np.sin(2 * np.pi * x2), 2)) \
        + 4 * np.pi * np.power((x2 - 1), 2) * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x1)
    return g1, g2


# Bukin Gradient
def bukin_grad(x1, x2):
    sign_rad = np.sign(x2 - 0.01 * np.power(x1, 2)) / np.sqrt(np.abs(x2 - 0.01 * np.power(x1, 2)))
    g1 = 0.01 * np.sign(x1 + 10) - 0.5 * x1 * sign_rad
    g2 = 50 * sign_rad
    return g1, g2


# n-D Rastrigin Gradient
def n_d_rastrigin_grad(x):
    g = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    return g
# ***********************************************************************************************************


# **************************************** Optimizers Implementation ****************************************
# Point Class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy


# Simple Gradient Descend
class SimpleGradientDescend:
    def __init__(self, grad_func, init_point):
        self.eta = 0.001
        self.grad_func = grad_func
        self.position = init_point
        self.movements = []
        self.all_positions = []
        self.all_positions.append(init_point)

    def update(self):
        dx, dy = self.grad_func(self.position.x, self.position.y)
        self.position.move(-self.eta * dx, -self.eta * dy)
        self.movements.append(np.sqrt(np.power(self.eta * dx, 2) + np.power(self.eta * dy, 2)))
        self.all_positions.append(Point(self.position.x, self.position.y))


# Nesterov Accelerated Gradient
class NAG:
    def __init__(self, grad_func, init_point):
        self.gamma = 0.9
        self.eta = 0.001
        self.velocity = Point(0, 0)
        self.grad_func = grad_func
        self.position = init_point
        self.movements = []
        self.all_positions = []
        self.all_positions.append(init_point)

    def update(self):
        moved_x = self.position.x - self.gamma * self.velocity.x
        moved_y = self.position.y - self.gamma * self.velocity.y
        gx, gy = self.grad_func(moved_x, moved_y)
        self.velocity = Point(self.gamma * self.velocity.x + self.eta * gx,
                              self.gamma * self.velocity.y + self.eta * gy)
        self.position.move(-self.velocity.x, -self.velocity.y)
        self.movements.append(np.sqrt(np.power(self.velocity.x, 2) + np.power(self.velocity.y, 2)))
        self.all_positions.append(Point(self.position.x, self.position.y))


# RMSprop
class RMSprop:
    def __init__(self, grad_func, init_point):
        self.eta = 0.001
        self.epsilon = 0.00000001
        self.rho = 0.9
        self.r = Point(0, 0)
        self.grad_func = grad_func
        self.position = init_point
        self.movements = []
        self.all_positions = []
        self.all_positions.append(init_point)

    def update(self):
        gx, gy = self.grad_func(self.position.x, self.position.y)
        ggx = gx * gx
        ggy = gy * gy
        self.r = Point(self.rho * self.r.x + (1 - self.rho) * ggx, self.rho * self.r.y + (1 - self.rho) * ggy)
        delta_move = Point(self.eta * gx / np.sqrt(self.epsilon + self.r.x),
                           self.eta * gy / np.sqrt(self.epsilon + self.r.y))
        self.position.move(-delta_move.x, -delta_move.y)
        self.movements.append(np.sqrt(np.power(delta_move.x, 2) + np.power(delta_move.y, 2)))
        self.all_positions.append(Point(self.position.x, self.position.y))


# Adam
class Adam:
    def __init__(self, grad_func, init_point):
        self.eta = 0.001
        self.epsilon = 0.00000001
        self.b1 = 0.9
        self.b2 = 0.999
        self.s = Point(0, 0)
        self.r = Point(0, 0)
        self.grad_func = grad_func
        self.position = init_point
        self.t = 0
        self.movements = []
        self.all_positions = []
        self.all_positions.append(init_point)

    def update(self):
        gx, gy = self.grad_func(self.position.x, self.position.y)
        ggx = gx * gx
        ggy = gy * gy
        self.s = Point(self.b1 * self.s.x + (1 - self.b1) * gx, self.b1 * self.s.x + (1 - self.b1) * gy)
        self.r = Point(self.b2 * self.r.x + (1 - self.b2) * ggx, self.b2 * self.r.x + (1 - self.b2) * ggy)
        s_hat = Point(self.s.x / (1 - np.power(self.b1, self.t + 1)), self.s.y / (1 - np.power(self.b1, self.t + 1)))
        r_hat = Point(self.r.x / (1 - np.power(self.b2, self.t + 1)), self.r.y / (1 - np.power(self.b2, self.t + 1)))
        delta_move = Point(self.eta * s_hat.x / np.sqrt(self.epsilon + r_hat.x),
                           self.eta * s_hat.y / np.sqrt(self.epsilon + r_hat.y))
        self.position.move(-delta_move.x, -delta_move.y)
        self.t = self.t + 1
        self.movements.append(np.sqrt(np.power(delta_move.x, 2) + np.power(delta_move.y, 2)))
        self.all_positions.append(Point(self.position.x, self.position.y))
# ***********************************************************************************************************


# Plotting Function
def plot_movements(movements):
    x = []
    y = []
    for p in movements:
        x.append(p.x)
        y.append(p.y)
    plt.scatter(x, y)
    plt.show()


# **************************************  Main ****************************************
grad = bukin_grad
start_point = Point(0.4, 0.4)

# **************  Fixing Eta in GD  ***************
# gd1 = SimpleGradientDescend(grad, start_point)
# gd2 = SimpleGradientDescend(grad, start_point)
# gd1.eta = 0.001
# gd2.eta = 0.0001
# for i in range(50):
#     gd1.update()
#     gd2.update()
# p1, = plt.plot(gd1.movements, label='rate = 0.01')
# p2, = plt.plot(gd2.movements, label='rate = 0.0001')
# plt.legend(handles=[p1])
# plt.xlabel('t')
# plt.ylabel('|dx|')
# plt.title("Rastrigid Function")
# plt.show()
# print(gd1.all_positions[50].x, gd1.all_positions[50].y)
# print(gd2.all_positions[50].x, gd2.all_positions[50].y)

# **************  Fixing Eta & Gamma in NAg **************
# nag1 = NAG(grad, start_point)
# nag2 = NAG(grad, start_point)
# nag3 = NAG(grad, start_point)
# nag1.eta = 0.01
# nag2.eta = 0.001
# nag3.eta = 0.0001
# for i in range(50):
#     nag1.update()
#     nag2.update()
#     nag3.update()
# p1, = plt.plot(nag1.movements, label='gamma = 0.7')
# p2, = plt.plot(nag2.movements, label='rate = 0.001')
# p3, = plt.plot(nag3.movements, label='rate = 0.0001')
# plt.legend(handles=[p1])
# plt.xlabel('t')
# plt.ylabel('|dx|')
# plt.title("Rastrigin Function")
# plt.show()
# print(nag1.all_positions[50].x, nag1.all_positions[50].y)
# print(nag2.all_positions[50].x, nag2.all_positions[50].y)
# print(nag3.all_positions[50].x, nag3.all_positions[50].y)

gd = SimpleGradientDescend(grad, start_point)
nag = NAG(grad, start_point)
rmsprop = RMSprop(grad, start_point)
adam = Adam(grad, start_point)
for i in range(50):
    gd.update()
    nag.update()
    rmsprop.update()
    adam.update()
p1, = plt.plot(gd.movements, label='Gradient Descend')
p2, = plt.plot(nag.movements, label='Nesterov')
p3, = plt.plot(rmsprop.movements, label='RMSprop')
p4, = plt.plot(adam.movements, label='Adam')
plt.legend(handles=[p1, p2, p3, p4])
plt.xlabel('t')
plt.ylabel('|dx|')
plt.title("Bukin Function")
plt.show()
print 'Gradient Descend Last Point:', (gd.all_positions[50].x, gd.all_positions[50].y)
print 'Nesterov Last Point:',(nag.all_positions[50].x, nag.all_positions[50].y)
print 'RMSprop Last Point:',(rmsprop.all_positions[50].x, rmsprop.all_positions[50].y)
print 'Adam Last Point:',(adam.all_positions[50].x, adam.all_positions[50].y)
# plot_movements(gd.all_positions)
# ************************************ End *******************************************
