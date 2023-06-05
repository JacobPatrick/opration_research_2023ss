import math
import multiprocessing

import hppfcl
import numpy as np
import scipy.optimize as opt
import sympy as sp
# import sympy.vector

multiprocessing.freeze_support()

t: sp.Symbol = sp.symbols('t')
C = sp.vector.CoordSys3D('C')
Scalar = float | sp.Symbol


def vec2(x: Scalar, y: Scalar) -> sp.vector.Vector:
    return x * C.i + y * C.j


def pr(var: sp.Symbol, name: str = ''):
    if name == '':
        try:
            name = var.name
        except:
            pass
    if name != '':
        print(name + '=')
    sp.pretty_print(var)


d = 2.8  # 轴距
w = 2  # 车长
h = 5  # 车宽

p0x, p0y = (0, 0)  # 初始坐标
v0x, v0y = (0, 0)  # 初速度
p1x, p1y = (20, 0)  # 终点坐标
targetPoint = vec2(p1x, p1y)  # 终点坐标

epsilon = 1e-5

# 给定的参数

# 一些优化控制参数
Tmax = 5
argrange = [0, 20]
N = 0  # 控制点个数
D = Tmax * 2  # 三角函数周期s
x0i = 1  # 初始参数

## 多项式系数
Xp = sp.symarray('Xp', N + 1)
Yp = sp.symarray('Yp', N + 1)


def tarlor(X, t):
    c = [x / math.factorial(i) for x, i in zip(X, range(len(X)))]
    return np.polynomial.Polynomial(c)(t)


def furior(X, t):
    items = [X[n] * sp.sin(2 * sp.pi * (n + 1) * t / D) for n in range(1, len(X))]
    pp = sum(items) + X[0]
    return pp


def lagrange(X, t):
    step = Tmax / (len(X) - 1)
    Y = [i * step for i in range(len(X))]
    return lagrange_fun(Y, X, t)


def lagrange_fun(x, y, t):
    n = len(x)
    s = 0
    for k in range(n):
        la = y[k]
        for j in range(k):
            la = la * (t - x[j]) / (x[k] - x[j])
        for j in range(k + 1, n):
            la = la * (t - x[j]) / (x[k] - x[j])
        s = s + la
    return s


# 位置
x: sp.Symbol = lagrange([p0x] + list(Xp) + [p1x], t)
y: sp.Symbol = lagrange([p0y] + list(Yp) + [p1y], t)
p = vec2(x, y)

# 速度
vx: sp.Symbol = x.diff(t)
vy: sp.Symbol = y.diff(t)
v = vec2(vx, vy).magnitude()

# 加速度
ax: sp.Symbol = vx.diff(t)

ay: sp.Symbol = vy.diff(t)
a = vec2(ax, ay).magnitude()

# 角度
alpha = sp.acos(vec2(vx, vy).dot(vec2(1, 0)) / v)
omega = alpha.diff(t)

# 方向盘转角
theta = 16 * sp.atan(d / sp.sqrt(v ** 2 / omega ** 2 - d ** 2 / 4))
r = sp.sqrt(d ** 2 / 4 + (d / (sp.tan(theta / 16))) ** 2)
tmp = d / sp.tan(theta / 16)
angle_acc_max = a / r - math.radians(400) * tmp * sp.sec(theta / 16) ** 2 / 16 / (d ** 2 / 4 + tmp ** 2) ** 1.5

# 碰撞
car_box = hppfcl.CollisionObject(hppfcl.Box(w, h, 1))
wall_box = hppfcl.CollisionObject(hppfcl.Box(2, 8, 1))
wall_box.setTransform(hppfcl.Transform3f(np.array([7, 3, 0])))


def car_wall_distance(x, y, angle):
    pos = np.array([x, y, 0])
    car_box.setTransform(hppfcl.Transform3f(np.array(pos)))
    car_box.setRotation(hppfcl.AngleAxis(angle, np.array([0, 0, 1])).toRotationMatrix())
    req = hppfcl.DistanceRequest()
    result = hppfcl.DistanceResult()
    hppfcl.distance(car_box, wall_box, req, result)
    # print(result.min_distance)
    return result.min_distance


def subs_xy(Var):
    subs = {}
    subs.update({Xp[i]: Var[i] for i in range(N + 1)})
    subs.update({Yp[i]: Var[N + 1 + i] for i in range(N + 1)})
    return subs


def subs_xyT(Var):
    subs = {t: Tmax}
    subs.update({Xp[i]: Var[i] for i in range(N + 1)})
    subs.update({Yp[i]: Var[N + 1 + i] for i in range(N + 1)})
    return subs


def f_min(f, T: float, tol=0.1):
    result = opt.minimize_scalar(f, method='bounded', bounds=(0, float(T)),
                                 options={'xatol': tol, 'disp': 1})
    if not result.success:
        print('expr_min failed: ' + result.message)
        return 1e10
    val = f(result.x)
    return val


def expr_min(expr, Var):
    expr = expr.subs(subs_xy(Var))
    f = lambda t0: float(expr.subs({t: t0}))
    return f_min(f, Tmax)


def expr_max(expr, Var):
    return -expr_min(-expr, Var)


def expr_in_range(expr, m, M):
    return [
        opt.NonlinearConstraint(
            lambda Var: expr_min(expr, Var),
            lb=m, ub=M,
        ),
        opt.NonlinearConstraint(
            lambda Var: expr_max(expr, Var),
            lb=m, ub=M,
        ),
    ]


def wall_dis_at_t(Var, t0):
    xx = x.subs(subs_xy(Var)).subs({t: t0})
    yy = y.subs(subs_xy(Var)).subs({t: t0})
    aa_last = alpha.subs(subs_xy(Var)).subs({t: t0 - epsilon})
    aa = alpha.subs(subs_xy(Var)).subs({t: t0})

    d = car_wall_distance(
        float(xx),
        float(yy),
        float(aa) if aa.is_Float else float(aa_last)  # 如果当前时刻停下了，就用上一时刻的角度
    )
    # print(f'dis at ({xx:.2f},{yy:.2f},{aa:.2f},{t0:.2f})={d}')
    return d


constraints = [
    *expr_in_range(v, -3, 5),
    *expr_in_range(a, -5, 3),
    opt.NonlinearConstraint(
        lambda Var: f_min(lambda t0: wall_dis_at_t(Var, t0), Tmax, tol=0.03),
        lb=0.3, ub=1e10,
    ),
]

bestc = 1e10


def cost(Var):
    global bestc
    c = expr_min(a, Var)
    if c < bestc:
        bestc = c
        print(Var)
        print(c)
    return float(c)


# T0, Xp[0..N], Yp[0..N]
x0 = [x0i, ] * 2 * (N + 1)
bounds = [argrange, ] * (2 * (N + 1))

Method = 3

print('begin minimize')
if Method == 1:
    res = opt.minimize(cost,
                       x0=np.array(x0),
                       bounds=bounds,
                       constraints=constraints,
                       options={'disp': True}
                       )
if Method == 2:
    res = opt.minimize(cost,
                       method='trust-constr',
                       x0=np.array(x0),
                       bounds=bounds,
                       constraints=constraints,
                       options={'disp': 2}
                       )
if Method == 3:
    res = opt.differential_evolution(cost,
                                     x0=np.array(x0),
                                     bounds=bounds,
                                     popsize=100,
                                     maxiter=2,
                                     constraints=constraints,
                                     strategy='rand1bin',
                                     tol=0.01,
                                     polish=True,
                                     init='random',
                                     disp=True,
                                     )

print(res)
print(res.x)


def show_xy_graph(Var):
    step = 0.1
    T = Tmax
    Ts = [i * step for i in range(int(T / step))]
    Xs = [float(x.subs(subs_xy(Var)).subs({t: t0})) for t0 in Ts]
    Ys = [float(y.subs(subs_xy(Var)).subs({t: t0})) for t0 in Ts]
    import matplotlib.pyplot as plt
    # plt.plot(Ts, Xs)
    # plt.plot(Ts, Ys)
    plt.plot(Xs, Ys)
    plt.show()


# show_xy_graph(x0)
show_xy_graph(res.x)
