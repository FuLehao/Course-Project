import pandas as pd
import numpy as np
from sympy import *
from sympy import symbols, Matrix, diff
from cvxopt import solvers, matrix
import cvxopt
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
cvxopt.solvers.options['show_progress'] = False

# 仅以2维问题作为示例
x = symbols('x_0 x_1')


def plugin(func, variable):
    return func.subs(list(zip(x, variable)))


def linesearch(func, p, xk, gradient):
    c = 0.2
    alpha = 1
    Value_0 = plugin(func, xk)
    linearFactor = c * (gradient.T * p)[0]
    while true:
        new = plugin(func, xk + alpha * p)
        linear = Value_0 + alpha * linearFactor
        bottom = Value_0 + alpha * (1 - c) * (gradient.T * p)[0]
        if new <= linear and new >= bottom:
            return alpha
        elif new <= linear:
            alpha = alpha * 1.5
        else:
            alpha = 0.5 * alpha


def SQP(func, iters, start, eq_cons, ineq_cons, epsilon, iter_log):
    n_eq = len(eq_cons)
    n_ineq = len(ineq_cons)
    n_con = len(eq_cons) + len(ineq_cons)
    gradient = Matrix([diff(func, x_i) for x_i in x])
    xk = start
    # 把线性和非线性的约束写为矩阵形式并求其梯度
    c_eq = Matrix([eq_cons[i] for i in range(n_eq)])
    c_ineq = Matrix([ineq_cons[i] for i in range(n_ineq)])
    A_eq = Matrix([[diff(eq_cons[i], x_i) for x_i in x] for i in range(n_eq)])
    A_ineq = Matrix([[diff(ineq_cons[i], x_i) for x_i in x] for i in range(n_ineq)])
    Hessen = eye(len(x))

    for iter in range(iters):
        msg = '第{}次迭代：当前迭代点  x0 = {:.3}, x1 = {:.3f} | 迭代点目标函数值: {:.3f}'
        print(msg.format(iter, float(xk[0]), float(xk[1]), float(plugin(func, xk))))
        if iter >= 1:
            iter_log.loc[len(iter_log.index)] = [iter, round(float(xk[0]), 3), round(float(xk[1]), 3),
                                                 round(float(plugin(func, xk)), 3)]

        W = plugin(Hessen, xk)
        g = plugin(gradient, xk)
        A1 = plugin(A_eq, xk)
        c1 = plugin(c_eq, xk)
        A2 = plugin(A_ineq, xk)
        c2 = plugin(c_ineq, xk)

        P = matrix([[float(W[0, 0]), float(W[0, 1])], [float(W[1, 0]), float(W[1, 1])]])
        Q = matrix([float(g[0]), float(g[1])])
        A = matrix([[float(A1[i * 2 + j]) for j in range(2)] for i in range(n_eq)]).T
        G = matrix([[float(A2[i * 2 + j]) for j in range(2)] for i in range(n_ineq)]).T
        b = matrix([-float(c1[i]) for i in range(n_eq)])
        h = matrix([-float(c2[i]) for i in range(n_ineq)])

        # 调用已知库求解QP子问题
        sol = solvers.qp(P, Q, G, h, A, b)
        solution = sol['x']
        d = Matrix([solution[0], solution[1]])

        # 对alpha采用线搜索策略
        alpha = linesearch(func, d, xk, g)
        xk_next = xk + alpha * d
        if (d).norm(None) <= epsilon:
            print("一共迭代了{}次".format(iter))
            return xk, iter_log

        # 迭代计算Hessen
        s = xk_next - xk
        gradientX_next = plugin(gradient, xk_next)
        y = gradientX_next - g
        Hs = Hessen * s
        Hessen = Hessen + (y * y.T) / (y.T * s)[0] - Hs * Hs.T / (s.T * Hs)[0]
        xk = xk_next.evalf()
        Hessen = Hessen.evalf()
    print("一共迭代了{}次, 未找到最优值，请尝试更改迭代次数iter和容忍度epsilon".format(iter + 1))
    return xk, iter_log


def draw(iter_log):
    iter_seq = iter_log['iter']
    x0_value = iter_log['x0']
    x1_value = iter_log['x1']
    f_value = list(iter_log['f(x)'])
    f_min = f_value[-1]
    plt.plot((np.array(f_value) - f_min).tolist(), label='f(x)-f_min')
    plt.yscale("log")
    plt.title(f'optimization curve when start: x=({x0_value[0]}, {x1_value[0]})')
    plt.xlabel("number of iterations")
    plt.ylabel("log(f(x)-f_min)")
    plt.savefig("gap.png")


def run(func, eq_cons, ineq_cons, start, iters=100, epsilon=1e-5):
    x0 = float(start[0])
    x1 = float(start[1])
    iter_log = {'iter': [0], 'x0': [round(x0, 3)],
                'x1': [round(x1, 3)], 'f(x)': [round(plugin(func, start), 3)]}
    iter_log = pd.DataFrame(iter_log)
    [x_opt, iter_log] = SQP(func, iters, start, eq_cons, ineq_cons, epsilon, iter_log)
    print("取到最小值的点为:", x_opt)
    print("目标函数的最小值为:", plugin(func, x_opt))
    iter_log.to_csv("迭代过程记录.csv", index=False)
    draw(iter_log)


def input_fun():
    func = input("请输入目标函数，其中分量用x_0和x_1表示：")
    func = sympify(func)
    ineq_cons = []
    eq_cons = []
    print("如果已经输入了所有不等式或者等式约束，请输入空格。")
    while True:
        ineq = input("请逐个输入不等式约束(<=0)：")
        if ineq == ' ':
            break
        ineq_cons.append(sympify(ineq))
    while True:
        eq = input("请逐个输入等式约束：")
        if eq == ' ':
            break
        eq_cons.append(sympify(eq))
    x0 = input("请输入x0的初始值： ")
    x0 = float(x0)
    x1 = input("请输入x1的初始值： ")
    x1 = float(x1)

    start = Matrix([[x0, x1]]).T
    return func, eq_cons, ineq_cons, start


# 测试样例
'''
func = (1 - x_0) ** 2 + 100 * (x_1 - x_0 ** 2) ** 2
ineq_cons = [-(1 - x_0 - 2 * x_1),
             -(1 - x_0 ** 2 - x_1),
             -(1 - x_0 ** 2 + x_1)]
eq_cons = [2 * x_0 + x_1 - 1]
'''
func = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
ineq_cons = [-(1 - x[0] - 2 * x[1]),
             -(1 - x[0] ** 2 - x[1]),
             -(1 - x[0] ** 2 + x[1])]
eq_cons = [2 * x[0] + x[1] - 1]
x0 = 1.0; x1 = 3.0
start = Matrix([[x0, x1]]).T
func, eq_cons, ineq_cons, start = input_fun()

run(func, eq_cons, ineq_cons, start, iters=100, epsilon=1e-6)
