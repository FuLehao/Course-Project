import pandas as pd
import numpy as np
import sympy
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


def phi(ieqcons):
    f = 0
    for x in ieqcons:
        f -= sympy.log(-x)
    return f


def Centering(func, start, A, iters=100, epsilon=1e-5):
    xk = start
    m = 0
    while m < iters:
        step = Newton_step(func, xk, A)
        hess = hessian(func, [x[0], x[1]]).subs(list(zip(x, xk)))
        decrement = (step.T*hess*step)[0, 0] # make it a scalar
        if decrement/2 <= epsilon:
            return xk
        # alpha = backtracking(func, xk, step)
        alpha = linesearch(func, xk, step)
        xk += alpha*step
        m += 1
    print("Maximum Iterations Reached in Inner Iteration!")
    return xk


def backtracking(func, xk, direction, alpha=0.1, beta=0.5, iters=100):
    t = 1
    m = 0
    gradient = plugin(Matrix([diff(func, x_i) for x_i in x]), xk)
    while m < iters:
        if plugin(func, xk+t*direction) > (plugin(func, xk)+(alpha*t*gradient.T*direction)[0, 0]):
            t = beta*t
        else:
            return t
    print("Maximum Iterations Reached in Line Search!")
    return t


def linesearch(func, xk, direction):
    c = 0.2
    alpha = 1
    gradient = plugin(Matrix([diff(func, x_i) for x_i in x]), xk)
    Value_0 = plugin(func, xk)
    while true:
        new = plugin(func, xk + alpha * direction)
        upper = Value_0 + alpha * c * (gradient.T * direction)[0]
        lower = Value_0 + alpha * (1 - c) * (gradient.T * direction)[0]
        if new <= upper and new >= lower:
            return alpha
        elif new <= upper:
            alpha = alpha * 1.5
        else:
            alpha = 0.5 * alpha


def Newton_step(func, xk, A):
    gradient = plugin(Matrix([diff(func, x_i) for x_i in x]), xk)
    Hessen = plugin(hessian(func, [x[0], x[1]]), xk)
    if A == 0:
        return -Hessen**(-1)*gradient
    w = (A*Hessen**(-1)*A.T)**(-1)*(-A*Hessen**(-1)*gradient)
    return Hessen**(-1)*(-A.T*w-gradient)


def checking_condition(xk, eq_cons, ineq_cons, epsilon=1e-5):
    n_eq = len(eq_cons)
    n_ineq = len(ineq_cons)
    # 不满足等式约束返回-1，否则返回-2
    for i in range(n_eq):
        if abs(plugin(eq_cons[i], xk)) > epsilon:
            return -1
    for j in range(n_ineq):
        if plugin(ineq_cons[j], xk) > epsilon:
            return -2
    return True


def barrier_method(func, eq_cons, ineq_cons, start, iter_log, gamma=1.5, t=1.0, iters=100, epsilon=1e-5):
    n_eq = len(eq_cons)
    n_ineq = len(ineq_cons)
    if n_eq == 0:
        A = 0
    else:
        A = Matrix(
            [[diff(eq_cons[i], x_i) for x_i in x] for i in range(n_eq)]
        )
    xk = start
    # -1：初始点不满足等式约束，-2：初始点不满足不等式约束
    if checking_condition(xk, eq_cons, ineq_cons, epsilon) == -1:
        print("初始值不满足等式约束条件!")
        return -1, iter_log
    elif checking_condition(xk, eq_cons, ineq_cons, epsilon) == -2:
        print("初始值不满足不等式约束条件!")
        return -2, iter_log
    for iter in range(iters):
        msg = '第{}次迭代：当前迭代点  x0 = {:.3}, x1 = {:.3f} | 迭代点目标函数值: {:.3f}'
        print(msg.format(iter, float(xk[0]), float(xk[1]), float(plugin(func, xk))))
        if iter >= 1:
            iter_log.loc[len(iter_log.index)] = [iter, round(float(xk[0]), 3), round(float(xk[1]), 3),
                                                 round(float(plugin(func, xk)), 3)]

        g = t*func + phi(ineq_cons)
        xk = Centering(g, xk, A, iters, epsilon)
        if len(ineq_cons)/t < epsilon:
            print("一共迭代了{}次".format(iter))
            return xk, iter_log
        t = gamma * t
        iter += 1
    print("一共迭代了{}次, 未找到最优值，请尝试更改迭代次数iter和容忍度epsilon".format(iter))
    return -3, iter_log


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
    gamma = input("请输入参数gamma的值(gamma>1)：")
    gamma = float(gamma)
    t = input("请输入参数t的初始值(t>0)：")
    t = float(t)

    start = Matrix([[x0, x1]]).T
    return func, eq_cons, ineq_cons, start, gamma, t


def draw(iter_log, gamma, t):
    iter_seq = iter_log['iter']
    x0_value = iter_log['x0']
    x1_value = iter_log['x1']
    f_value = list(iter_log['f(x)'])
    f_min = f_value[-1]
    plt.plot((np.array(f_value) - f_min).tolist(), label='f(x)-f_min')
    plt.yscale("log")
    plt.title(f'optimization curve when start: x=({x0_value[0]}, {x1_value[0]}), gamma={gamma} and t={t}')
    plt.xlabel("number of iterations")
    plt.ylabel("log(f(x)-f_min)")
    plt.savefig("Barrier_output/gap.png")


def run(func, eq_cons, ineq_cons, start, gamma, t, iters=100, epsilon=1e-3):
    x0 = float(start[0])
    x1 = float(start[1])
    iter_log = {'iter': [0], 'x0': [round(x0, 3)],
                'x1': [round(x1, 3)], 'f(x)': [round(plugin(func, start), 3)]}
    iter_log = pd.DataFrame(iter_log)
    [x_opt, iter_log] = barrier_method(func, eq_cons, ineq_cons, start, iter_log, gamma, t, iters, epsilon)
    if x_opt == -1 or x_opt == -2:
        print("求解失败!")
        return False
    if x_opt == -3:
        return False
    print("取到最小值的点为:", round(float(x_opt[0]), 3), round(float(x_opt[1]), 3))
    print("目标函数的最小值为:", round(float(plugin(func, x_opt)), 3))
    iter_log.to_csv("Barrier_output/迭代过程记录.csv", index=False)
    draw(iter_log, gamma, t)


'''
用户手动输入样例：
测试样例1
func = 4 * x_0**2 + x_1**2
ineq_cons = [x_1 - 1]
eq_cons = [x_0 + x_1 - 1]

测试样例2
func = -log(x_0 + x_1)
ineq_cons = [-1 - x_1]
eq_cons = [x_0 + 2*x_1 - 1]
'''

'''
程序端直接输入：
# 测试样例1
func = 4 * x[0]**2 + x[1]**2
ineq_cons = [x[1] - 1]
eq_cons = [x[0] + x[1] - 1]
x0 = 1; x1 = 0
start = Matrix([[x0, x1]]).T
gamma = 1.5
t = 10
'''

# 测试样例2
func = -log(x[0] + x[1])
ineq_cons = [-1 - x[1]]
eq_cons = [x[0] + 2*x[1] - 1]
x0 = 1; x1 = 0
start = Matrix([[x0, x1]]).T
gamma = 1.1
t = 1

func, eq_cons, ineq_cons, start, gamma, t = input_fun()
run(func, eq_cons, ineq_cons, start, gamma, t, iters=100, epsilon=1e-3)
