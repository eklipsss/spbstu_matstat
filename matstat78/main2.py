import math
import statistics as st
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
def hyp_test(m, n, alpha, F_quant = 0):
    state = np.random.get_state()
    # print("state: ", state)

    distrib = sps.norm.rvs(size=100)

    x_sample = np.random.choice(distrib, size=m, replace=False)
    y_sample = np.random.choice(distrib, size=n, replace=False)

    np.random.set_state(state)

    x_sqr = [(x_sample[i] - np.mean(x_sample)) ** 2 for i in range(m)]
    y_sqr = [(y_sample[i] - np.mean(y_sample)) ** 2 for i in range(n)]

    plt.hist(x_sample, bins=5, density=True, color='lightblue', alpha=0.5, label=f'размер выборки {m}')
    plt.hist(y_sample, bins=5, density=True, color='yellow', alpha=0.5, label=f'размер выборки {n}')
    plt.hist(distrib, bins=5, density=True, color='crimson', alpha=0.5, label='нормальное распределение n = 100')
    plt.title("Гистограммы выборок из генеральной совокупности")
    plt.legend(loc='best')
    plt.show()

    # несмещенные оценки дисперcий

    s_x = sum(x_sqr) / (m - 1)
    print("S_x = ", s_x)
    s_y = sum(y_sqr) / (n - 1)
    print("S_y = ", s_y)

    if s_x > s_y:
        F_b = s_x / s_y
    else:
        F_b = s_y / s_x

    F_quant = sps.f(m - 1, n - 1).ppf(1 - alpha / 2)
    print("F_quant = ", F_quant)
    print("F_B = ", F_b)

    print('%.2f' % s_x, "&", '%.2f' % s_y, "&", '%.2f' % F_b, "&", '%.2f' % F_quant, "\\\\ \\hline\n")

    if F_quant > F_b:
        print("\nF_quant > F_b => гипотеза 𝐻_0 на данном этапе проверки принимается\n\n")
        return True
    else:
        print("\nF_quant <= F_b => гипотеза 𝐻_0 на данном этапе проверки отвергается: выберем одно из альтернативных"
              " распределений и повторим процедуру проверки\n\n")
        return False


print("\nNormal distribution; size = 100; m = 20, n = 40; alpha = 0.05\n")
hyp_test(20, 40, 0.05)

print("\nNormal distribution; size = 100; m = 20, n = 100; alpha = 0.05\n")
hyp_test(20, 100, 0.05)