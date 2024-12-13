import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import math

from scipy.stats import norm, t, uniform, chi2
size = [20, 100]
quantiles = [30.14, 124.34]


def hyp_testing(distName, param=0):
    results = []
    for i in range(len(size)):
        if distName == t:
            x = (distName.rvs(df=param, size=size[i]))
            name = "Student's distribution"
        elif distName == uniform:
            x = (distName.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size[i]))
            name = "Uniform distribution"
        else:
            x = (distName.rvs(size=size[i]))
            name = "Normal distribution"

        print(f"РАСПРЕДЕЛЕНИЕ {name}; РАЗМЕР ВЫБОРКИ n = {size[i]}\n")

        alpha = 0.05
        k = round(1 + (3.3 * math.log10(size[i])))
        print("k = ", k)

        # chi_q = quantiles[i]
        chi_q = round(chi2.ppf(1 - alpha, k - 1), 2)
        a = min(x)
        b = max(x)

        deltas = np.linspace(a, b, k+1)

        print("------------------------deltas: ")
        deltas_r = [0 for _ in range(len(deltas))]
        deltas_r[-1] = float('inf')
        deltas_r[0] = float('-inf')
        for m___ in range(1,len(deltas)-1):
            deltas_r[m___] = round(deltas[m___],2)
        print("($-\infty$, ")
        for k___ in range(1,len(deltas)-1):
            print(deltas_r[k___], "] & [", deltas_r[k___], "," )
        print("$\infty$) & ")
        print("-------------------------------\n")
        # m = 0
        # while m != len(deltas) - 1:
        #     if distName == t:
        #         p_i = distName.cdf(deltas[m + 1], df=param) - distName.cdf(deltas[m], df=param)
        #     elif distName == uniform:
        #         p_i = (distName.cdf(deltas[m + 1], loc=-np.sqrt(3), scale=2 * np.sqrt(3))
        #                - distName.cdf(deltas[m], loc=-np.sqrt(3), scale=2 * np.sqrt(3)))
        #     else:
        #         p_i = distName.cdf(deltas[m + 1]) - distName.cdf(deltas[m])
        #     if size[i] * p_i >= 5:
        #         m += 1
        #         continue
        #     if m + 1 == len(deltas) - 1:
        #         deltas = np.delete(deltas, m)
        #         k -= 1
        #         continue
        #     deltas = np.delete(deltas, (m+1))
        #     k -= 1
        #
        # print("new k = ", k)
        # print("------------------------new deltas: ")
        # print(deltas)
        # print("-------------------------------")


        P = []
        for j in range(len(deltas)-1):
            if distName == t:
                p_i = distName.cdf(deltas[j + 1], df=param) - distName.cdf(deltas[j], df=param)
            elif distName == uniform:
                p_i = (distName.cdf(deltas[j + 1], loc=-np.sqrt(3), scale=2 * np.sqrt(3))
                       - distName.cdf(deltas[j], loc=-np.sqrt(3), scale=2 * np.sqrt(3)))
            else:
                p_i = distName.cdf(deltas[j + 1]) - distName.cdf(deltas[j])
            P.append(p_i)

        P_r = [0 for _ in range(len(P))]
        for m_ in range(len(P)):
            P_r[m_] = round(P[m_], 2)


        sumP = sum(P)
        if sumP == 1.0:
            print("SUM { p_i } = ", sumP, " - ВЕРНО\n")
        else:
            print("SUM { p_i } != 1.0 - woopsie! sth went wrong...\n")

        print("-------------------------------P: ")
        print(*P_r, sep=" & ", end=f" & {round(sumP, 2)}")
        print("\n-------------------------------")

        n_k = [0 for _ in range(k)]
        for value in x:
            for j in range(k):
                if deltas[j] < value <= deltas[j + 1]:
                    n_k[j] += 1

        print("------------------------------n_i: ")
        print(*n_k, sep=" & ")
        print("-------------------------------")

        sumN_K = sum(n_k)
        if sumN_K == size[i]:
            print("sum n_k = ", sumN_K, " - ВЕРНО\n")
        else:
            print("sum n_k != ", size[i], " - woopsie! sth went wrong...\n")

        np_k = [0 for _ in range(k)]
        for i_ in range(k):
            np_k[i_] = round(size[i]*P[i_], 2)

        print("------------------------------np_i: ")
        print(*np_k, sep=" & ")
        print("-------------------------------")
        print("sum np_i = ", sum(np_k), "\n")

        summ1 = 0
        print("------------------------------n_i - np_i: ")
        for i_ in range(k):
            summ1 += n_k[i_] - np_k[i_]
            print(round(n_k[i_] - np_k[i_], 2), end=" & ")
        print("\n-------------------------------")
        print("sum (n_i - np_i) = ", summ1, "\n")

        summ2 = 0
        print("------------------------------(n_i - np_i)**2: ")
        for i_ in range(k):
            summ2 += (n_k[i_] - np_k[i_])**2
            print(round((n_k[i_] - np_k[i_])**2,2), end=" & ")
        print("\n-------------------------------")
        print("sum (n_i - np_i)**2 = ", summ2, "\n")


        chi_Bs = [((n_k[m] - size[i] * P[m]) ** 2) / (size[i] * P[m]) for m in range(k)]

        chi_Bs_r = [0 for _ in range(len(chi_Bs))]
        for m__ in range(len(chi_Bs)):
            chi_Bs_r[m__]=round(chi_Bs[m__],2)

        print("-------------------------------chi_Bs: ")
        print(*chi_Bs_r, sep=" & ")
        print("-------------------------------")

        chi_B = sum(chi_Bs)
        print("sum chi_Bs = ", chi_B, "\n")


        print("chi_B = ", chi_B)

        print("chi_q = ", chi_q)

        if chi_B < chi_q:
            results.append(True)
            print("\nchi_B < chi_q => гипотеза 𝐻_0 на данном этапе проверки принимается\n\n")
        else:
            results.append(False)
            print("\nchi_B >= chi_q => гипотеза 𝐻_0 на данном этапе проверки отвергается: выберем одно из альтернативных распределений"
                  " и повторим процедуру проверки\n\n")

    return results


print("\n----------> testing NORMAL DISTRIBUTION\n")
print("РЕЗУЛЬТАТ: ", hyp_testing(norm))
print("<--------------------------------------------------------------\n")

print("\n----------> testing STUDENT'S DISTRIBUTION\n")
print("РЕЗУЛЬТАТ: ", hyp_testing(t, 3))
print("<--------------------------------------------------------------\n")

print("\n----------> testing UNIFORM DISTRIBUTION\n")
print("РЕЗУЛЬТАТ: ", hyp_testing(uniform))
print("<--------------------------------------------------------------\n")
