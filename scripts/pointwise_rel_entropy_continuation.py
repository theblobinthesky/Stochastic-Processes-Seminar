import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.rcParams['text.usetex'] = True

    num = 1024
    eps = 0.05
    max_value = 10.0
    x = np.linspace(0, 1, num=num)
    y = np.linspace(0, 1, num=num)
    P, Q = np.meshgrid(x, y)
    Z = np.zeros_like(P)

    case1 = Q <= eps
    Z[case1] = max_value

    case2 = P <= eps
    Z[case2] = 0.0

    caseOther = (P != 0) & (Q != 0)
    Z[caseOther] = P[caseOther] * np.log(P[caseOther] / Q[caseOther])

    case3 = (P <= eps) & (Q <= eps)
    Z[case3] = 0.0


    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(right=0.5) 

    C = np.zeros((num, num, 4))
    C[:] = [0.0, 0.0, 1.0, 0.3]
    C[case1] = [0.0, 1.0, 0.0, 1.0]
    C[case2] = [1.0, 0.0, 0.0, 1.0]
    C[case3] = [1.0, 1.0, 0.0, 1.0]

    ax.plot_surface(
        Q, 
        P, 
        Z, 
        facecolors=C,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel(r"$q$", fontsize=24)
    ax.set_ylabel(r"$p$", fontsize=24)
    ax.set_zlabel(r"$p \log \frac{p}{q}$", fontsize=24)

    handles = [
        ax.plot([], [], [], marker='s', color='red', linestyle='', label=r'$p = 0$'),
        ax.plot([], [], [], marker='s', color='yellow', linestyle='', label=r'$p = q = 0$'),
        ax.plot([], [], [], marker='s', color='green', linestyle='', label=r'$q = 0$'),
        ax.plot([], [], [], marker='s', color='blue', linestyle='', label=r'$p \neq 0, q \neq 0$'),
    ]
    ax.legend(
        handles=[handle[0] for handle in handles],
        loc='center left',
        bbox_to_anchor=(1.2, 0.5),
        fontsize=24,
    )

    fig.savefig("../plots/pointwise_rel_entropy_continuation.pdf")

if __name__ == "__main__":
    main()
