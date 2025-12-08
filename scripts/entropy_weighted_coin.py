import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.rcParams['text.usetex'] = True

    p = np.linspace(0, 1, num=64)
    H = np.zeros_like(p)
    non_zero = (p > 0) & (p < 1)
    H[non_zero] = -p[non_zero] * np.log2(p[non_zero]) - (1 - p[non_zero]) * np.log2(1 - p[non_zero])

    fig, ax = plt.subplots(figsize = (8, 4), tight_layout=True)
    ax.plot(p, H)
    ax.set_xlabel(r'$p$', fontsize=16)
    ax.set_ylabel(r'$H(X_p)$', fontsize=16)
    ax.grid(True, alpha=0.3)

    fig.savefig("../plots/entropy_weighted_coin.pdf")

if __name__ == "__main__":
    main()
