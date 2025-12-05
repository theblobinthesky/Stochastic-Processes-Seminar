import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.rcParams['text.usetex'] = True

    p = np.linspace(0, 1, num=64)
    H = np.zeros_like(p)
    non_zero = p > 0
    H[non_zero] = -p[non_zero] * np.log(p[non_zero])

    fig, ax = plt.subplots(figsize = (6, 4), tight_layout=True)
    ax.plot(p, H)
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$-p\log(p)$')
    ax.grid(True, alpha=0.3)

    fig.savefig("../plots/entropy_function.pdf")

if __name__ == "__main__":
    main()