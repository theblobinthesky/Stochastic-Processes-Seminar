import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom
from scipy.stats import entropy

def main():
    plt.rcParams['text.usetex'] = True

    subplots = [
        (0.0, 1.0),
        (0.1, 0.9),
        (0.3, 0.7),
        (0.5, 0.5),
    ]
    supp_size = 20
    fig, axs = plt.subplots(1, len(subplots), figsize = (len(subplots) * 5, 4), tight_layout=True)

    def set_ax_settings(ax, title):
        ax.set_title(title)
        # ax.set_xticklabels([]) 
        # ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)
    
    def plot_bars(lam, color):
        supp = np.arange(supp_size + 1)
        distr = binom(supp_size, supp) * (lam ** supp) * ((1 - lam) ** (supp_size - supp))

        ax.bar(
            supp,
            distr,
            width=0.8,
            color=color,
            alpha=0.5,
            edgecolor='black'
        )

        return distr

    for ax, subplot in zip(axs, subplots):
        x, y = subplot
        distr1 = plot_bars(x, 'blue')
        distr2 = plot_bars(y, 'green')
        relative_entropy = entropy(distr1, distr2)
        print(f"{relative_entropy=}")
        set_ax_settings(ax, f"$(p,q) = ({x}, {y})$")

    fig.savefig("../plots/relative_entropy_binomial_distributions.pdf")

if __name__ == "__main__":
    main()

