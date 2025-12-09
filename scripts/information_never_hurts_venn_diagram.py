import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(1, 1, figsize = (8, 4), tight_layout=True)

    def add_ellipse(ax, xy, width):
        ax.add_patch(Ellipse(
            xy=xy,
            width=width, 
            height=0.5, 
            angle=0, 
            facecolor='cyan', 
            alpha=0.2, 
            edgecolor='black', 
            linewidth=1.5
        ))

    def add_text(ax, xy, name):
        ax.text(
            *xy, 
            name,
            ha='center', 
            va='center',
            fontsize=24
        )

    def set_ax_settings(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticklabels([]) 
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    add_ellipse(ax, (0.4, 0.5), 0.6) 
    add_ellipse(ax, (0.6, 0.5), 0.6) 

    add_text(ax, (0.4, 0.8), r"$H(X)$")
    add_text(ax, (0.2, 0.5), r"$H(X \mid Y)$")

    add_text(ax, (0.6, 0.8), r"$H(Y)$")
    add_text(ax, (0.8, 0.5), r"$H(Y \mid X)$")

    set_ax_settings(ax)

    fig.savefig("../plots/information_never_hurts_venn_diagram.pdf")

if __name__ == "__main__":
    main()
