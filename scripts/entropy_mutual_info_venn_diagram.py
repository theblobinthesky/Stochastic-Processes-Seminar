import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():
    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(1, 3, figsize = (9, 4), tight_layout=True)

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
            va='center'
        )

    def add_ellipse_and_text(ax, xy, width, name):
        add_ellipse(ax, xy, width)
        add_text(ax, xy, name)
    
    def set_ax_settings(ax, title):
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticklabels([]) 
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    ax = axs[0]
    add_ellipse_and_text(ax, (0.3, 0.5), 0.3, r"$H(X)$")
    add_ellipse_and_text(ax, (0.7, 0.5), 0.3, r"$H(Y)$")
    set_ax_settings(ax, 'No mutual information')

    ax = axs[1]
    add_ellipse(ax, (0.4, 0.5), 0.6) 
    add_text(ax, (0.4, 0.8), r"$H(X)$")
    add_text(ax, (0.2, 0.5), r"$H(X \mid Y)$")

    add_ellipse(ax, (0.6, 0.5), 0.6) 
    add_text(ax, (0.6, 0.8), r"$H(Y)$")
    add_text(ax, (0.8, 0.5), r"$H(Y \mid X)$")

    add_text(ax, (0.5, 0.5), r"$I(X; Y)$")

    set_ax_settings(ax, 'Some mutual information')

    ax = axs[2]
    add_ellipse(ax, (0.5, 0.5), 0.6) 
    add_ellipse_and_text(ax, (0.5, 0.5), 0.6, r"$H(X) = H(Y) = I(X; Y)$")
    set_ax_settings(ax, 'Identical information')

    fig.savefig("../plots/entropy_mutual_info_venn_diagram.pdf")

if __name__ == "__main__":
    main()
