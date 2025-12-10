from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor, nn

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_PATH = PLOTS_DIR / "peak_distribution_minimises_entropy.pdf"


@Tensor.train()
def main():
    plt.rcParams["text.usetex"] = True

    N = 10
    logits = Tensor.randn(N, requires_grad=True)

    optim = nn.optim.Adam([logits], lr=0.1)

    history_probs = []
    history_entropy = []

    steps = 50
    for i in range(steps):
        optim.zero_grad()

        probs = logits.softmax()

        entropy = -(probs * probs.log2()).sum()
        loss = entropy

        loss.backward()
        optim.schedule_step()

        if i % 5 == 0 or i == steps - 1:
            history_probs.append(probs.numpy())
            history_entropy.append(entropy.item())

    snapshots = [0, 3, 6, 9]
    fig, axes = plt.subplots(1, 4, figsize=(8, 4), constrained_layout=True)

    x = range(1, N + 1)

    for ax_idx, snapshot_idx in enumerate(snapshots):
        ax = axes[ax_idx]
        p = history_probs[snapshot_idx]
        h = history_entropy[snapshot_idx]

        ax.bar(x, p, color="tab:blue", alpha=0.7, edgecolor="black")
        ax.set_ylim(0, 1)
        ax.set_title(f"Step {(snapshot_idx + 1) * 5}\n$H(X)={h:.2f}$ bits", fontsize=16)
        ax.set_xticks([])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH)
    print(f"Final Entropy: {history_entropy[-1]:.4f} (Max possible: {np.log2(N):.4f})")


if __name__ == "__main__":
    main()
