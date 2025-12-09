from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit, nn
from tinygrad.helpers import colored, getenv, trange
from tinygrad.nn.datasets import mnist

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
LOSS_PLOT_PATH = PLOTS_DIR / "mnist_loss.pdf"
PREDICTIONS_PLOT_PATH = PLOTS_DIR / "mnist_predictions.pdf"
SAMPLE_COUNT = 3
SEED = getenv("SEED", 1)


class Model:
    def __init__(self) -> None:
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.relu,
            nn.Conv2d(32, 32, 5),
            Tensor.relu,
            nn.BatchNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            nn.Conv2d(64, 64, 3),
            Tensor.relu,
            nn.BatchNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def _save_loss_plot(loss_history: Sequence[float]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="tab:blue")
    ax.set_xlabel("Optimization Step", fontsize=16)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=16)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH)
    plt.close(fig)


def _save_prediction_plot(
    images: Sequence[np.ndarray],
    probs: Sequence[np.ndarray],
    labels: Sequence[int],
) -> None:
    fig, axes = plt.subplots(SAMPLE_COUNT, 2, figsize=(8, 8))
    for row in range(SAMPLE_COUNT):
        img_ax = axes[row, 0]
        prob_ax = axes[row, 1]
        img_ax.imshow(images[row], cmap="gray")

        digits = list(range(10))
        prob_ax.bar(digits, probs[row], color="tab:orange")
        prob_ax.set_xticks(digits)
        prob_ax.set_ylim(0, 1)
        prob_ax.set_ylabel("Probability", fontsize=16)
        prob_ax.set_xlabel("Digit", fontsize=16)
        predicted = int(np.argmax(probs[row]))
        prob_ax.set_title(f"Predicted={predicted}", fontsize=16)
    fig.tight_layout()
    fig.savefig(PREDICTIONS_PLOT_PATH)
    plt.close(fig)


def _collect_samples(
    model: Model, images: Tensor, labels: Tensor, count: int = SAMPLE_COUNT
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    rng = np.random.default_rng(SEED)
    indices_np = rng.choice(images.shape[0], size=count, replace=False)
    indices_tensor = Tensor(indices_np)
    batch_images = images[indices_tensor]
    batch_labels = labels[indices_tensor]
    batch_probs = model(batch_images).softmax(axis=1)

    image_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    label_list: list[int] = []
    for img, probs, lbl in zip(batch_images, batch_probs, batch_labels):
        image_list.append(img.numpy().squeeze())
        probs_list.append(probs.numpy())
        label_list.append(int(lbl.numpy()))
    return image_list, probs_list, label_list


if __name__ == "__main__":
    np.random.seed(SEED)
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    model = Model()
    opt = nn.optim.SGD(nn.state.get_parameters(model), lr=5e-3)
    loss_history: list[float] = []

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        loss = (
            model(X_train[samples])
            .sparse_categorical_crossentropy(Y_train[samples])
            .backward()
        )
        return loss.realize(*opt.schedule_step())

    @TinyJit
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    test_acc = float("nan")
    steps = getenv("STEPS", 60)
    for i in (t := trange(steps)):
        GlobalCounters.reset()
        loss = train_step()
        loss_history.append(loss.item())
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    sample_images, sample_probs, sample_labels = _collect_samples(model, X_test, Y_test)
    _save_loss_plot(loss_history)
    _save_prediction_plot(sample_images, sample_probs, sample_labels)
