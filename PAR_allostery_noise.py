import numpy as np
import networkx as nx
from matplotlib import colors
import matplotlib.pyplot as plt
from numba import njit
from scipy import signal
from tqdm import tqdm

contact_pairs = {
     0: [1, 5, 6],                   1: [0, 2, 7],               2: [1, 3, 8],
     3: [2, 4, 9],                   4: [3, 9, 10],              5: [0, 6, 16],
     6: [0, 5, 7, 15, 16],           7: [1, 6, 8, 14, 15],       8: [2, 7, 9, 14],
     9: [3, 4, 8, 10, 13],          10: [4, 9, 11, 12, 13],     11: [10, 12],
    12: [10, 11, 13, 21, 29],       13: [9, 10, 12, 14, 21],    14: [7, 8, 13, 15, 20, 21],
    15: [6, 7, 14, 16, 18, 19],     16: [5, 6, 15, 17, 18],     17: [16, 18],
    18: [15, 16, 17, 19, 24, 25],   19: [15, 18, 20, 23, 24],   20: [14, 19, 21, 22, 23],
    21: [12, 13, 14, 20, 22],       22: [20, 21, 28, 29],       23: [19, 20, 24, 26, 27],
    24: [18, 19, 23, 25, 26],       25: [18, 24],               26: [23, 24, 27],
    27: [23, 26, 28],               28: [22, 27, 29],           29: [12, 22, 28],
}

N = 30
IN, OUT = 0, N - 1
A = nx.to_numpy_array(nx.Graph(contact_pairs), nodelist=range(N))


def input_signal(T, dt, n_periods):
    n = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, n)
    return t, np.pi + np.pi * signal.sawtooth(2 * np.pi * n_periods * t / T, width=0.5)


def trapezoid_weights(t):
    dt = t[1] - t[0]
    w = np.full(len(t), dt)
    w[[0, -1]] *= 0.5
    return w

def make_noise(rng, sigma, n_steps, dt, SHARE_NOISE):
    # Euler-Maruyama increments: sigma * sqrt(dt) * N(0, 1) for every oscillator and time step.
    # The free and clamped runs get the same noise if SHARE_NOISE is True, otherwise they get independent noise.
    if sigma == 0:
        z = np.zeros((N, n_steps))
        return z, z
    dWF = sigma * np.sqrt(dt) * rng.standard_normal((N, n_steps))
    if SHARE_NOISE:
        dWC = dWF
    else:
        dWC = sigma * np.sqrt(dt) * rng.standard_normal((N, n_steps))
    return dWF, dWC

@njit(cache=True)
def run_trial(K, A, theta0, omega, drive, target, dt, eta, clamp, IN, OUT, F, C, dWF, dWC):
    N = K.shape[0]
    n = drive.shape[0]
    for a in range(N):
        F[a, 0] = theta0[a]
        C[a, 0] = theta0[a]
    F[IN, 0] = drive[0]
    C[IN, 0] = drive[0]
    for i in range(1, n):
        for a in range(N):
            sf = 0.0
            sc = 0.0
            for b in range(N):
                if A[a, b] != 0.0:
                    sf += K[a, b] * np.sin(F[b, i - 1] - F[a, i - 1])
                    if clamp:
                        sc += K[a, b] * np.sin(C[b, i - 1] - C[a, i - 1])
            F[a, i] = F[a, i - 1] + dt * (omega[a] + sf) + dWF[a, i]      # + noise kick
            if clamp:
                C[a, i] = C[a, i - 1] + dt * (omega[a] + sc) + dWC[a, i]  # + noise kick
        F[IN, i] = drive[i]
        if clamp:
            C[IN, i] = drive[i]
            C[OUT, i] = F[OUT, i] + eta * np.sin(target[i] - F[OUT, i])


@njit(cache=True)
def contrastive_update(F, C, A, w, alpha, eta):
    N, n = F.shape
    dK = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            if A[a, b] != 0.0:
                s = 0.0
                for i in range(n):
                    s += w[i] * np.sin(C[a, i] - F[a, i]) * np.sin(F[b, i] - F[a, i])
                dK[a, b] = alpha / eta * s
    return dK


def cost(F, target):
    return float(np.mean(1.0 - np.cos(F[OUT] - target)))

def train(T, DT, N_PERIODS, n_epochs, alpha, eta, seed, sigma, SHARE_NOISE, MARGIN_CAP, log_every=1000):
    rng = np.random.default_rng(seed)
    t, drive = input_signal(T, DT, N_PERIODS)
    dt, w = t[1] - t[0], trapezoid_weights(t)
    omega = np.zeros(N)
    K = rng.random((N, N)) * A
    F = np.empty((N, len(t)))
    C = np.empty((N, len(t)))
    hist = np.empty(n_epochs)

    disable_tqdm = False if log_every > 0 else True

    for ep in tqdm(range(n_epochs), disable=disable_tqdm, desc="Training"):
        theta0 = rng.uniform(0, 2 * np.pi, N)
        dWF, dWC = make_noise(rng, sigma, len(t), dt, SHARE_NOISE)
        run_trial(K, A, theta0, omega, drive, drive, dt, eta, True, IN, OUT, F, C, dWF, dWC)
        hist[ep] = cost(F, drive)
        K += contrastive_update(F, C, A, w, alpha, eta)
        if MARGIN_CAP > 0:
            margin = np.abs(K * A).sum(axis=1) * dt
            K *= np.where(margin > MARGIN_CAP, MARGIN_CAP / margin, 1.0)[:, None]
        if log_every > 0:
            if ep % log_every == 0:
                tqdm.write(f"ep {ep:6d}  cost {hist[ep]:.4f}  max|K| {np.abs(K * A).max():.1f}")

    return K, omega, hist

def test(K, omega, ETA, T, DT, N_PERIODS, SIGMA_TEST, SHARE_NOISE, th0_seed, noise_seed):

    t, drive = input_signal(T, DT, N_PERIODS)
    dt = t[1] - t[0]
    F = np.empty((N, len(t)))
    C = np.empty((N, len(t)))
    th0 = np.random.default_rng(th0_seed).uniform(0, 2 * np.pi, N)
    dWF, dWC = make_noise(np.random.default_rng(noise_seed), SIGMA_TEST, len(t), dt, SHARE_NOISE)
    run_trial(K, A, th0, omega, drive, drive, dt, ETA, False, IN, OUT, F, C, dWF, dWC)
    out = np.unwrap(F[OUT])
    out += 2 * np.pi * np.round(np.mean(drive - out) / (2 * np.pi))

    return t, drive, out

def main(T, DT, N_PERIODS, ETA, ALPHA, N_EPOCHS, SEED, SIGMA, SIGMA_TEST, SHARE_NOISE, MARGIN_CAP, log_every=1000, th0_seed=2026, noise_seed=2027):

    K, omega, hist = train(T=T, DT=DT, N_PERIODS=N_PERIODS, n_epochs=N_EPOCHS, alpha=ALPHA, eta=ETA, seed=SEED, sigma=SIGMA, SHARE_NOISE=SHARE_NOISE, MARGIN_CAP=MARGIN_CAP, log_every=log_every)
    t, drive, out = test(K, omega, ETA=ETA, T=T, DT=DT, N_PERIODS=N_PERIODS, SIGMA_TEST=SIGMA_TEST, SHARE_NOISE=SHARE_NOISE, th0_seed=th0_seed, noise_seed=noise_seed)

    return hist, t, drive, out, K

def plot_results(hist, t, drive, out, K):

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist, lw=0.5)
    ax[0].set(xlabel="epoch", ylabel="Cost", yscale="log")
    ax[1].plot(t, drive, label="input / target")
    ax[1].plot(t, out, "--", label="free output (trained)")
    ax[1].set(xlabel="t", ylabel="phase (0 - 2π)")
    ax[1].legend()
    ax[1].grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

    G = nx.Graph(contact_pairs)

    nx.spring_layout(G, weight="weight", k=1.2, iterations=300, seed=42)

    pos = nx.spring_layout(G, weight="weight", k=1.2, iterations=300, seed=42)

    G = nx.from_numpy_array(K, create_using=nx.DiGraph)

    edges = list(G.edges())
    weights = [G[u][v]["weight"] for u, v in edges]

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)

    nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges,
    edge_color=weights,
    edge_cmap=plt.cm.seismic,
    width=3,
    arrows=True,
    arrowsize=20,
    connectionstyle="arc3,rad=0.15",
    node_size=1600,
    ax=ax,
    )

    norm = colors.Normalize(vmin=min(weights), vmax=max(weights))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm)

    plt.colorbar(sm, ax=ax, label="Contact Weight")

    ax.set_title("Contact Network (Weighted DiGraph)")
    ax.axis("off")
    plt.show()

if __name__ == "__main__":

    ETA = 0.01
    ALPHA = 0.005
    T = 20.0
    DT = 0.01
    N_PERIODS = 1
    N_EPOCHS = 10_000
    SEED = 0

    SIGMA = 1.0           # training noise
    SIGMA_TEST = 1.0      # testing noise
    MARGIN_CAP = 1.0      # Euler stability guard
    SHARE_NOISE = False   # whether free and clamped runs share the same noise

    hist, t, drive, out, K = main(T, DT, N_PERIODS, ETA, ALPHA, N_EPOCHS, SEED, SIGMA, SIGMA_TEST, SHARE_NOISE, MARGIN_CAP)
    plot_results(hist, t, drive, out, K)