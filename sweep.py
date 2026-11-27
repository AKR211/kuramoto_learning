import numpy as np
import networkx as nx
from matplotlib import colors
import matplotlib.pyplot as plt
from numba import njit
from scipy import signal
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import argparse

import PAR_allostery_noise as PAR

def main(sigma_train_list, sigma_test_list, seeds, settings):

    jobs = [
        (sigma_train, sigma_test, {
            **settings,
            "SIGMA": sigma_train,
            "SIGMA_TEST": sigma_test,
            "SEED": seed,
        })
        for sigma_train in sigma_train_list
        for sigma_test in sigma_test_list
        for seed in seeds
    ]

    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(sweep, jobs), total=len(jobs), desc="Sweeping over noise levels"))

    for sigma_train, sigma_test, error in results:
        print(f"sigma_train={sigma_train:g}, sigma_test={sigma_test:g}, "
              f"final average error={error:.6g}")

    return results


def sweep(args):
    sigma_train, sigma_test, settings = args
    hist, t, drive, out, K = PAR.main(
        T=settings["T"],
        DT=settings["DT"],
        N_PERIODS=settings["N_PERIODS"],
        ETA=settings["ETA"],
        ALPHA=settings["ALPHA"],
        N_EPOCHS=settings["N_EPOCHS"],
        SEED=settings["SEED"],
        SIGMA=sigma_train,
        SIGMA_TEST=sigma_test,
        SHARE_NOISE=settings["SHARE_NOISE"],
        MARGIN_CAP=settings["MARGIN_CAP"],
        log_every=settings["log_every"]
    )
    error = np.mean(1.0 - np.cos(out - drive))
    return sigma_train, sigma_test, error

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=40.0, help="total time")
    parser.add_argument("--DT", type=float, default=0.02, help="time step")
    parser.add_argument("--N_PERIODS", type=int, default=1, help="number of periods in the input signal")
    parser.add_argument("--ETA", type=float, default=0.01, help="learning rate for omega")
    parser.add_argument("--ALPHA", type=float, default=0.005, help="learning rate for K")
    parser.add_argument("--N_EPOCHS", type=int, default=10000, help="number of training epochs")
    parser.add_argument("--MARGIN_CAP", type=float, default=1.0, help="Euler stability guard")
    parser.add_argument("--SHARE_NOISE", action="store_true", help="whether free and clamped runs share the same noise")
    parser.add_argument("--log_every", type=int, default=0, help="how often to log training progress")
    parser.add_argument("--sigma_train", type=float, nargs='+', default=[0.0, 2.0, 4.0, 6.0, 8.0], help="list of training noise levels")
    parser.add_argument("--sigma_test", type=float, nargs='+', default=[0.0, 2.0, 4.0, 6.0, 8.0], help="list of testing noise levels")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 5, 10, 15, 20], help="list of seeds for reproducibility")

    args = parser.parse_args()

    # Parameters
    T = args.T               # total time
    DT = args.DT             # time step
    N_PERIODS = args.N_PERIODS         # number of periods in the input signal
    ETA = args.ETA            # learning rate for omega
    ALPHA = args.ALPHA         # learning rate for K
    N_EPOCHS = args.N_EPOCHS      # number of training epochs
    MARGIN_CAP = args.MARGIN_CAP      # Euler stability guard
    SHARE_NOISE = args.SHARE_NOISE   # whether free and clamped runs share the same noise
    log_every = args.log_every         # how often to log training progress

    settings = {
        "T": T,
        "DT": DT,
        "N_PERIODS": N_PERIODS,
        "ETA": ETA,
        "ALPHA": ALPHA,
        "N_EPOCHS": N_EPOCHS,
        "MARGIN_CAP": MARGIN_CAP,
        "SHARE_NOISE": SHARE_NOISE,
        "log_every": log_every,
    }

    sigma_train_list = args.sigma_train
    sigma_test_list = args.sigma_test
    seeds = args.seeds

    results = main(
        sigma_train_list=sigma_train_list,
        sigma_test_list=sigma_test_list,
        seeds=seeds,
        settings=settings
    )

    grid = np.zeros((len(sigma_train_list), len(sigma_test_list)))
    sq_grid = np.zeros((len(sigma_train_list), len(sigma_test_list)))
    err_grid = np.zeros((len(sigma_train_list), len(sigma_test_list)))

    for sigma_train, sigma_test, error in results:
        i = sigma_train_list.index(sigma_train)
        j = sigma_test_list.index(sigma_test)
        grid[i, j] += error
        sq_grid[i, j] += error ** 2

    grid /= len(seeds)
    sq_grid /= len(seeds)

    err_grid = np.sqrt(sq_grid - grid ** 2) / np.sqrt(len(seeds))

    df = pd.DataFrame(grid, index=sigma_train_list, columns=sigma_test_list)
    df.to_csv(f"sweep_results_T{settings['T']}_DT{settings['DT']}_N{settings['N_PERIODS']}_ETA{settings['ETA']}_ALPHA{settings['ALPHA']}_EPOCHS{settings['N_EPOCHS']}_MARGIN{settings['MARGIN_CAP']}_SHARE{settings['SHARE_NOISE']}.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, origin='lower', cmap='viridis',
               aspect='auto', norm=colors.Normalize(vmin=0, vmax=np.max(grid)))

    plt.xticks(ticks=np.arange(len(sigma_test_list)), labels=sigma_test_list)
    plt.yticks(ticks=np.arange(len(sigma_train_list)), labels=sigma_train_list)

    for i in range(len(sigma_train_list)):
        for j in range(len(sigma_test_list)):
            val = grid[i, j]
            err = err_grid[i, j]
            text_color = "white" if val < 0.5 else "black"
            plt.text(j, i, f"{val:.2f} ± {err:.2f}", ha="center", va="center", color=text_color)
    
    plt.colorbar(label='Average Error')
    plt.xlabel('Test Noise Level')
    plt.ylabel('Train Noise Level')
    plt.title('Sweep Results')
    plt.savefig(f"./sweep_results_T{settings['T']}_DT{settings['DT']}_N{settings['N_PERIODS']}_ETA{settings['ETA']}_ALPHA{settings['ALPHA']}_EPOCHS{settings['N_EPOCHS']}_MARGIN{settings['MARGIN_CAP']}_SHARE{settings['SHARE_NOISE']}.png", dpi=300)
    plt.close()