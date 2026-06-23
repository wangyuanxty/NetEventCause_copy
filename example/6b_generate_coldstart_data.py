"""
Generate synthetic data for cold-start evaluation.
====================================================

Causal structure:
  A → B, A → C, C → D, C → G
  E, F: independent root types

Type index:
  0: A  (root, causes B and C)
  1: B  (derivative of A)
  2: C  (derivative of A, causes D and G)
  3: D  (derivative of C)
  4: E  (independent root)
  5: F  (independent root)      ← cold-start type
  6: G  (derivative of C)       ← cold-start type

Cold-start experiment:
  Train: types 0-4 (A, B, C, D, E).  Sequences without F and G.
  Test:  types 0-6.  Full sequences.

Output:
  data_train.pkl  — sequences without type 5 or 6
  data_test.pkl   — full sequences with all 7 types
  config.json     — generation parameters
  statistics.txt  — per-type statistics
"""

import os
import os.path as osp
import sys
import argparse
import pickle

if "__file__" in globals():
    os.chdir(os.path.dirname(__file__) + "/..")
sys.path.append(".")

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from cause.event.pkg.utils.misc import set_rand_seed, export_json, makedirs, Timer
from cause.event.pkg.utils.pp import get_event_seqs_report
from cause.event.pkg.models.ggem import GammaGraphicalEventModel, EventNode


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate event sequences for cold-start evaluation."
    )
    parser.add_argument("--n_train_seqs", type=int, default=700,
                        help="number of training sequences (default: 700)")
    parser.add_argument("--n_val_seqs", type=int, default=100,
                        help="number of validation sequences (default: 100)")
    parser.add_argument("--n_test_seqs", type=int, default=200,
                        help="number of test sequences (default: 200)")
    parser.add_argument("--max_t", type=int, default=500)
    parser.add_argument("--rand_seed", type=int, default=42)
    # A→B
    parser.add_argument("--alpha_b_a", type=float, default=20)
    parser.add_argument("--beta_b_a", type=float, default=0.2)
    parser.add_argument("--ratio_b_a", type=float, default=5)
    parser.add_argument("--win_b_a", type=float, default=10)
    # A→C
    parser.add_argument("--alpha_c_a", type=float, default=20)
    parser.add_argument("--beta_c_a", type=float, default=0.4)
    parser.add_argument("--ratio_c_a", type=float, default=5)
    parser.add_argument("--win_c_a", type=float, default=20)
    # C→D
    parser.add_argument("--alpha_d_c", type=float, default=40)
    parser.add_argument("--beta_d_c", type=float, default=0.4)
    parser.add_argument("--ratio_d_c", type=float, default=15)
    parser.add_argument("--win_d_c", type=float, default=20)
    # C→G (same as C→D)
    parser.add_argument("--alpha_g_c", type=float, default=40)
    parser.add_argument("--beta_g_c", type=float, default=0.4)
    parser.add_argument("--ratio_g_c", type=float, default=15)
    parser.add_argument("--win_g_c", type=float, default=20)
    # base intensities
    parser.add_argument("--intens_a", type=float, default=0.05)
    parser.add_argument("--intens_b", type=float, default=0.02)
    parser.add_argument("--intens_c", type=float, default=0.02)
    parser.add_argument("--intens_d", type=float, default=0.02)
    parser.add_argument("--intens_e", type=float, default=0.03)
    parser.add_argument("--intens_f", type=float, default=0.03)
    parser.add_argument("--intens_g", type=float, default=0.02)
    return parser


def build_model(args):
    """Build 7-type model: A->B, A->C, C->D, C->G, E(root), F(root)."""
    # type indices
    A, B, C, D, E, F, G = 0, 1, 2, 3, 4, 5, 6

    nodes = [
        EventNode(A, parent=-1, intensity_base=args.intens_a),
        EventNode(B, parent=A, intensity_base=args.intens_b,
                  window=args.win_b_a, alpha=args.alpha_b_a,
                  beta=args.beta_b_a, ratio=args.ratio_b_a),
        EventNode(C, parent=A, intensity_base=args.intens_c,
                  window=args.win_c_a, alpha=args.alpha_c_a,
                  beta=args.beta_c_a, ratio=args.ratio_c_a),
        EventNode(D, parent=C, intensity_base=args.intens_d,
                  window=args.win_d_c, alpha=args.alpha_d_c,
                  beta=args.beta_d_c, ratio=args.ratio_d_c),
        EventNode(E, parent=-1, intensity_base=args.intens_e),
        EventNode(F, parent=-1, intensity_base=args.intens_f),
        EventNode(G, parent=C, intensity_base=args.intens_g,
                  window=args.win_g_c, alpha=args.alpha_g_c,
                  beta=args.beta_g_c, ratio=args.ratio_g_c),
    ]
    return GammaGraphicalEventModel(nodes)


def strip_cold_types(seq: np.ndarray, cold_types=(5, 6)) -> np.ndarray:
    """Remove events with cold-start types from a sequence."""
    mask = ~np.isin(seq[:, 1].astype(int), list(cold_types))
    return seq[mask]


def main():
    args = get_parser().parse_args()
    set_rand_seed(args.rand_seed)

    model = build_model(args)
    n_types = 7
    cold_types = (5, 6)  # F and G

    # ── Generate train/val/test sequences ──
    def generate_seqs(n, strip_cold, desc):
        seqs = []
        for _ in tqdm(range(n), desc=desc):
            seq = model.simulate(0, args.max_t)
            seq = np.array(seq)
            if strip_cold:
                seq = strip_cold_types(seq, cold_types)
            if len(seq) >= 2:
                seqs.append(seq)
        return seqs

    with Timer("Simulating train+val+test sequences"):
        train_seqs = generate_seqs(args.n_train_seqs, True, "Train")
        val_seqs   = generate_seqs(args.n_val_seqs,   True, "Val  ")
        test_seqs  = generate_seqs(args.n_test_seqs,  False, "Test ")

    # ── Save ──
    out_dir = "cache/toy/dataset/coldstart-7"
    makedirs([out_dir])
    export_json(vars(args), osp.join(out_dir, "config.json"))

    with open(osp.join(out_dir, "statistics.txt"), "w") as f:
        for name, seqs in [("Train", train_seqs), ("Val", val_seqs), ("Test", test_seqs)]:
            f.write(f"=== {name} ===\n")
            report = get_event_seqs_report(seqs, n_types)
            print(f"\n{name}: {len(seqs)} seqs"); print(report)
            f.write(report)

    # data.pkl: 800 train+val + 200 test (train_nn_models 内部做 8/9 拆分)
    all_seqs = train_seqs + val_seqs + test_seqs
    train_idx = np.arange(len(train_seqs) + len(val_seqs))        # 0..799
    test_idx  = np.arange(len(train_seqs) + len(val_seqs), len(all_seqs))  # 800..999

    with open(osp.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump({
            "event_seqs": all_seqs,
            "train_test_splits": [(train_idx, test_idx)],
            "n_types": n_types,
        }, f)

    print(f"\nSaved to {out_dir}/")
    print(f"  data.pkl: {len(train_seqs)} train + {len(val_seqs)} val + {len(test_seqs)} test")


if __name__ == "__main__":
    main()
