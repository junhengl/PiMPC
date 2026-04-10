"""
PiMPC PyTorch Benchmark & Verification
=======================================
Tests the PyTorch solver against the AFTI-16 aircraft example,
benchmarks solve time, and reports convergence accuracy.

Run:
    python benchmark.py [--device cpu|cuda] [--batch N] [--dtype float32|float64]
"""

import argparse
import time
import torch
import numpy as np
from scipy.linalg import expm

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pimpc import Model


# -----------------------------------------------------------------------
# AFTI-16 aircraft model (same as Julia example)
# -----------------------------------------------------------------------

def build_AFTI16(Ts: float = 0.05):
    As = np.array([
        [-0.0151, -60.5651,  0.0,    -32.174],
        [-0.0001,  -1.3411,  0.9929,   0.0  ],
        [ 0.00018, 43.2541, -0.86939,  0.0  ],
        [ 0.0,      0.0,     1.0,      0.0  ]
    ])
    Bs = np.array([
        [-2.516,  -13.136 ],
        [-0.1689,  -0.2514],
        [-17.251,  -1.5766],
        [ 0.0,      0.0   ]
    ])
    nx, nu = 4, 2
    M  = expm(np.block([[As, Bs], [np.zeros((nu, nx)), np.zeros((nu, nu))]]) * Ts)
    A  = M[:nx, :nx]
    B  = M[:nx, nx:]
    C  = np.array([[0., 1., 0., 0.],   # angle of attack
                   [0., 0., 0., 1.]])  # pitch angle
    return A, B, C


# -----------------------------------------------------------------------
# Closed-loop simulation
# -----------------------------------------------------------------------

def simulate_closed_loop(model, A_np, B_np, C_np,
                         x0, u0, yref_traj, uref_np, w_np, Nsim,
                         device, dtype):
    nx, nu = B_np.shape
    ny     = C_np.shape[0]
    A_t    = torch.tensor(A_np, dtype=dtype, device=device)
    B_t    = torch.tensor(B_np, dtype=dtype, device=device)
    C_t    = torch.tensor(C_np, dtype=dtype, device=device)

    x_hist    = torch.zeros(nx, Nsim + 1, dtype=dtype)
    y_hist    = torch.zeros(ny, Nsim + 1, dtype=dtype)
    u_hist    = torch.zeros(nu, Nsim,     dtype=dtype)
    iter_hist = []
    time_hist = []

    x_cur  = torch.tensor(x0,  dtype=dtype, device=device)
    u_prev = torch.tensor(u0,  dtype=dtype, device=device)
    w      = torch.tensor(w_np, dtype=dtype, device=device)
    uref   = torch.tensor(uref_np, dtype=dtype, device=device)

    x_hist[:, 0] = x_cur.cpu()
    y_hist[:, 0] = (C_t @ x_cur).cpu()

    for k in range(Nsim):
        yref = torch.tensor(yref_traj[:, k], dtype=dtype, device=device)
        res  = model.solve(x_cur, u_prev, yref, uref, w)
        u_apply = res.u[:, 0]

        x_next = A_t @ x_cur + B_t @ u_apply
        y_next = C_t @ x_next

        x_hist[:, k+1] = x_next.cpu()
        y_hist[:, k+1] = y_next.cpu()
        u_hist[:, k]   = u_apply.cpu()
        iter_hist.append(res.iterations)
        time_hist.append(res.solve_time)

        x_cur  = x_next
        u_prev = u_apply

    return x_hist.numpy(), y_hist.numpy(), u_hist.numpy(), iter_hist, time_hist


# -----------------------------------------------------------------------
# Solve-time benchmark
# -----------------------------------------------------------------------

def benchmark_solve_time(model, A_np, B_np, C_np,
                         x0, u0, yref_np, uref_np, w_np,
                         device, dtype,
                         n_warmup=10, n_runs=200):
    """
    Measures median/mean/std of per-solve wall-clock time.
    Warm-starts are active (the model retains warm_vars between calls).
    """
    x_cur  = torch.tensor(x0,    dtype=dtype, device=device)
    u_prev = torch.tensor(u0,    dtype=dtype, device=device)
    yref   = torch.tensor(yref_np[:, 0], dtype=dtype, device=device)
    uref   = torch.tensor(uref_np, dtype=dtype, device=device)
    w      = torch.tensor(w_np,   dtype=dtype, device=device)

    model.warm_vars = None  # cold start for first call

    # Warm-up
    for _ in range(n_warmup):
        res = model.solve(x_cur, u_prev, yref, uref, w)
        u_prev = res.u[:, 0]
        x_cur  = torch.tensor(A_np, dtype=dtype, device=device) @ x_cur \
                + torch.tensor(B_np, dtype=dtype, device=device) @ u_prev

    # Timed runs
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    x_cur  = torch.tensor(x0,  dtype=dtype, device=device)
    u_prev = torch.tensor(u0,  dtype=dtype, device=device)
    model.warm_vars = None

    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = model.solve(x_cur, u_prev, yref, uref, w)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms

        u_prev = res.u[:, 0]
        x_cur  = torch.tensor(A_np, dtype=dtype, device=device) @ x_cur \
                + torch.tensor(B_np, dtype=dtype, device=device) @ u_prev

    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms":   float(np.mean(times)),
        "std_ms":    float(np.std(times)),
        "min_ms":    float(np.min(times)),
        "max_ms":    float(np.max(times)),
    }


# -----------------------------------------------------------------------
# Batched benchmark
# -----------------------------------------------------------------------

def benchmark_batch(model, A_np, B_np,
                    x0_batch, u0_batch, yref_batch, uref_np, w_np,
                    device, dtype,
                    n_warmup=5, n_runs=50):
    """
    Measures wall-clock time for a single batched solve over B initial conditions.
    """
    x0   = torch.tensor(x0_batch,  dtype=dtype, device=device)
    u0   = torch.tensor(u0_batch,  dtype=dtype, device=device)
    yref = torch.tensor(yref_batch, dtype=dtype, device=device)
    uref = torch.tensor(uref_np,    dtype=dtype, device=device)
    w    = torch.tensor(w_np,       dtype=dtype, device=device)

    for _ in range(n_warmup):
        _ = model.solve_batch(x0, u0, yref, uref, w)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.solve_batch(x0, u0, yref, uref, w)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms":   float(np.mean(times)),
        "std_ms":    float(np.std(times)),
        "batch_size": x0_batch.shape[0],
    }


# -----------------------------------------------------------------------
# Accuracy check: compare trajectories between two models
# -----------------------------------------------------------------------

def compare_trajectories(x1, u1, x2, u2, label1="model1", label2="model2"):
    dx = np.abs(x1 - x2)
    du = np.abs(u1 - u2)
    print(f"  Trajectory comparison ({label1} vs {label2}):")
    print(f"    x  max-abs error: {dx.max():.4e}   mean: {dx.mean():.4e}")
    print(f"    u  max-abs error: {du.max():.4e}   mean: {du.mean():.4e}")
    return dx.max(), du.max()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu",     choices=["cpu", "cuda"])
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--dtype",   default="float64", choices=["float32", "float64"])
    parser.add_argument("--Np",      type=int, default=5)
    parser.add_argument("--Nsim",    type=int, default=200)
    parser.add_argument("--maxiter", type=int, default=10000)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("=" * 60)
    print(f"  PiMPC PyTorch Benchmark")
    print(f"  device={device}  dtype={args.dtype}  Np={args.Np}")
    print("=" * 60)

    # ---- Build system ----
    Ts  = 0.05
    A, B, C = build_AFTI16(Ts)
    nx, nu, ny = 4, 2, 2

    Np      = args.Np
    Nsim    = args.Nsim
    x0      = np.zeros(nx)
    u0      = np.zeros(nu)
    uref_np = np.zeros(nu)
    w_np    = np.zeros(nx)

    half = min(100, Nsim)
    yref_traj = np.vstack([
        np.zeros((1, Nsim)),
        np.hstack([10.0 * np.ones((1, half)), np.zeros((1, Nsim - half))])
    ])

    # ---- Setup models ----
    common_kwargs = dict(
        A=A, B=B, C=C, Np=Np,
        Wy=100.0 * np.eye(ny),
        Wu=0.0   * np.eye(nu),
        Wdu=0.1  * np.eye(nu),
        umin=-25.0 * np.ones(nu),
        umax= 25.0 * np.ones(nu),
        xmin=[-np.inf, -0.5, -np.inf, -100.0],
        xmax=[ np.inf,  0.5,  np.inf,  100.0],
        rho=1.0, tol=1e-6,
        maxiter=args.maxiter,
        precond=True,
        device=device,
        dtype=dtype,
    )

    print("\n[1] Closed-loop simulation (no acceleration)")
    m_base = Model()
    m_base.setup(**common_kwargs, accel=False)
    x_b, y_b, u_b, iters_b, times_b = simulate_closed_loop(
        m_base, A, B, C, x0, u0, yref_traj, uref_np, w_np, Nsim, device, dtype)
    final_target = 0.0 if Nsim > 100 else 10.0
    print(f"  avg iters : {np.mean(iters_b):.1f}")
    print(f"  avg time  : {np.mean(times_b)*1e3:.3f} ms/step")
    print(f"  final y2  : {y_b[1, -1]:.6f}  (target {final_target})")

    print("\n[2] Closed-loop simulation (Nesterov acceleration)")
    m_accel = Model()
    m_accel.setup(**common_kwargs, accel=True)
    x_a, y_a, u_a, iters_a, times_a = simulate_closed_loop(
        m_accel, A, B, C, x0, u0, yref_traj, uref_np, w_np, Nsim, device, dtype)
    print(f"  avg iters : {np.mean(iters_a):.1f}")
    print(f"  avg time  : {np.mean(times_a)*1e3:.3f} ms/step")
    print(f"  final y2  : {y_a[1, -1]:.6f}  (target {final_target})")

    print("\n[3] Trajectory comparison (base vs accel):")
    compare_trajectories(x_b, u_b, x_a, u_a, "base", "accel")

    print("\n[4] Solve-time benchmark (single solve, warm start)")
    m_bench = Model()
    m_bench.setup(**common_kwargs, accel=True)
    stats = benchmark_solve_time(m_bench, A, B, C, x0, u0, yref_traj,
                                  uref_np, w_np, device, dtype,
                                  n_warmup=20, n_runs=500)
    print(f"  median : {stats['median_ms']:.4f} ms")
    print(f"  mean   : {stats['mean_ms']:.4f} ms  ±  {stats['std_ms']:.4f} ms")
    print(f"  min    : {stats['min_ms']:.4f} ms    max: {stats['max_ms']:.4f} ms")

    print(f"\n[5] Batched solve benchmark (B={args.batch})")
    rng = np.random.default_rng(42)
    x0_batch    = rng.normal(0, 0.1, (args.batch, nx))
    u0_batch    = np.zeros((args.batch, nu))
    yref_batch  = np.tile(yref_traj[:, 0], (args.batch, 1))

    m_batch = Model()
    m_batch.setup(**common_kwargs, accel=True)
    bstats = benchmark_batch(m_batch, A, B,
                              x0_batch, u0_batch, yref_batch, uref_np, w_np,
                              device, dtype, n_warmup=10, n_runs=100)
    B_sz = bstats["batch_size"]
    print(f"  batch size   : {B_sz}")
    print(f"  median total : {bstats['median_ms']:.4f} ms")
    print(f"  per-sample   : {bstats['median_ms']/B_sz:.4f} ms")

    print("\n[6] Convergence accuracy test")
    tols = [1e-3, 1e-4, 1e-6]
    x0_t = np.array([0.0, 0.0, 0.0, 0.0])
    u0_t = np.zeros(nu)
    yref_t = np.array([0.0, 10.0])
    w_t = np.zeros(nx)

    ref_model = Model()
    ref_model.setup(**{**common_kwargs, "tol": 1e-10, "maxiter": 50000, "accel": True})
    res_ref = ref_model.solve(
        torch.tensor(x0_t, dtype=dtype, device=device),
        torch.tensor(u0_t, dtype=dtype, device=device),
        torch.tensor(yref_t, dtype=dtype, device=device),
        torch.tensor(uref_np, dtype=dtype, device=device),
        torch.tensor(w_t, dtype=dtype, device=device)
    )
    u_ref = res_ref.u.cpu().numpy()

    print(f"  Reference solve: {res_ref.iterations} iters, obj={res_ref.obj_val:.2e}")
    for tol in tols:
        m_t = Model()
        m_t.setup(**{**common_kwargs, "tol": tol, "accel": True})
        res_t = m_t.solve(
            torch.tensor(x0_t, dtype=dtype, device=device),
            torch.tensor(u0_t, dtype=dtype, device=device),
            torch.tensor(yref_t, dtype=dtype, device=device),
            torch.tensor(uref_np, dtype=dtype, device=device),
            torch.tensor(w_t, dtype=dtype, device=device)
        )
        err = np.abs(res_t.u.cpu().numpy() - u_ref).max()
        print(f"  tol={tol:.0e} : {res_t.iterations:5d} iters, "
              f"u1 err={err:.4e}, converged={res_t.converged}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
