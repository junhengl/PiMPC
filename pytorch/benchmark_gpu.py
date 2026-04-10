"""
PiMPC Centroidal MPC GPU Batch Benchmark
=========================================
Benchmarks solve_batch() across B parallel environments for a centroidal
dynamics MPC problem matching bipedal locomotion.

System (centroidal model):
  state  x = [c, l, k]  in R^9   (CoM position, linear momentum, angular momentum)
  input  u = [fLF, tauLF, fRF, tauRF] in R^12  (two 6D foot wrenches)
  ny = 9   (full-state observation, C = I)
  Np = 10  (prediction horizon)

Dynamics:
  x_{k+1} = A x_k + B u_k + e
  where A, B are the discrete centroidal dynamics (dt=0.07, mass=37 kg),
  B is linearised at a nominal standing pose (both feet in contact),
  and e encodes gravitational drift.

Sweeps batch sizes B in {1, 4, 16, 64, 256, 1024, 4096} and reports:
  - Total solve time
  - Per-sample time (ms)
  - Throughput (samples / second)
  - GPU vs CPU speedup

Run:
    python pytorch/benchmark_gpu.py
    python pytorch/benchmark_gpu.py --Np 20 --maxiter 500 --tol 1e-3
    python pytorch/benchmark_gpu.py --batch-sizes 1,64,512,4096
"""

import argparse
import time
import sys
import os

import numpy as np
import torch

# Allow running from repo root or pytorch/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch.pimpc import Model


# ---------------------------------------------------------------------------
# Centroidal dynamics builder
# ---------------------------------------------------------------------------

def build_centroidal_system(dt=0.07, mass=37.0, gravity=(0.0, 0.0, -9.81),
                            com_height=0.8, foot_spread=0.1):
    """
    Build the discrete-time centroidal dynamics for bipedal locomotion.

    State  x = [c(3), l(3), k(3)]  in R^9
      c : CoM position
      l : linear momentum
      k : angular momentum

    Input  u = [fLF(3), tauLF(3), fRF(3), tauRF(3)]  in R^12
      Two 6D foot wrenches (force + torque per foot)

    Dynamics:  x_{k+1} = A x_k + B u_k + e

    B is linearised at the nominal standing pose:
      CoM at [0, 0, com_height], both feet on ground,
      left foot at [0, +foot_spread, 0], right foot at [0, -foot_spread, 0].

    Returns A (9x9), B (9x12), e (9,) as numpy arrays.
    """
    g = np.array(gravity)
    m = mass
    nx, nu = 9, 12
    I3 = np.eye(3)

    # -- A matrix (9x9) --
    A = np.zeros((nx, nx))
    A[0:3, 0:3] = I3
    A[0:3, 3:6] = (dt / m) * I3
    A[3:6, 3:6] = I3
    A[6:9, 6:9] = I3

    # -- Affine drift e (gravity) --
    e = np.zeros(nx)
    e[0:3] = 0.5 * dt**2 * g       # position drift
    e[3:6] = m * dt * g             # momentum drift

    # -- B matrix (9x12) at nominal standing pose --
    c_nom = np.array([0.0, 0.0, com_height])
    r_LF  = np.array([0.0,  foot_spread, 0.0])
    r_RF  = np.array([0.0, -foot_spread, 0.0])

    def skew(v):
        return np.array([[    0, -v[2],  v[1]],
                         [ v[2],     0, -v[0]],
                         [-v[1],  v[0],     0]])

    # Force selection: Ef (3x12) maps wrenches -> total force
    Ef = np.zeros((3, nu))
    Ef[:, 0:3] = I3   # left foot force
    Ef[:, 6:9] = I3   # right foot force

    # Torque selection: Et (3x12) maps wrenches -> total torque about CoM
    Et = np.zeros((3, nu))
    Et[:, 0:3]  = skew(r_LF - c_nom)   # cross product from left foot force
    Et[:, 3:6]  = I3                    # left foot torque direct
    Et[:, 6:9]  = skew(r_RF - c_nom)   # cross product from right foot force
    Et[:, 9:12] = I3                    # right foot torque direct

    B = np.zeros((nx, nu))
    B[0:3, :] = (dt**2 / (2 * m)) * Ef   # CoM position
    B[3:6, :] = dt * Ef                   # linear momentum
    B[6:9, :] = dt * Et                   # angular momentum

    return A, B, e


def build_centroidal_weights(mass=37.0, mu=0.6, fz_max=800.0):
    """
    Build cost weight matrices and box constraints matching the centroidal MPC.

    Returns dict with Wy, Wu, Wdu, Wf, umin, umax, dumin, dumax.
    """
    nx, nu = 9, 12

    # Output (state) weights: Q = diag(Q_c, Q_l, Q_k)
    Q_c  = np.array([100.0, 100.0, 200.0])   # CoM position
    Q_l  = np.array([ 10.0,  10.0,  20.0])   # linear momentum
    Q_k  = np.array([ 50.0,  50.0,  50.0])   # angular momentum
    Wy = np.diag(np.concatenate([Q_c, Q_l, Q_k]))

    # Terminal weight: Qf = 10 * Q
    Wf = 10.0 * Wy

    # Input weight: R  (force & torque regularisation)
    R_f_foot   = 1e-4
    R_tau_foot = 1e-4
    R_diag = np.zeros(nu)
    R_diag[0:3]  = R_f_foot     # left foot force
    R_diag[3:6]  = R_tau_foot   # left foot torque
    R_diag[6:9]  = R_f_foot     # right foot force
    R_diag[9:12] = R_tau_foot   # right foot torque
    Wu = np.diag(R_diag)

    # Input increment weight: R_delta * I
    R_delta = 1e-3
    Wdu = R_delta * np.eye(nu)

    # -- Box constraints on inputs --
    # Approximate friction-cone + normal-force limits as box constraints:
    #   |fx|, |fy| <= mu * fz_max   per foot
    #   0 <= fz <= fz_max           per foot
    #   torques bounded by CoP / yaw limits (conservative)
    f_lat_max = mu * fz_max          # 480 N
    tau_max   = 50.0                  # conservative torque bound [N*m]

    umin = np.array([
        -f_lat_max, -f_lat_max, 0.0,        # left foot force
        -tau_max,   -tau_max,  -tau_max,     # left foot torque
        -f_lat_max, -f_lat_max, 0.0,        # right foot force
        -tau_max,   -tau_max,  -tau_max,     # right foot torque
    ])
    umax = np.array([
         f_lat_max,  f_lat_max, fz_max,     # left foot force
         tau_max,    tau_max,   tau_max,     # left foot torque
         f_lat_max,  f_lat_max, fz_max,     # right foot force
         tau_max,    tau_max,   tau_max,     # right foot torque
    ])

    # Input rate limits
    du_force_max  = 200.0    # N per step
    du_torque_max = 20.0     # N*m per step
    dumin = np.array([
        -du_force_max, -du_force_max, -du_force_max,
        -du_torque_max, -du_torque_max, -du_torque_max,
        -du_force_max, -du_force_max, -du_force_max,
        -du_torque_max, -du_torque_max, -du_torque_max,
    ])
    dumax = -dumin

    return dict(Wy=Wy, Wu=Wu, Wdu=Wdu, Wf=Wf,
                umin=umin, umax=umax, dumin=dumin, dumax=dumax)


# ---------------------------------------------------------------------------
# Single benchmark run for one (device, B) combination
# ---------------------------------------------------------------------------

def run_benchmark(model_cfg, B, device, dtype, n_warmup=5, n_runs=20,
                  seed=0):
    """
    Build a Model, run n_warmup + n_runs batch solves.
    Returns dict of timing statistics (milliseconds, per-sample).
    """
    nx  = model_cfg["nx"]
    nu  = model_cfg["nu"]
    ny  = model_cfg["ny"]
    Np  = model_cfg["Np"]

    m = Model()
    m.setup(
        A=model_cfg["A"], B=model_cfg["B"], Np=Np,
        e=model_cfg["e"],
        Wy=model_cfg["Wy"], Wu=model_cfg["Wu"],
        Wdu=model_cfg["Wdu"], Wf=model_cfg["Wf"],
        umin=model_cfg["umin"], umax=model_cfg["umax"],
        dumin=model_cfg["dumin"], dumax=model_cfg["dumax"],
        rho=model_cfg["rho"], tol=model_cfg["tol"],
        maxiter=model_cfg["maxiter"],
        accel=True, precond=False,
        device=device, dtype=dtype,
    )

    mass = 37.0
    g_z = 9.81
    half_weight = mass * g_z / 2.0
    rng = np.random.default_rng(seed)

    # Realistic initial conditions for centroidal state [c, l, k]
    x0 = np.zeros((B, nx))
    x0[:, 0] = rng.uniform(-0.05, 0.05, B)          # CoM x perturbation
    x0[:, 1] = rng.uniform(-0.05, 0.05, B)          # CoM y perturbation
    x0[:, 2] = 0.8 + rng.uniform(-0.02, 0.02, B)   # CoM z ~ standing height
    x0[:, 3:6] = rng.uniform(-5.0, 5.0, (B, 3))    # linear momentum
    x0[:, 6:9] = rng.uniform(-1.0, 1.0, (B, 3))    # angular momentum
    x0 = torch.tensor(x0, dtype=dtype, device=device)

    # Previous input: nominal standing (each foot carries half body weight)
    u0_np = np.zeros((B, nu))
    u0_np[:, 2] = half_weight   # left foot fz
    u0_np[:, 8] = half_weight   # right foot fz
    u0 = torch.tensor(u0_np, dtype=dtype, device=device)

    # Reference: stand at origin with zero momentum
    yref = torch.zeros(B, ny, dtype=dtype, device=device)
    yref[:, 2] = 0.8   # target CoM height

    # Input reference: nominal standing wrenches
    uref = torch.tensor(u0_np, dtype=dtype, device=device)

    # Disturbance: zero
    w = torch.zeros(B, nx, dtype=dtype, device=device)

    # warm-up
    for _ in range(n_warmup):
        _ = m.solve_batch(x0, u0, yref, uref, w)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(n_runs):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = m.solve_batch(x0, u0, yref, uref, w)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms total

    times = np.array(times)
    per_sample = times / B  # ms per env

    return {
        "total_median_ms":  np.median(times),
        "total_mean_ms":    np.mean(times),
        "total_std_ms":     np.std(times),
        "per_ms_median":    np.median(per_sample),
        "per_ms_mean":      np.mean(per_sample),
        "throughput":       B / (np.median(times) * 1e-3),  # samples/s
        "iterations":       res.iterations,
        "converged":        res.converged,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PiMPC centroidal MPC GPU batch benchmark")
    parser.add_argument("--Np",          type=int, default=10,
                        help="Prediction horizon")
    parser.add_argument("--dt",          type=float, default=0.07,
                        help="Discrete time step (s)")
    parser.add_argument("--mass",        type=float, default=37.0,
                        help="Robot mass (kg)")
    parser.add_argument("--rho",         type=float, default=1.0,
                        help="ADMM penalty")
    parser.add_argument("--tol",         type=float, default=1e-3)
    parser.add_argument("--maxiter",     type=int, default=500)
    parser.add_argument("--n-warmup",    type=int, default=5)
    parser.add_argument("--n-runs",      type=int, default=20)
    parser.add_argument("--batch-sizes", type=str,
                        default="1,4,16,64,256,1024,4096",
                        help="Comma-separated list of batch sizes")
    parser.add_argument("--dtype",       choices=["float32", "float64"],
                        default="float32",
                        help="float32 recommended for GPU throughput")
    parser.add_argument("--no-cpu",      action="store_true",
                        help="Skip CPU benchmarks (much slower at large B)")
    parser.add_argument("--no-gpu",      action="store_true",
                        help="Skip GPU benchmarks")
    args = parser.parse_args()

    nx, nu, ny = 9, 12, 9
    Np = args.Np
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    cuda_available = torch.cuda.is_available() and not args.no_gpu
    run_cpu = not args.no_cpu

    # -----------------------------------------------------------------------
    # Build centroidal dynamics
    # -----------------------------------------------------------------------
    A, B_mat, e = build_centroidal_system(
        dt=args.dt, mass=args.mass, com_height=0.8, foot_spread=0.1)
    weights = build_centroidal_weights(mass=args.mass)

    model_cfg = dict(
        nx=nx, nu=nu, ny=ny, A=A, B=B_mat, e=e,
        Np=Np, rho=args.rho, tol=args.tol, maxiter=args.maxiter,
        **weights,
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  PiMPC Centroidal MPC GPU Batch Benchmark")
    print("=" * 70)
    print(f"  Model    : bipedal centroidal dynamics")
    print(f"  State    : x = [c, l, k]  nx={nx}")
    print(f"  Input    : u = [fLF, tauLF, fRF, tauRF]  nu={nu}")
    print(f"  Horizon  : Np={Np},  dt={args.dt} s")
    print(f"  Robot    : mass={args.mass} kg")
    print(f"  nx_bar   : {nx+nu}  (augmented state dimension)")
    print(f"  Solver   : ADMM + Nesterov accel, rho={args.rho}, "
          f"tol={args.tol:.0e}, maxiter={args.maxiter}")
    print(f"  dtype    : {args.dtype}")
    if cuda_available:
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU mem  : {mem_gb:.1f} GB")
    else:
        print("  GPU      : not available")
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"  A rho(A) : {sr:.4f}  (spectral radius)")
    print()

    # -----------------------------------------------------------------------
    # Memory estimate
    # -----------------------------------------------------------------------
    print("  Memory estimate per batch (float32):")
    nx_bar = nx + nu
    for b in [64, 256, 1024, 4096]:
        n_vars = (nx_bar * (Np + 1) +   # X
                  nu * Np +              # DU
                  5 * nx_bar * Np)       # Z,V,Theta,Beta,Lambda
        mb = b * n_vars * 4 / 1e6
        print(f"    B={b:>5}: {mb:>8.1f} MB")
    print()

    # -----------------------------------------------------------------------
    # Warm-up single solve to ensure JIT / kernel launch overhead is excluded
    # -----------------------------------------------------------------------
    if cuda_available:
        print("  CUDA kernel warm-up ...", end="", flush=True)
        _ = run_benchmark(model_cfg, B=1, device="cuda",
                          dtype=dtype, n_warmup=3, n_runs=2)
        print(" done")

    # -----------------------------------------------------------------------
    # Benchmark sweep
    # -----------------------------------------------------------------------
    cpu_results = {}
    gpu_results = {}

    # --- CPU ---
    if run_cpu:
        print()
        print("  [CPU sweep]")
        hdr = (f"  {'B':>6}  {'iters':>6}  {'conv':>5}  {'total ms':>10}  "
               f"{'per-env ms':>11}  {'samples/s':>12}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        cpu_max_b = 64
        cpu_batch_sizes = [b for b in batch_sizes if b <= cpu_max_b]

        for b in cpu_batch_sizes:
            r = run_benchmark(model_cfg, b, device="cpu", dtype=dtype,
                              n_warmup=args.n_warmup, n_runs=args.n_runs)
            cpu_results[b] = r
            print(f"  {b:>6}  {r['iterations']:>6}  {str(r['converged']):>5}  "
                  f"{r['total_median_ms']:>10.2f}  {r['per_ms_median']:>11.4f}  "
                  f"{r['throughput']:>12.0f}", flush=True)

        if any(b > cpu_max_b for b in batch_sizes):
            print(f"  (B > {cpu_max_b} skipped on CPU)")

    # --- GPU ---
    if cuda_available:
        print()
        print("  [GPU sweep]")
        hdr = (f"  {'B':>6}  {'iters':>6}  {'conv':>5}  {'total ms':>10}  "
               f"{'per-env ms':>11}  {'samples/s':>12}  {'speedup':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for b in batch_sizes:
            try:
                r = run_benchmark(model_cfg, b, device="cuda", dtype=dtype,
                                  n_warmup=args.n_warmup, n_runs=args.n_runs)
                gpu_results[b] = r

                speedup_str = ""
                if b in cpu_results:
                    speedup = (cpu_results[b]["per_ms_median"]
                               / r["per_ms_median"])
                    speedup_str = f"{speedup:>7.1f}x"

                print(f"  {b:>6}  {r['iterations']:>6}  "
                      f"{str(r['converged']):>5}  "
                      f"{r['total_median_ms']:>10.2f}  "
                      f"{r['per_ms_median']:>11.4f}  "
                      f"{r['throughput']:>12.0f}  {speedup_str:>8}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  {b:>6}  -- OOM --")
                break

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Summary: GPU throughput scaling")
    print("=" * 70)
    if gpu_results:
        print(f"  {'B':>6}  {'samples/s':>14}  {'per-env ms':>12}  {'vs B=1':>10}")
        base_tp = gpu_results[min(gpu_results)]["throughput"]
        for b, r in sorted(gpu_results.items()):
            scale = r["throughput"] / base_tp
            print(f"  {b:>6}  {r['throughput']:>14,.0f}  "
                  f"{r['per_ms_median']:>12.4f}  {scale:>9.1f}x")
    else:
        print("  (no GPU results)")

    if cpu_results and gpu_results:
        print()
        print("  GPU vs CPU speedup (per-sample, shared B values):")
        shared = sorted(set(cpu_results) & set(gpu_results))
        for b in shared:
            speedup = (cpu_results[b]["per_ms_median"]
                       / gpu_results[b]["per_ms_median"])
            print(f"    B={b:>5}: {speedup:.1f}x  "
                  f"(CPU {cpu_results[b]['per_ms_median']:.3f} ms  "
                  f"GPU {gpu_results[b]['per_ms_median']:.4f} ms)")

    print()
    print("=" * 70)
    print("  Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
