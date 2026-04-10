"""
PiMPC Julia Benchmark & Verification
======================================
Tests the Julia solver on the AFTI-16 aircraft example,
benchmarks solve time, and produces results for comparison
with the PyTorch implementation.

Run from the PiMPC root:
    julia --project benchmark/benchmark_julia.jl [cpu|gpu]
"""

using PiMPC
using LinearAlgebra
using Printf
using Statistics

# -----------------------------------------------------------------------
# Parse device argument, fall back to CPU if GPU is unavailable
# -----------------------------------------------------------------------
_requested = length(ARGS) > 0 ? Symbol(ARGS[1]) : :cpu
device = if _requested == :gpu && PiMPC._cuda_ok()
    :gpu
elseif _requested == :gpu
    @warn "GPU requested but CUDA is not available – running on CPU."
    :cpu
else
    :cpu
end

# -----------------------------------------------------------------------
# AFTI-16 aircraft model
# -----------------------------------------------------------------------

function build_AFTI16(Ts=0.05)
    As = [-0.0151  -60.5651   0.0      -32.174;
          -0.0001   -1.3411   0.9929    0.0;
           0.00018  43.2541  -0.86939   0.0;
           0.0       0.0      1.0       0.0]
    Bs = [-2.516   -13.136;
          -0.1689   -0.2514;
         -17.251    -1.5766;
           0.0       0.0]
    nx, nu = 4, 2
    M = exp([As Bs; zeros(nu, nx) zeros(nu, nu)] * Ts)
    A = M[1:nx, 1:nx]
    B = M[1:nx, nx+1:end]
    C = [0.0 1.0 0.0 0.0;
         0.0 0.0 0.0 1.0]
    return A, B, C
end

# -----------------------------------------------------------------------
# Closed-loop simulation
# -----------------------------------------------------------------------

function simulate_closed_loop(model, A, B, C, x0, u0, yref_traj, uref, w, Nsim)
    nx, nu = size(B)
    ny     = size(C, 1)

    x_hist    = zeros(nx, Nsim + 1)
    y_hist    = zeros(ny, Nsim + 1)
    u_hist    = zeros(nu, Nsim)
    iter_hist = zeros(Int, Nsim)
    time_hist = zeros(Nsim)

    x_hist[:, 1] = x0
    y_hist[:, 1] = C * x0
    x_cur  = copy(x0)
    u_prev = copy(u0)

    for k in 1:Nsim
        yref   = yref_traj[:, k]
        res    = solve!(model, x_cur, u_prev, yref, uref, w)
        u_app  = res.u[:, 1]
        x_next = A * x_cur + B * u_app

        x_hist[:, k+1] = x_next
        y_hist[:, k+1] = C * x_next
        u_hist[:, k]   = u_app
        iter_hist[k]   = res.info.iterations
        time_hist[k]   = res.info.solve_time

        x_cur  = x_next
        u_prev = u_app
    end
    return x_hist, y_hist, u_hist, iter_hist, time_hist
end

# -----------------------------------------------------------------------
# Solve-time benchmark
# -----------------------------------------------------------------------

function benchmark_solve_time(model, A, B, x0, u0, yref, uref, w;
                               n_warmup=10, n_runs=200)
    x_cur  = copy(x0)
    u_prev = copy(u0)
    model.warm_vars = nothing   # cold start

    # Warm-up (JIT + cache)
    for _ in 1:n_warmup
        res    = solve!(model, x_cur, u_prev, yref, uref, w)
        u_prev = res.u[:, 1]
        x_cur  = A * x_cur + B * u_prev
    end

    # Timed runs
    times = zeros(n_runs)
    x_cur  = copy(x0)
    u_prev = copy(u0)
    model.warm_vars = nothing

    for i in 1:n_runs
        res      = solve!(model, x_cur, u_prev, yref, uref, w)
        times[i] = res.info.solve_time * 1000.0   # ms
        u_prev   = res.u[:, 1]
        x_cur    = A * x_cur + B * u_prev
    end
    return Dict(
        "median_ms" => median(times),
        "mean_ms"   => mean(times),
        "std_ms"    => std(times),
        "min_ms"    => minimum(times),
        "max_ms"    => maximum(times),
    )
end

# -----------------------------------------------------------------------
# Trajectory comparison
# -----------------------------------------------------------------------

function compare_trajectories(x1, u1, x2, u2; label1="m1", label2="m2")
    dx = abs.(x1 .- x2)
    du = abs.(u1 .- u2)
    @printf("  Trajectory comparison (%s vs %s):\n", label1, label2)
    @printf("    x  max-abs: %.4e   mean: %.4e\n", maximum(dx), mean(dx))
    @printf("    u  max-abs: %.4e   mean: %.4e\n", maximum(du), mean(du))
    return maximum(dx), maximum(du)
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

println("=" ^ 60)
println("  PiMPC Julia Benchmark")
@printf("  device=%s\n", device)
println("=" ^ 60)

Ts    = 0.05
A, B, C = build_AFTI16(Ts)
nx, nu, ny = 4, 2, 2

Np   = 5
Nsim = 200
x0   = zeros(nx)
u0   = zeros(nu)
uref = zeros(nu)
w    = zeros(nx)

yref_traj = [zeros(1, Nsim);
             [10.0 * ones(1, 100) zeros(1, 100)]]

common_kwargs = (
    A=A, B=B, C=C, Np=Np,
    Wy=100.0 * diagm(ones(ny)),
    Wu=0.0   * diagm(ones(nu)),
    Wdu=0.1  * diagm(ones(nu)),
    umin=-25.0 * ones(nu),
    umax= 25.0 * ones(nu),
    xmin=[-Inf, -0.5, -Inf, -100.0],
    xmax=[ Inf,  0.5,  Inf,  100.0],
    rho=1.0, tol=1e-6,
    maxiter=10000,
    precond=true,
    device=device,
)

println("\n[1] Closed-loop simulation (no acceleration)")
m_base = Model()
setup!(m_base; common_kwargs..., accel=false)
x_b, y_b, u_b, iters_b, times_b = simulate_closed_loop(
    m_base, A, B, C, x0, u0, yref_traj, uref, w, Nsim)
@printf("  avg iters : %.1f\n",  mean(iters_b))
@printf("  avg time  : %.3f ms/step\n", mean(times_b) * 1000)
@printf("  final y2  : %.6f  (target 0.0)\n", y_b[2, end])

println("\n[2] Closed-loop simulation (Nesterov acceleration)")
m_accel = Model()
setup!(m_accel; common_kwargs..., accel=true)
x_a, y_a, u_a, iters_a, times_a = simulate_closed_loop(
    m_accel, A, B, C, x0, u0, yref_traj, uref, w, Nsim)
@printf("  avg iters : %.1f\n",  mean(iters_a))
@printf("  avg time  : %.3f ms/step\n", mean(times_a) * 1000)
@printf("  final y2  : %.6f  (target 0.0)\n", y_a[2, end])

println("\n[3] Trajectory comparison (base vs accel):")
compare_trajectories(x_b, u_b, x_a, u_a; label1="base", label2="accel")

println("\n[4] Solve-time benchmark (single solve, warm start)")
m_bench = Model()
setup!(m_bench; common_kwargs..., accel=true)
stats = benchmark_solve_time(m_bench, A, B, x0, u0, yref_traj[:, 1], uref, w;
                              n_warmup=20, n_runs=500)
@printf("  median : %.4f ms\n",  stats["median_ms"])
@printf("  mean   : %.4f ms  ±  %.4f ms\n", stats["mean_ms"], stats["std_ms"])
@printf("  min    : %.4f ms    max: %.4f ms\n", stats["min_ms"], stats["max_ms"])

println("\n[5] Convergence accuracy test")
tols   = [1e-3, 1e-4, 1e-6]
x0_t   = zeros(nx)
u0_t   = zeros(nu)
yref_t = [0.0, 10.0]

ref_model = Model()
setup!(ref_model; common_kwargs..., tol=1e-10, maxiter=50000, accel=true)
res_ref = solve!(ref_model, x0_t, u0_t, yref_t, uref, w)
u_ref   = res_ref.u
@printf("  Reference solve: %d iters, obj=%.2e\n",
        res_ref.info.iterations, res_ref.info.obj_val)

for tol in tols
    m_t = Model()
    setup!(m_t; common_kwargs..., tol=tol, accel=true)
    res_t = solve!(m_t, x0_t, u0_t, yref_t, uref, w)
    err   = maximum(abs.(res_t.u .- u_ref))
    @printf("  tol=%.0e : %5d iters, u1 err=%.4e, converged=%s\n",
            tol, res_t.info.iterations, err, res_t.info.converged ? "true" : "false")
end

println("\n" * "=" ^ 60)
println("  Benchmark complete.")
println("=" ^ 60)
