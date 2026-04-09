import time

import numpy as np
import paddle

# 设置随机种子以保证可复现性
paddle.seed(42)
np.random.seed(42)
paddle.framework.core.set_prim_eager_enabled(True)

# ===================== 性能测试配置 =====================
WARMUP_RUNS = 20  # 预热次数（排除 JIT/初始化影响）
BENCH_RUNS = 100  # 正式测量次数


def benchmark(func, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """
    通用性能测试函数。
    返回 (mean_ms, std_ms)，单位毫秒，保留两位小数。
    """
    # --- 预热 ---
    for _ in range(warmup):
        func()
    if paddle.is_compiled_with_cuda():
        paddle.device.synchronize()

    # --- 正式计时 ---
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func()
        if paddle.is_compiled_with_cuda():
            paddle.device.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    mean = round(float(np.mean(timings)), 2)
    std = round(float(np.std(timings)), 2)
    return mean, std


def print_result(name, mean, std):
    print(f"  {name:<52}  均值: {mean:>8.2f} ms  标准差: {std:>7.2f} ms")


# ===================== 各算子 Benchmark 函数 =====================


def bench_autograd_jacobian():
    def fn():
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype="float32", stop_gradient=False
        )
        y = x**2
        paddle.autograd.jacobian(y, x, batch_axis=0)

    return benchmark(fn)


def bench_autograd_hessian():
    def fn():
        x = paddle.to_tensor([1.0, 2.0], dtype="float32", stop_gradient=False)
        y = paddle.sum(x**3)
        paddle.autograd.hessian(y, x)

    return benchmark(fn)


def bench_add():
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = paddle.to_tensor([4.0, 5.0, 6.0])
    return benchmark(lambda: paddle.add(x, y))


def bench_subtract():
    x = paddle.to_tensor([5.0, 6.0, 7.0])
    y = paddle.to_tensor([1.0, 2.0, 3.0])
    return benchmark(lambda: paddle.subtract(x, y))


def bench_multiply():
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    y = paddle.to_tensor([5.0, 6.0, 7.0])
    return benchmark(lambda: paddle.multiply(x, y))


def bench_divide():
    x = paddle.to_tensor([10.0, 15.0, 20.0])
    y = paddle.to_tensor([2.0, 3.0, 4.0])
    return benchmark(lambda: paddle.divide(x, y))


def bench_tanh():
    x = paddle.to_tensor([-1.0, 0.0, 1.0, 2.0])
    return benchmark(lambda: paddle.tanh(x))


def bench_sin():
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    return benchmark(lambda: paddle.sin(x))


def bench_cos():
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    return benchmark(lambda: paddle.cos(x))


def bench_sigmoid():
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    return benchmark(lambda: paddle.nn.functional.sigmoid(x))


def bench_matmul():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
    return benchmark(lambda: paddle.matmul(x, y))


def bench_pow():
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    return benchmark(lambda: paddle.pow(x, 3))


def bench_assign():
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    return benchmark(lambda: paddle.assign(x))


def bench_concat():
    x1 = paddle.to_tensor([[1.0, 2.0]])
    x2 = paddle.to_tensor([[3.0, 4.0]])
    return benchmark(lambda: paddle.concat([x1, x2], axis=0))


def bench_expand():
    x = paddle.to_tensor([[1.0, 2.0]])
    return benchmark(lambda: paddle.expand(x, shape=[3, 2]))


def bench_squeeze():
    x = paddle.to_tensor([[[1.0, 2.0, 3.0]]])
    return benchmark(lambda: paddle.squeeze(x))


def bench_unsqueeze():
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    return benchmark(lambda: paddle.unsqueeze(x, axis=0))


def bench_scale():
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    return benchmark(lambda: paddle.scale(x, scale=2.0, bias=1.0))


def bench_transpose():
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return benchmark(lambda: paddle.transpose(x, perm=[1, 0]))


def bench_sign():
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    return benchmark(lambda: paddle.sign(x))


def bench_cast():
    x = paddle.to_tensor([1.5, 2.7, 3.2])
    return benchmark(lambda: paddle.cast(x, dtype="int32"))


def bench_slice():
    x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    return benchmark(lambda: paddle.slice(x, axes=[0, 1], starts=[0, 1], ends=[2, 3]))


def bench_sum():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    return benchmark(lambda: paddle.sum(x))


def bench_mean():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    return benchmark(lambda: paddle.mean(x))


def bench_is_complex():
    x_real = paddle.to_tensor([1.0, 2.0])
    x_complex = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    return benchmark(lambda: (paddle.is_complex(x_real), paddle.is_complex(x_complex)))


def bench_angle():
    x = paddle.to_tensor([1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j])
    return benchmark(lambda: paddle.angle(x))


def bench_polar():
    abs_val = paddle.to_tensor([1.0, 2.0])
    angle_val = paddle.to_tensor([np.pi / 4, np.pi / 2])
    return benchmark(lambda: paddle.polar(abs_val, angle_val))


def bench_mod_complex():
    x = paddle.to_tensor([3.0 + 4.0j, 5.0 + 12.0j], dtype="complex64")
    y = paddle.to_tensor([2.0 + 1.0j, 3.0 + 2.0j], dtype="complex64")
    return benchmark(lambda: paddle.mod(x, y))


def bench_as_real():
    x = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    return benchmark(lambda: paddle.as_real(x))


def bench_as_complex():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    return benchmark(lambda: paddle.as_complex(x))


def bench_einsum_complex():
    x = paddle.to_tensor([1.0 + 1.0j, 2.0 + 2.0j])
    y = paddle.to_tensor([3.0 + 1.0j, 4.0 + 2.0j])
    return benchmark(lambda: paddle.einsum("i,i->", x, y))


def bench_uniform_complex():
    return benchmark(
        lambda: paddle.uniform([2, 3], dtype="complex64", min=-1.0, max=1.0)
    )


def bench_full_complex():
    return benchmark(
        lambda: paddle.full([2, 3], fill_value=1.0 + 2.0j, dtype="complex64")
    )


def bench_zeros_complex():
    return benchmark(lambda: paddle.zeros([2, 3], dtype="complex64"))


def bench_slice_complex():
    x = paddle.to_tensor(
        [[1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], [4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j]],
        dtype="complex64",
    )
    return benchmark(lambda: paddle.slice(x, axes=[0, 1], starts=[0, 0], ends=[1, 2]))


def bench_rfft2():
    paddle.seed(42)
    np.random.seed(42)
    x = paddle.to_tensor(np.random.randn(4, 16, 16, 3).astype(np.float32))
    return benchmark(lambda: paddle.fft.rfft2(x, axes=(1, 2), norm="ortho"))


def bench_irfft2():
    B, H, W, C = 4, 16, 16, 3
    paddle.seed(42)
    np.random.seed(42)
    x_np = (
        np.random.randn(B, H, W // 2 + 1, C) + 1j * np.random.randn(B, H, W // 2 + 1, C)
    ).astype(np.complex64)
    x = paddle.to_tensor(x_np)
    return benchmark(lambda: paddle.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho"))


def bench_norm():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    return benchmark(lambda: paddle.linalg.norm(x))


# ---- PaddleScience 算子（可选，失败时跳过）----


def bench_ppsci_fractional_diff():
    try:
        import ppsci

        def f(x):
            return x * x

        return benchmark(
            lambda: ppsci.experimental.fractional_diff(
                f, alpha=0.5, a=0, t=1.0, h=1e-5, dtype="float64"
            ),
            runs=10,
        )
    except Exception:
        return None, None


def bench_ppsci_montecarlo_integrate():
    try:
        import ppsci

        def some_function(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2

        return benchmark(
            lambda: ppsci.experimental.montecarlo_integrate(
                some_function,
                dim=2,
                N=100000,
                integration_domain=[[0, 1], [0, 1]],
                seed=42,
            ),
            runs=10,
        )
    except Exception:
        return None, None


def bench_ppsci_gaussian_integrate():
    try:
        import ppsci

        func = lambda x: x**2
        return benchmark(
            lambda: ppsci.experimental.gaussian_integrate(
                func, 1, 500, [[0, 1]], dtype="float32"
            ),
            runs=10,
        )
    except Exception:
        return None, None


def bench_ppsci_trapezoid_integrate():
    try:
        import ppsci

        y = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="float32")
        return benchmark(lambda: ppsci.experimental.trapezoid_integrate(y), runs=10)
    except Exception:
        return None, None


# ===================== 主函数 =====================


def main():
    print("=" * 78)
    print("  PaddlePaddle 算子性能基准测试")
    print(f"  预热次数: {WARMUP_RUNS}   测量次数: {BENCH_RUNS}   时间单位: ms")
    print("=" * 78)

    benchmarks = [
        ("paddle.autograd.jacobian", bench_autograd_jacobian),
        ("paddle.autograd.hessian", bench_autograd_hessian),
        ("paddle.add", bench_add),
        ("paddle.subtract", bench_subtract),
        ("paddle.multiply", bench_multiply),
        ("paddle.divide", bench_divide),
        ("paddle.tanh", bench_tanh),
        ("paddle.sin", bench_sin),
        ("paddle.cos", bench_cos),
        ("paddle.nn.functional.sigmoid", bench_sigmoid),
        ("paddle.matmul", bench_matmul),
        ("paddle.pow", bench_pow),
        ("paddle.assign", bench_assign),
        ("paddle.concat", bench_concat),
        ("paddle.expand", bench_expand),
        ("paddle.squeeze", bench_squeeze),
        ("paddle.unsqueeze", bench_unsqueeze),
        ("paddle.scale", bench_scale),
        ("paddle.transpose", bench_transpose),
        ("paddle.sign", bench_sign),
        ("paddle.cast", bench_cast),
        ("paddle.slice", bench_slice),
        ("paddle.sum", bench_sum),
        ("paddle.mean", bench_mean),
        ("paddle.is_complex", bench_is_complex),
        ("paddle.angle", bench_angle),
        ("paddle.polar", bench_polar),
        ("paddle.mod (complex)", bench_mod_complex),
        ("paddle.as_real", bench_as_real),
        ("paddle.as_complex", bench_as_complex),
        ("paddle.einsum (complex)", bench_einsum_complex),
        ("paddle.uniform (complex)", bench_uniform_complex),
        ("paddle.full (complex)", bench_full_complex),
        ("paddle.zeros (complex)", bench_zeros_complex),
        ("paddle.slice (complex)", bench_slice_complex),
        ("paddle.fft.rfft2", bench_rfft2),
        ("paddle.fft.irfft2", bench_irfft2),
        ("paddle.linalg.norm", bench_norm),
        ("ppsci.experimental.fractional_diff", bench_ppsci_fractional_diff),
        ("ppsci.experimental.montecarlo_integrate", bench_ppsci_montecarlo_integrate),
        ("ppsci.experimental.gaussian_integrate", bench_ppsci_gaussian_integrate),
        ("ppsci.experimental.trapezoid_integrate", bench_ppsci_trapezoid_integrate),
    ]

    success_count = 0
    skip_count = 0
    fail_count = 0

    for name, bench_fn in benchmarks:
        try:
            mean, std = bench_fn()
            if mean is None:
                print(f"  [SKIP] {name:<52}  (依赖未安装，已跳过)")
                skip_count += 1
            else:
                print_result(f"[OK]   {name}", mean, std)
                success_count += 1
        except Exception as e:
            print(f"  [FAIL] {name:<52}  错误: {e}")
            fail_count += 1

    print()
    print("=" * 78)
    print(f"  完成!  成功: {success_count}  跳过: {skip_count}  失败: {fail_count}")
    print("=" * 78)


if __name__ == "__main__":
    main()
