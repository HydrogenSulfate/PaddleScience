import numpy as np
import paddle

# 设置随机种子以保证可复现性
paddle.seed(42)
np.random.seed(42)
paddle.framework.core.set_prim_eager_enabled(True)


def check_accuracy(paddle_result, numpy_result, test_name, rtol=1e-3):
    """检查paddle和numpy结果的精度"""
    paddle_np = (
        paddle_result.numpy()
        if isinstance(paddle_result, paddle.Tensor)
        else paddle_result
    )
    if np.iscomplexobj(paddle_np) or np.iscomplexobj(numpy_result):
        # 对于复数，分别比较实部和虚部
        real_close = np.allclose(paddle_np.real, numpy_result.real, rtol=rtol)
        imag_close = np.allclose(paddle_np.imag, numpy_result.imag, rtol=rtol)
        passed = real_close and imag_close
    else:
        passed = np.allclose(paddle_np, numpy_result, rtol=rtol)

    if passed:
        print(f"✓ {test_name} 精度测试通过 (相对误差 < {rtol})")
    else:
        print(f"✗ {test_name} 精度测试失败")
        print(f"  Paddle结果: {paddle_np}")
        print(f"  Numpy结果: {numpy_result}")
        max_error = np.max(np.abs((paddle_np - numpy_result) / (numpy_result + 1e-10)))
        print(f"  最大相对误差: {max_error}")
    return passed


def test_autograd_jacobian():
    """测试反向计算雅可比矩阵"""
    print("\n=== 测试 paddle.autograd.jacobian ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32", stop_gradient=False)
    y = x**2
    jacobian = paddle.autograd.jacobian(y, x, batch_axis=0)
    print(f"输入: {x.numpy()}")
    print(f"输出: {y.numpy()}")
    print(f"雅可比矩阵: {jacobian.numpy()}")

    # Numpy验证: y = x^2, dy/dx = 2x (对角矩阵)
    x_np = x.numpy()
    # 对于每个样本,雅可比矩阵是对角的
    numpy_jacobian = np.zeros((2, 2, 2))
    for i in range(2):
        numpy_jacobian[i] = np.diag(2 * x_np[i])

    check_accuracy(jacobian, numpy_jacobian, "autograd.jacobian")


def test_autograd_hessian():
    """测试海森矩阵"""
    print("\n=== 测试 paddle.autograd.hessian ===")
    x = paddle.to_tensor([1.0, 2.0], dtype="float32", stop_gradient=False)
    y = paddle.sum(x**3)
    hessian = paddle.autograd.hessian(y, x)
    print(f"输入: {x.numpy()}")
    print(f"输出: {y.numpy()}")
    print(f"海森矩阵: {hessian.numpy()}")

    # Numpy验证: y = sum(x^3), dy/dx_i = 3*x_i^2, d²y/dx_i² = 6*x_i
    x_np = x.numpy()
    numpy_hessian = np.diag(6 * x_np)

    check_accuracy(hessian, numpy_hessian, "autograd.hessian")


def test_add():
    """测试逐元素加法"""
    print("\n=== 测试 paddle.add ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = paddle.to_tensor([4.0, 5.0, 6.0])
    result = paddle.add(x, y)
    print(f"x + y = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = x_np + y_np

    check_accuracy(result, numpy_result, "add")


def test_subtract():
    """测试逐元素减法"""
    print("\n=== 测试 paddle.subtract ===")
    x = paddle.to_tensor([5.0, 6.0, 7.0])
    y = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.subtract(x, y)
    print(f"x - y = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = x_np - y_np

    check_accuracy(result, numpy_result, "subtract")


def test_multiply():
    """测试逐元素乘法"""
    print("\n=== 测试 paddle.multiply ===")
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    y = paddle.to_tensor([5.0, 6.0, 7.0])
    result = paddle.multiply(x, y)
    print(f"x * y = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = x_np * y_np

    check_accuracy(result, numpy_result, "multiply")


def test_divide():
    """测试逐元素除法"""
    print("\n=== 测试 paddle.divide ===")
    x = paddle.to_tensor([10.0, 15.0, 20.0])
    y = paddle.to_tensor([2.0, 3.0, 4.0])
    result = paddle.divide(x, y)
    print(f"x / y = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = x_np / y_np

    check_accuracy(result, numpy_result, "divide")


def test_tanh():
    """测试逐元素双曲正切函数"""
    print("\n=== 测试 paddle.tanh ===")
    x = paddle.to_tensor([-1.0, 0.0, 1.0, 2.0])
    result = paddle.tanh(x)
    print(f"tanh(x) = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.tanh(x_np)

    check_accuracy(result, numpy_result, "tanh")


def test_sin():
    """测试逐元素正弦函数"""
    print("\n=== 测试 paddle.sin ===")
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = paddle.sin(x)
    print(f"sin(x) = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.sin(x_np)

    check_accuracy(result, numpy_result, "sin")


def test_cos():
    """测试逐元素余弦函数"""
    print("\n=== 测试 paddle.cos ===")
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = paddle.cos(x)
    print(f"cos(x) = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.cos(x_np)

    check_accuracy(result, numpy_result, "cos")


def test_sigmoid():
    """测试逐元素Sigmoid函数"""
    print("\n=== 测试 paddle.nn.functional.sigmoid ===")
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = paddle.nn.functional.sigmoid(x)
    print(f"sigmoid(x) = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = 1 / (1 + np.exp(-x_np))

    check_accuracy(result, numpy_result, "sigmoid")


def test_matmul():
    """测试矩阵乘法"""
    print("\n=== 测试 paddle.matmul ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = paddle.matmul(x, y)
    print(f"矩阵乘法结果:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = np.matmul(x_np, y_np)

    check_accuracy(result, numpy_result, "matmul")


def test_pow():
    """测试矩阵幂运算"""
    print("\n=== 测试 paddle.pow ===")
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    result = paddle.pow(x, 3)
    print(f"x^3 = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.power(x_np, 3)

    check_accuracy(result, numpy_result, "pow")


def test_assign():
    """测试张量拷贝"""
    print("\n=== 测试 paddle.assign ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = paddle.assign(x)
    print(f"原始张量: {x.numpy()}")
    print(f"拷贝张量: {y.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np.copy()

    check_accuracy(y, numpy_result, "assign")


def test_concat():
    """测试张量拼接"""
    print("\n=== 测试 paddle.concat ===")
    x1 = paddle.to_tensor([[1.0, 2.0]])
    x2 = paddle.to_tensor([[3.0, 4.0]])
    result = paddle.concat([x1, x2], axis=0)
    print(f"拼接结果:\n{result.numpy()}")

    # Numpy验证
    x1_np = x1.numpy()
    x2_np = x2.numpy()
    numpy_result = np.concatenate([x1_np, x2_np], axis=0)

    check_accuracy(result, numpy_result, "concat")


def test_expand():
    """测试张量扩展"""
    print("\n=== 测试 paddle.expand ===")
    x = paddle.to_tensor([[1.0, 2.0]])
    result = paddle.expand(x, shape=[3, 2])
    print(f"扩展结果:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.broadcast_to(x_np, (3, 2))

    check_accuracy(result, numpy_result, "expand")


def test_squeeze():
    """测试张量删除维度"""
    print("\n=== 测试 paddle.squeeze ===")
    x = paddle.to_tensor([[[1.0, 2.0, 3.0]]])
    result = paddle.squeeze(x)
    print(f"原始形状: {x.shape}, 压缩后形状: {result.shape}")
    print(f"压缩结果: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.squeeze(x_np)

    check_accuracy(result, numpy_result, "squeeze")


def test_unsqueeze():
    """测试张量增加维度"""
    print("\n=== 测试 paddle.unsqueeze ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.unsqueeze(x, axis=0)
    print(f"原始形状: {x.shape}, 增维后形状: {result.shape}")
    print(f"增维结果: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.expand_dims(x_np, axis=0)

    check_accuracy(result, numpy_result, "unsqueeze")


def test_scale():
    """测试张量缩放和偏置"""
    print("\n=== 测试 paddle.scale ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.scale(x, scale=2.0, bias=1.0)
    print(f"x * 2 + 1 = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np * 2.0 + 1.0

    check_accuracy(result, numpy_result, "scale")


def test_transpose():
    """测试张量重排"""
    print("\n=== 测试 paddle.transpose ===")
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = paddle.transpose(x, perm=[1, 0])
    print(f"原始形状: {x.shape}, 转置后形状: {result.shape}")
    print(f"转置结果:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.transpose(x_np, (1, 0))

    check_accuracy(result, numpy_result, "transpose")


def test_sign():
    """测试张量逐元素正负判断"""
    print("\n=== 测试 paddle.sign ===")
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = paddle.sign(x)
    print(f"sign(x) = {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.sign(x_np)

    check_accuracy(result, numpy_result, "sign")


def test_cast():
    """测试张量类型转换"""
    print("\n=== 测试 paddle.cast ===")
    x = paddle.to_tensor([1.5, 2.7, 3.2])
    result = paddle.cast(x, dtype="int32")
    print(f"float32 -> int32: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np.astype(np.int32)

    check_accuracy(result, numpy_result, "cast")


def test_slice():
    """测试张量切片"""
    print("\n=== 测试 paddle.slice ===")
    x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = paddle.slice(x, axes=[0, 1], starts=[0, 1], ends=[2, 3])
    print(f"切片结果:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np[0:2, 1:3]

    check_accuracy(result, numpy_result, "slice")


def test_sum():
    """测试张量求和"""
    print("\n=== 测试 paddle.sum ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.sum(x)
    print(f"总和: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.sum(x_np)

    check_accuracy(result, numpy_result, "sum")


def test_mean():
    """测试张量均值"""
    print("\n=== 测试 paddle.mean ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.mean(x)
    print(f"均值: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.mean(x_np)

    check_accuracy(result, numpy_result, "mean")


def test_is_complex():
    """测试判断是否是复数"""
    print("\n=== 测试 paddle.is_complex ===")
    x_real = paddle.to_tensor([1.0, 2.0])
    x_complex = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    print(f"实数张量是复数: {paddle.is_complex(x_real)}")
    print(f"复数张量是复数: {paddle.is_complex(x_complex)}")

    # Numpy验证
    x_real_np = x_real.numpy()
    x_complex_np = x_complex.numpy()
    numpy_result_real = np.iscomplexobj(x_real_np)
    numpy_result_complex = np.iscomplexobj(x_complex_np)

    passed = (paddle.is_complex(x_real) == numpy_result_real) and (
        paddle.is_complex(x_complex) == numpy_result_complex
    )
    if passed:
        print("✓ is_complex 精度测试通过")
    else:
        print("✗ is_complex 精度测试失败")


def test_angle():
    """测试计算相位角"""
    print("\n=== 测试 paddle.angle ===")
    x = paddle.to_tensor([1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j])
    result = paddle.angle(x)
    print(f"相位角: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.angle(x_np)

    check_accuracy(result, numpy_result, "angle")


def test_polar():
    """测试由极坐标表示计算复平面坐标"""
    print("\n=== 测试 paddle.polar ===")
    abs_val = paddle.to_tensor([1.0, 2.0])
    angle_val = paddle.to_tensor([np.pi / 4, np.pi / 2])
    result = paddle.polar(abs_val, angle_val)
    print(f"极坐标转复数: {result.numpy()}")

    # Numpy验证
    abs_np = abs_val.numpy()
    angle_np = angle_val.numpy()
    numpy_result = abs_np * np.exp(1j * angle_np)

    check_accuracy(result, numpy_result, "polar")


def test_mod_complex():
    """测试复数取模"""
    print("\n=== 测试 paddle.mod (复数) ===")
    x = paddle.to_tensor([3.0 + 4.0j, 5.0 + 12.0j], dtype="complex64")
    y = paddle.to_tensor([2.0 + 1.0j, 3.0 + 2.0j], dtype="complex64")
    result = paddle.mod(x, y)
    print(f"复数取模: {result.numpy()}")

    # 手动计算复数取模 (基于高斯整数理论)
    # 对于复数 x, y, x mod y = x - y * round(x/y)
    # 其中 round 是四舍五入到最近的高斯整数
    x_np = x.numpy()
    y_np = y.numpy()

    # 计算 x / y
    quotient = x_np / y_np

    # 四舍五入实部和虚部到最近的整数
    rounded_quotient = np.round(quotient.real) + 1j * np.round(quotient.imag)

    # 计算余数: x - y * round(x/y)
    numpy_result = x_np - y_np * rounded_quotient

    check_accuracy(result, numpy_result, "mod(complex)")


def test_as_real():
    """测试复数转换为实数"""
    print("\n=== 测试 paddle.as_real ===")
    x = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    result = paddle.as_real(x)
    print(f"复数转实数:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = np.stack([x_np.real, x_np.imag], axis=-1)

    check_accuracy(result, numpy_result, "as_real")


def test_as_complex():
    """测试实数转化为复数"""
    print("\n=== 测试 paddle.as_complex ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.as_complex(x)
    print(f"实数转复数: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np[..., 0] + 1j * x_np[..., 1]

    check_accuracy(result, numpy_result, "as_complex")


def test_einsum_complex():
    """测试复数爱因斯坦求和"""
    print("\n=== 测试 paddle.einsum (复数) ===")
    x = paddle.to_tensor([1.0 + 1.0j, 2.0 + 2.0j])
    y = paddle.to_tensor([3.0 + 1.0j, 4.0 + 2.0j])
    result = paddle.einsum("i,i->", x, y)
    print(f"复数爱因斯坦求和: {result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    y_np = y.numpy()
    numpy_result = np.einsum("i,i->", x_np, y_np)

    check_accuracy(result, numpy_result, "einsum(complex)")


def test_uniform_complex():
    """测试以均匀分布随机数创建稠密复数张量"""
    print("\n=== 测试 paddle.uniform (复数) ===")
    # 固定种子以便比较
    paddle.seed(42)
    result = paddle.uniform([2, 3], dtype="complex64", min=-1.0, max=1.0)
    print(f"均匀分布复数张量:\n{result.numpy()}")

    # 验证范围
    result_np = result.numpy()
    real_in_range = np.all((result_np.real >= -1.0) & (result_np.real <= 1.0))
    imag_in_range = np.all((result_np.imag >= -1.0) & (result_np.imag <= 1.0))

    if real_in_range and imag_in_range:
        print("✓ uniform(complex) 范围测试通过")
    else:
        print("✗ uniform(complex) 范围测试失败")


def test_full_complex():
    """测试创建元素值全为指定复数的稠密复数张量"""
    print("\n=== 测试 paddle.full (复数) ===")
    result = paddle.full([2, 3], fill_value=1.0 + 2.0j, dtype="complex64")
    print(f"全为指定复数的张量:\n{result.numpy()}")

    # Numpy验证
    numpy_result = np.full([2, 3], 1.0 + 2.0j, dtype=np.complex64)

    check_accuracy(result, numpy_result, "full(complex)")


def test_zeros_complex():
    """测试创建元素值全为0+0i的稠密复数张量"""
    print("\n=== 测试 paddle.zeros (复数) ===")
    result = paddle.zeros([2, 3], dtype="complex64")
    print(f"全零复数张量:\n{result.numpy()}")

    # Numpy验证
    numpy_result = np.zeros([2, 3], dtype=np.complex64)

    check_accuracy(result, numpy_result, "zeros(complex)")


def test_slice_complex():
    """测试复数张量切片"""
    print("\n=== 测试 paddle.slice (复数) ===")
    x = paddle.to_tensor(
        [[1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], [4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j]],
        dtype="complex64",
    )
    result = paddle.slice(x, axes=[0, 1], starts=[0, 0], ends=[1, 2])
    print(f"复数张量切片:\n{result.numpy()}")

    # Numpy验证
    x_np = x.numpy()
    numpy_result = x_np[0:1, 0:2]

    check_accuracy(result, numpy_result, "slice(complex)")


def test_rfft2():
    """测试二维离散傅里叶变换"""
    print("\n=== 测试 paddle.fft.rfft2 ===")
    # 使用固定输入以便验证
    paddle.seed(42)
    np.random.seed(42)
    x_np = np.random.randn(4, 16, 16, 3).astype(np.float32)
    x = paddle.to_tensor(x_np)
    result = paddle.fft.rfft2(x, axes=(1, 2), norm="ortho")
    print(f"二维FFT结果形状: {result.shape}")

    # Numpy验证
    numpy_result = np.fft.rfft2(x_np, axes=(1, 2), norm="ortho")

    check_accuracy(result, numpy_result, "fft.rfft2")


def test_irfft2():
    """测试二维离散逆傅里叶变换"""
    print("\n=== 测试 paddle.fft.irfft2 ===")
    B, H, W, C = 4, 16, 16, 3
    # 使用固定输入
    paddle.seed(42)
    np.random.seed(42)
    x_np = (
        np.random.randn(B, H, W // 2 + 1, C) + 1j * np.random.randn(B, H, W // 2 + 1, C)
    ).astype(np.complex64)
    x = paddle.to_tensor(x_np)
    result = paddle.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")
    print(f"二维逆FFT结果形状: {result.shape}")

    # Numpy验证
    numpy_result = np.fft.irfft2(x_np, s=(H, W), axes=(1, 2), norm="ortho")

    check_accuracy(result, numpy_result, "fft.irfft2")


def test_norm():
    """测试 paddle.linalg.norm 的基本用法"""
    print("\n=== 测试 paddle.linalg.norm ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    x_np = x.numpy()

    # 默认：Frobenius 范数（对矩阵）
    norm_default = paddle.linalg.norm(x)
    print(f"默认范数 (Frobenius): {norm_default.numpy()}")
    numpy_default = np.linalg.norm(x_np, ord="fro")
    check_accuracy(norm_default, numpy_default, "linalg.norm(default)")


def test_ppsci_fractional_diff():
    """测试分数阶微分(需要安装PaddleScience)"""
    print("\n=== 测试 ppsci.experimental.fractional_diff ===")
    try:
        from scipy.special import gamma

        import ppsci

        def f(x):
            return x * x

        result = ppsci.experimental.fractional_diff(
            f, alpha=0.5, a=0, t=1.0, h=1e-5, dtype="float64"  # 使用float64提高精度
        )
        print(f"分数阶微分结果: {result.numpy()}")

        # 对于幂函数 f(x) = x^n 的Caputo分数阶导数有解析公式:
        # D^α[x^n] = Gamma(n+1)/Gamma(n+1-α) * x^(n-α)
        #
        # 对于 f(x) = x^2, n=2, α=0.5, t=1:
        # D^0.5[x^2](1) = Gamma(3)/Gamma(2.5) * 1^1.5
        #             = 2/Gamma(2.5)

        n = 2
        alpha = 0.5
        t = 1.0

        # 使用解析公式，使用float64保持精度
        analytical = gamma(n + 1) / gamma(n + 1 - alpha) * (t ** (n - alpha))
        numpy_result = np.array(analytical, dtype=np.float64)

        print(f"  理论值: {numpy_result:.10f}")

        check_accuracy(result, numpy_result, "ppsci.fractional_diff")

    except ImportError:
        print("请安装 PaddleScience: pip install ppsci")
    except Exception as e:
        print(f"执行出错: {e}")
        raise e


def test_ppsci_montecarlo_integrate():
    """测试蒙特卡洛积分(需要安装PaddleScience)"""
    print("\n=== 测试 ppsci.experimental.montecarlo_integrate ===")
    try:
        import ppsci

        # 使用 f(x,y) = x^2 + y^2
        def some_function(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2

        # 固定种子
        paddle.seed(42)
        np.random.seed(42)
        result = ppsci.experimental.montecarlo_integrate(
            some_function,
            dim=2,
            N=100000,  # 增加采样点数
            integration_domain=[[0, 1], [0, 1]],
            seed=42,
        )
        print(f"蒙特卡洛积分结果: {result}")

        # 解析解: ∫₀¹∫₀¹ (x^2 + y^2) dy dx
        # = ∫₀¹ (x^2 + 1/3) dx
        # = 1/3 + 1/3 = 2/3
        analytical = 2.0 / 3.0
        numpy_result = np.array(analytical, dtype=np.float32)

        print(f"  理论值: {numpy_result:.6f}")

        # 蒙特卡洛方法有随机性，使用更宽松的误差
        check_accuracy(result, numpy_result, "ppsci.montecarlo_integrate", rtol=0.01)

    except ImportError:
        print("请安装 PaddleScience: pip install ppsci")
    except Exception as e:
        print(f"执行出错: {e}")
        raise e


def test_ppsci_gaussian_integrate():
    """测试高斯积分(需要安装PaddleScience)"""
    print("\n=== 测试 ppsci.experimental.gaussian_integrate ===")
    try:
        import ppsci

        # 使用 f(x) = x^2
        func = lambda x: x**2
        dim = 1
        N = 500
        integration_domains = [[0, 1]]
        result = ppsci.experimental.gaussian_integrate(
            func, dim, N, integration_domains, dtype="float32"
        )
        print(f"高斯积分结果: {result}")

        # 解析解: ∫₀¹ x^2 dx = 1/3
        analytical = 1.0 / 3.0
        numpy_result = np.array(analytical, dtype=np.float32)

        print(f"  理论值: {numpy_result:.6f}")

        check_accuracy(result, numpy_result, "ppsci.gaussian_integrate")

    except ImportError:
        print("请安装 PaddleScience: pip install ppsci")
    except Exception as e:
        print(f"执行出错: {e}")
        raise e


def test_ppsci_trapezoid_integrate():
    """测试梯形公式积分(需要安装PaddleScience)"""
    print("\n=== 测试 ppsci.experimental.trapezoid_integrate ===")
    try:
        import ppsci

        y = paddle.to_tensor([[0, 1, 2], [3, 4, 5]], dtype="float32")
        result = ppsci.experimental.trapezoid_integrate(y)
        print(f"梯形积分结果: {result.numpy()}")

        # Numpy验证
        y_np = y.numpy()
        numpy_result = np.trapz(y_np, axis=-1)

        check_accuracy(result, numpy_result, "ppsci.trapezoid_integrate")

    except ImportError:
        print("请安装 PaddleScience: pip install ppsci")
    except Exception as e:
        print(f"执行出错: {e}")
        raise e


def main():
    """主函数,运行所有测试"""
    print("=" * 60)
    print("开始测试 42 个 Paddle 算子")
    print("=" * 60)

    test_functions = [
        test_autograd_jacobian,
        test_autograd_hessian,
        test_add,
        test_subtract,
        test_multiply,
        test_divide,
        test_tanh,
        test_sin,
        test_cos,
        test_sigmoid,
        test_matmul,
        test_pow,
        test_assign,
        test_concat,
        test_expand,
        test_squeeze,
        test_unsqueeze,
        test_scale,
        test_transpose,
        test_sign,
        test_cast,
        test_slice,
        test_sum,
        test_mean,
        test_is_complex,
        test_angle,
        test_polar,
        test_mod_complex,
        test_as_real,
        test_as_complex,
        test_einsum_complex,
        test_uniform_complex,
        test_full_complex,
        test_zeros_complex,
        test_slice_complex,
        test_rfft2,
        test_irfft2,
        test_norm,
        test_ppsci_fractional_diff,
        test_ppsci_montecarlo_integrate,
        test_ppsci_gaussian_integrate,
        test_ppsci_trapezoid_integrate,
    ]

    success_count = 0
    fail_count = 0

    for test_func in test_functions:
        try:
            test_func()
            success_count += 1
        except Exception as e:
            print(f"\n错误: {test_func.__name__} 执行失败")
            print(f"错误信息: {str(e)}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"测试完成! 成功: {success_count}, 失败: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
