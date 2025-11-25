import numpy as np
import paddle

# 设置随机种子以保证可复现性
paddle.seed(42)
np.random.seed(42)
paddle.framework.core.set_prim_eager_enabled(True)


def test_autograd_jacobian():
    """测试反向计算雅可比矩阵"""
    print("\n=== 测试 paddle.autograd.jacobian ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32", stop_gradient=False)
    y = x**2
    jacobian = paddle.autograd.jacobian(y, x, batch_axis=0)
    print(f"输入: {x.numpy()}")
    print(f"输出: {y.numpy()}")
    print(f"雅可比矩阵: {jacobian.numpy()}")


def test_autograd_hessian():
    """测试海森矩阵"""
    print("\n=== 测试 paddle.autograd.hessian ===")
    x = paddle.to_tensor([1.0, 2.0], dtype="float32", stop_gradient=False)
    y = paddle.sum(x**3)
    hessian = paddle.autograd.hessian(y, x)
    print(f"输入: {x.numpy()}")
    print(f"输出: {y.numpy()}")
    print(f"海森矩阵: {hessian.numpy()}")


def test_add():
    """测试逐元素加法"""
    print("\n=== 测试 paddle.add ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = paddle.to_tensor([4.0, 5.0, 6.0])
    result = paddle.add(x, y)
    print(f"x + y = {result.numpy()}")


def test_subtract():
    """测试逐元素减法"""
    print("\n=== 测试 paddle.subtract ===")
    x = paddle.to_tensor([5.0, 6.0, 7.0])
    y = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.subtract(x, y)
    print(f"x - y = {result.numpy()}")


def test_multiply():
    """测试逐元素乘法"""
    print("\n=== 测试 paddle.multiply ===")
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    y = paddle.to_tensor([5.0, 6.0, 7.0])
    result = paddle.multiply(x, y)
    print(f"x * y = {result.numpy()}")


def test_divide():
    """测试逐元素除法"""
    print("\n=== 测试 paddle.divide ===")
    x = paddle.to_tensor([10.0, 15.0, 20.0])
    y = paddle.to_tensor([2.0, 3.0, 4.0])
    result = paddle.divide(x, y)
    print(f"x / y = {result.numpy()}")


def test_tanh():
    """测试逐元素双曲正切函数"""
    print("\n=== 测试 paddle.tanh ===")
    x = paddle.to_tensor([-1.0, 0.0, 1.0, 2.0])
    result = paddle.tanh(x)
    print(f"tanh(x) = {result.numpy()}")


def test_sin():
    """测试逐元素正弦函数"""
    print("\n=== 测试 paddle.sin ===")
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = paddle.sin(x)
    print(f"sin(x) = {result.numpy()}")


def test_cos():
    """测试逐元素余弦函数"""
    print("\n=== 测试 paddle.cos ===")
    x = paddle.to_tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    result = paddle.cos(x)
    print(f"cos(x) = {result.numpy()}")


def test_sigmoid():
    """测试逐元素Sigmoid函数"""
    print("\n=== 测试 paddle.nn.functional.sigmoid ===")
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = paddle.nn.functional.sigmoid(x)
    print(f"sigmoid(x) = {result.numpy()}")


def test_matmul():
    """测试矩阵乘法"""
    print("\n=== 测试 paddle.matmul ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
    result = paddle.matmul(x, y)
    print(f"矩阵乘法结果:\n{result.numpy()}")


def test_pow():
    """测试矩阵幂运算"""
    print("\n=== 测试 paddle.pow ===")
    x = paddle.to_tensor([2.0, 3.0, 4.0])
    result = paddle.pow(x, 3)
    print(f"x^3 = {result.numpy()}")


def test_assign():
    """测试张量拷贝"""
    print("\n=== 测试 paddle.assign ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    y = paddle.assign(x)
    print(f"原始张量: {x.numpy()}")
    print(f"拷贝张量: {y.numpy()}")


def test_concat():
    """测试张量拼接"""
    print("\n=== 测试 paddle.concat ===")
    x1 = paddle.to_tensor([[1.0, 2.0]])
    x2 = paddle.to_tensor([[3.0, 4.0]])
    result = paddle.concat([x1, x2], axis=0)
    print(f"拼接结果:\n{result.numpy()}")


def test_expand():
    """测试张量扩展"""
    print("\n=== 测试 paddle.expand ===")
    x = paddle.to_tensor([[1.0, 2.0]])
    result = paddle.expand(x, shape=[3, 2])
    print(f"扩展结果:\n{result.numpy()}")


def test_squeeze():
    """测试张量删除维度"""
    print("\n=== 测试 paddle.squeeze ===")
    x = paddle.to_tensor([[[1.0, 2.0, 3.0]]])
    result = paddle.squeeze(x)
    print(f"原始形状: {x.shape}, 压缩后形状: {result.shape}")
    print(f"压缩结果: {result.numpy()}")


def test_unsqueeze():
    """测试张量增加维度"""
    print("\n=== 测试 paddle.unsqueeze ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.unsqueeze(x, axis=0)
    print(f"原始形状: {x.shape}, 增维后形状: {result.shape}")
    print(f"增维结果: {result.numpy()}")


def test_scale():
    """测试张量缩放和偏置"""
    print("\n=== 测试 paddle.scale ===")
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    result = paddle.scale(x, scale=2.0, bias=1.0)
    print(f"x * 2 + 1 = {result.numpy()}")


def test_transpose():
    """测试张量重排"""
    print("\n=== 测试 paddle.transpose ===")
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = paddle.transpose(x, perm=[1, 0])
    print(f"原始形状: {x.shape}, 转置后形状: {result.shape}")
    print(f"转置结果:\n{result.numpy()}")


def test_sign():
    """测试张量逐元素正负判断"""
    print("\n=== 测试 paddle.sign ===")
    x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = paddle.sign(x)
    print(f"sign(x) = {result.numpy()}")


def test_cast():
    """测试张量类型转换"""
    print("\n=== 测试 paddle.cast ===")
    x = paddle.to_tensor([1.5, 2.7, 3.2])
    result = paddle.cast(x, dtype="int32")
    print(f"float32 -> int32: {result.numpy()}")


def test_slice():
    """测试张量切片"""
    print("\n=== 测试 paddle.slice ===")
    x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = paddle.slice(x, axes=[0, 1], starts=[0, 1], ends=[2, 3])
    print(f"切片结果:\n{result.numpy()}")


def test_sum():
    """测试张量求和"""
    print("\n=== 测试 paddle.sum ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.sum(x)
    print(f"总和: {result.numpy()}")


def test_mean():
    """测试张量均值"""
    print("\n=== 测试 paddle.mean ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.mean(x)
    print(f"均值: {result.numpy()}")


def test_is_complex():
    """测试判断是否是复数"""
    print("\n=== 测试 paddle.is_complex ===")
    x_real = paddle.to_tensor([1.0, 2.0])
    x_complex = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    print(f"实数张量是复数: {paddle.is_complex(x_real)}")
    print(f"复数张量是复数: {paddle.is_complex(x_complex)}")


def test_angle():
    """测试计算相位角"""
    print("\n=== 测试 paddle.angle ===")
    x = paddle.to_tensor([1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j])
    result = paddle.angle(x)
    print(f"相位角: {result.numpy()}")


def test_polar():
    """测试由极坐标表示计算复平面坐标"""
    print("\n=== 测试 paddle.polar ===")
    abs_val = paddle.to_tensor([1.0, 2.0])
    angle_val = paddle.to_tensor([np.pi / 4, np.pi / 2])
    result = paddle.polar(abs_val, angle_val)
    print(f"极坐标转复数: {result.numpy()}")


def test_mod_complex():
    """测试复数取模"""
    print("\n=== 测试 paddle.mod (复数) ===")
    x = paddle.to_tensor([3.0 + 4.0j, 5.0 + 12.0j], dtype="complex64")
    y = paddle.to_tensor([2.0 + 1.0j, 3.0 + 2.0j], dtype="complex64")
    result = paddle.mod(x, y)
    print(f"复数取模: {result.numpy()}")


def test_as_real():
    """测试复数转换为实数"""
    print("\n=== 测试 paddle.as_real ===")
    x = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
    result = paddle.as_real(x)
    print(f"复数转实数:\n{result.numpy()}")


def test_as_complex():
    """测试实数转化为复数"""
    print("\n=== 测试 paddle.as_complex ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
    result = paddle.as_complex(x)
    print(f"实数转复数: {result.numpy()}")


def test_einsum_complex():
    """测试复数爱因斯坦求和"""
    print("\n=== 测试 paddle.einsum (复数) ===")
    x = paddle.to_tensor([1.0 + 1.0j, 2.0 + 2.0j])
    y = paddle.to_tensor([3.0 + 1.0j, 4.0 + 2.0j])
    result = paddle.einsum("i,i->", x, y)
    print(f"复数爱因斯坦求和: {result.numpy()}")


def test_uniform_complex():
    """测试以均匀分布随机数创建稠密复数张量"""
    print("\n=== 测试 paddle.uniform (复数) ===")
    result = paddle.uniform([2, 3], dtype="complex64", min=-1.0, max=1.0)
    print(f"均匀分布复数张量:\n{result.numpy()}")


def test_full_complex():
    """测试创建元素值全为指定复数的稠密复数张量"""
    print("\n=== 测试 paddle.full (复数) ===")
    result = paddle.full([2, 3], fill_value=1.0 + 2.0j, dtype="complex64")
    print(f"全为指定复数的张量:\n{result.numpy()}")


def test_zeros_complex():
    """测试创建元素值全为0+0i的稠密复数张量"""
    print("\n=== 测试 paddle.zeros (复数) ===")
    result = paddle.zeros([2, 3], dtype="complex64")
    print(f"全零复数张量:\n{result.numpy()}")


def test_slice_complex():
    """测试复数张量切片"""
    print("\n=== 测试 paddle.slice (复数) ===")
    x = paddle.to_tensor(
        [[1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], [4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j]],
        dtype="complex64",
    )
    result = paddle.slice(x, axes=[0, 1], starts=[0, 0], ends=[1, 2])
    print(f"复数张量切片:\n{result.numpy()}")


def test_rfft2():
    """测试二维离散傅里叶变换"""
    print("\n=== 测试 paddle.fft.rfft2 ===")
    import paddle

    x = paddle.randn(4, 16, 16, 3)
    result = paddle.fft.rfft2(x, axes=(1, 2), norm="ortho")
    print(f"二维FFT结果:\n{result.numpy()}")


def test_irfft2():
    """测试二维离散逆傅里叶变换"""
    print("\n=== 测试 paddle.fft.irfft2 ===")
    import paddle

    B, H, W, C = 4, 16, 16, 3
    x = paddle.randn(B, H, W // 2 + 1, C)
    result = paddle.fft.irfft2(x, s=(H, W), axes=(1, 2), norm="ortho")
    print(f"二维逆FFT结果:\n{result.numpy()}")


def test_eig():
    """测试计算一般方阵的特征值与特征向量"""
    print("\n=== 测试 paddle.linalg.eig ===")
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    eigenvalues, eigenvectors = paddle.linalg.eig(x)
    print(f"特征值: {eigenvalues.numpy()}")
    print(f"特征向量:\n{eigenvectors.numpy()}")


def test_ppsci_fractional_diff():
    """测试分数阶微分(需要安装PaddleScience)"""
    print("\n=== 测试 ppsci.experimental.fractional_diff ===")
    try:
        import ppsci

        def f(x):
            return x * x

        result = ppsci.experimental.fractional_diff(
            f, alpha=0.5, a=0, t=1.0, h=1e-6, dtype="float32"
        )
        print(f"分数阶微分结果: {result.numpy()}")
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

        def some_function(x):
            return paddle.sin(x[:, 0]) + paddle.exp(x[:, 1])

        result = ppsci.experimental.montecarlo_integrate(
            some_function,
            dim=2,
            N=10000,
            integration_domain=[[0, 1], [-1, 1]],
        )
        print(f"蒙特卡洛积分结果: {result}")
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

        func = lambda x: paddle.sin(x)
        dim = 1
        N = 500
        integration_domains = [[0, np.pi]]
        result = ppsci.experimental.gaussian_integrate(
            func, dim, N, integration_domains, dtype="float32"
        )
        print(f"高斯积分结果: {result}")
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
        test_eig,
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
