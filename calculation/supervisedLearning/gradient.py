import numpy as np

def numerical_diff(f, x):
    """
        f: 被微分関数

        使用例:
        def function_1(x):
            return 0.01 * x ** 2 + 0.1 * x
        
        で表される関数のx=5での微分係数を計算するときは

        numerical_diff(function_1, 5)
        とプログラムする。
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)     # xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x + h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    
    return grad
