import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


# 設置字體以支持中文顯示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字體
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號


def plot_parallel_rlc_response(R, L, C, v0, iL0, t_max=0.01, num_points=1000):
    """
    繪製並聯RLC電路的電壓響應v(t)

    參數:
    R: 電阻 (Ω)
    L: 電感 (H)
    C: 電容 (F)
    v0: 電容初始電壓 (V)
    iL0: 電感初始電流 (A)
    t_max: 最大時間 (s)
    num_points: 時間點數量
    """
    # 計算特徵參數
    w0 = 1 / sqrt(L * C)  # 自然頻率
    alpha = 1 / (2 * R * C)  # neper頻率

    print(f"自然頻率 ω₀ = {w0:.2f} rad/s")
    print(f"Neper頻率 α = {alpha:.2f} rad/s")
    print(f"α² = {alpha ** 2:.2f}, ω₀² = {w0 ** 2:.2f}")

    # 時間數組
    t = np.linspace(0, t_max, num_points)

    if alpha ** 2 > w0 ** 2:  # 過阻尼
        print("系統狀態: 過阻尼")
        s1 = -alpha + sqrt(alpha ** 2 - w0 ** 2)
        s2 = -alpha - sqrt(alpha ** 2 - w0 ** 2)

        # 解方程組求A1, A2
        # v(0⁺) = A1 + A2
        # dv/dt(0⁺) = -[iL(0⁺) + v(0⁺)/R]/C = s1*A1 + s2*A2
        A = np.array([[1, 1], [s1, s2]])
        dv0 = -(iL0 + v0 / R) / C  # dv/dt(0⁺)的初始條件
        b = np.array([v0, dv0])
        A1, A2 = np.linalg.solve(A, b)

        v_t = A1 * np.exp(s1 * t) + A2 * np.exp(s2 * t)

        print(f"根: s1 = {s1:.2f}, s2 = {s2:.2f}")
        print(f"係數: A1 = {A1:.4f}, A2 = {A2:.4f}")
        print(f"dv/dt(0⁺) = {dv0:.4f}")

    elif alpha ** 2 < w0 ** 2:  # 欠阻尼
        print("系統狀態: 欠阻尼")
        w_d = sqrt(w0 ** 2 - alpha ** 2)  # 阻尼自然頻率

        # 解方程組求B1, B2
        # v(0⁺) = B1
        # dv/dt(0⁺) = -[iL(0⁺) + v(0⁺)/R]/C = -αB1 + ω_dB2
        B1 = v0
        dv0 = -(iL0 + v0 / R) / C  # dv/dt(0⁺)的初始條件
        B2 = (dv0 + alpha * B1) / w_d

        v_t = np.exp(-alpha * t) * (B1 * np.cos(w_d * t) + B2 * np.sin(w_d * t))

        print(f"阻尼自然頻率 ω_d = {w_d:.2f} rad/s")
        print(f"係數: B1 = {B1:.4f}, B2 = {B2:.4f}")
        print(f"dv/dt(0⁺) = {dv0:.4f}")

    else:  # 臨界阻尼
        print("系統狀態: 臨界阻尼")
        s = -alpha  # s1 = s2 = -alpha

        # 解方程組求D1, D2
        # v(0⁺) = D2
        # dv/dt(0⁺) = -[iL(0⁺) + v(0⁺)/R]/C = D1 - αD2
        D2 = v0
        dv0 = -(iL0 + v0 / R) / C  # dv/dt(0⁺)的初始條件
        D1 = dv0 + alpha * D2

        v_t = np.exp(-alpha * t) * (D1 * t + D2)

        print(f"重根: s = {s:.2f}")
        print(f"係數: D1 = {D1:.4f}, D2 = {D2:.4f}")
        print(f"dv/dt(0⁺) = {dv0:.4f}")

    return t, v_t


def main():
    # 設置電路參數
    R = 1000  # 電阻 1kΩ
    L = 0.1  # 電感 0.1H
    C = 1e-6  # 電容 1μF

    # 初始條件
    v0 = 5.0  # 電容初始電壓 5V
    iL0 = 0.01  # 電感初始電流 10mA

    # 繪製響應
    t, v_t = plot_parallel_rlc_response(R, L, C, v0, iL0, t_max=0.005)

    # 創建圖形
    plt.figure(figsize=(12, 6))
    plt.plot(t * 1000, v_t, 'b-', linewidth=2, label='v(t)')
    plt.xlabel('時間 (ms)', fontsize=12)
    plt.ylabel('電壓 (V)', fontsize=12)
    plt.title('並聯RLC電路電壓響應', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 添加零電壓參考線
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


# 測試不同阻尼情況的例子
def test_different_cases():
    """測試不同阻尼情況的示例"""
    cases = [
        {"R": 1000, "L": 8, "C": 0.125e-6, "v0": 0, "iL0": -0.01225, "label": "過阻尼"},
        {"R": 20000, "L": 8, "C": 0.125e-6, "v0": 0, "iL0": -0.01225, "label": "欠阻尼"},
        {"R": 4000, "L": 8, "C": 0.125e-6, "v0": 0, "iL0": -0.01225, "label": "臨界阻尼"}  # R ≈ 1/(2w0C)
    ]

    plt.figure(figsize=(12, 8))

    for i, case in enumerate(cases):
        t, v_t = plot_parallel_rlc_response(case["R"], case["L"], case["C"],
                                            case["v0"], case["iL0"], t_max=0.01)

        plt.subplot(2, 2, i + 1)
        plt.plot(t * 1000, v_t, linewidth=2, label=case["label"])
        plt.xlabel('時間 (ms)')
        plt.ylabel('電壓 (V)')
        plt.title(f'{case["label"]}響應 (R={case["R"]}Ω)')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 運行主例子
    # main()

    # 運行測試不同阻尼情況的例子
    test_different_cases()