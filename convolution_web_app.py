import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. 健壮的函数解析器和离散化 (与原代码逻辑完全一致) ---

def evaluate_function(func_str, t):
    """
    健壮地解析并计算用户输入的函数表达式 f(t). 
    t 是时间轴数组。
    """
    
    # 辅助函数，在 eval 环境中定义
    def u(x):
        """阶跃函数 u(x)"""
        return (x >= 0).astype(float)
    
    def rect(x, width=1):
        """矩形脉冲函数 rect(x/width)"""
        return (np.abs(x / width) <= 0.5).astype(float)

    # 定义安全的执行环境
    env = {
        'np': np,
        't': t,
        'pi': np.pi,
        'exp': np.exp,
        'cos': np.cos,
        'sin': np.sin,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'u': u,
        'rect': rect 
    }

    try:
        y_raw = eval(func_str, {"__builtins__": None}, env)
        
        if isinstance(y_raw, bool) or isinstance(y_raw, (int, float)):
            y = np.full_like(t, float(y_raw), dtype=float)
        elif isinstance(y_raw, np.ndarray):
            y = y_raw.astype(float)
        else:
            raise TypeError(f"表达式返回了不支持的类型: {type(y_raw)}")

        y[np.isnan(y)] = 0.0
        y[np.isinf(y)] = 0.0
        
        return y
    except Exception as e:
        # 在 Streamlit 中使用 st.error 代替 messagebox
        st.error(f"解析表达式时出错: {e}")
        st.info("请检查表达式格式，例如：`u(t) * exp(-t)` 或 `rect(t, 2)`")
        return np.zeros_like(t)

# --- 2. Streamlit 主应用函数 ---

def main_convolution_app():
    st.set_page_config(layout="wide")
    st.title("连续信号卷积运算智能体")
    
    # --- A. Streamlit 输入控制区 (替代 Tkinter/ttk) ---
    
    st.sidebar.header("输入控制")
    
    f1_str = st.sidebar.text_input("f1(t) =", value="u(t) * exp(-t)")
    f2_str = st.sidebar.text_input("f2(t) =", value="rect(t, 2)")
    
    col1, col2 = st.sidebar.columns(2)
    # 使用 Streamlit 的输入组件，并设置默认值
    t_start = col1.number_input("T_start:", value=-5.0, step=1.0)
    t_end = col2.number_input("T_end:", value=10.0, step=1.0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**可用函数**:")
    st.sidebar.code("u(t), rect(t, width), exp(), cos(), sin()")
    
    # --- B. 数据计算和验证 ---
    
    dt = 0.01 
    
    if t_start >= t_end:
        st.error("起始时间必须小于结束时间。")
        return

    # 计算时间轴
    t = np.arange(t_start, t_end, dt)
    
    # 离散化信号
    f1 = evaluate_function(f1_str, t)
    f2 = evaluate_function(f2_str, t)

    # 卷积计算
    conv_result = signal.convolve(f1, f2, mode='full') * dt
    conv_t_len = len(t) + len(t) - 1
    # 确定卷积后的时间轴 (与原代码计算逻辑一致)
    conv_t = np.arange(conv_t_len) * dt + 2 * t_start
    
    if np.sum(np.abs(f1)) == 0 or np.sum(np.abs(f2)) == 0:
        st.warning("信号f1(t)或f2(t)在当前时间范围内为零。")
        return
        
    conv_t_start = conv_t[0]
    conv_t_end = conv_t[-1]

    # --- C. Streamlit 滑块控制 (替代 ttk.Scale) ---

    st.subheader("卷积过程控制")
    # Streamlit 滑块直接返回 t_shift 的值，简化了 Tkinter 中的回调和状态管理
    t_shift = st.slider(
        "时间平移 $t$: (当前值 $t = %.2f$)" % conv_t_start,
        min_value=conv_t_start,
        max_value=conv_t_end,
        value=conv_t_start,
        step=dt,
        format='t = %.2f'
    )
    
    # --- D. Matplotlib 绘图 (保留核心绘图逻辑和比例) ---
    
    # 1. 设置 Matplotlib 子图，注意 height_ratios 为 [1, 1, 2, 2]
    fig, (ax0_1, ax0_2, ax1, ax2) = plt.subplots(
        4, 1, 
        figsize=(8, 12), 
        sharex=True,
        gridspec_kw={
            'hspace': 0.6,
            'height_ratios': [1, 1, 2, 2] 
        }
    )
    
    # 2. 统一绘图辅助变量
    max_y_orig = np.max([np.max(f1), np.max(f2), 1.0]) * 1.2
    min_y_orig = np.min([np.min(f1), np.min(f2), 0.0]) * 1.2
    x_lim_orig = (t_start, t_end)
    max_y_conv = np.max([np.max(conv_result), 1.0]) * 1.2
    min_y_conv = np.min([np.min(conv_result), 0.0]) * 1.2
    
    
    # --- A'. 原始信号 f1(t) ---
    ax0_1.plot(t, f1, label='$f_1(t)$', color='red')
    ax0_1.set_title('$f_1(t)$ 原始信号', fontsize=10)
    ax0_1.set_ylabel('幅度', fontsize=8)
    ax0_1.grid(True, linestyle=':')
    ax0_1.legend(loc='upper right')
    ax0_1.set_ylim(min_y_orig, max_y_orig)
    ax0_1.set_xlim(x_lim_orig)
    ax0_1.set_xlabel('t', fontsize=8) 
    

    # --- B'. 原始信号 f2(t) ---
    ax0_2.plot(t, f2, label='$f_2(t)$', color='green')
    ax0_2.set_title('$f_2(t)$ 原始信号', fontsize=10)
    ax0_2.set_ylabel('幅度', fontsize=8)
    ax0_2.grid(True, linestyle=':')
    ax0_2.legend(loc='upper right')
    ax0_2.set_ylim(min_y_orig, max_y_orig)
    ax0_2.set_xlim(x_lim_orig)
    ax0_2.set_xlabel('t', fontsize=8)

    
    # --- C'. 卷积过程图 (ax1) - 动态部分 ---
    
    # 核心：计算平移和反转后的信号 f2(t-tau)
    t_for_f2 = t_shift - t 
    f2_shifted = evaluate_function(f2_str, t_for_f2)

    ax1.plot(t, f1, label='$f_1(\\tau)$', color='red')
    ax1.plot(t, f2_shifted, label='$f_2(t-\\tau)$', color='green', linestyle='--')
    
    # 乘积并填充区域
    product = f1 * f2_shifted
    ax1.fill_between(t, 0, product, color='orange', alpha=0.3, label='$f_1(\\tau)f_2(t-\\tau)$ 乘积')

    ax1.set_title(f'卷积过程: $f_1(\\tau)$ 和 $f_2({t_shift:.2f}-\\tau)$', fontsize=10)
    ax1.set_xlabel('$\\tau$')
    ax1.set_ylabel('幅度')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--')
    ax1.set_ylim(min_y_orig, max_y_orig)
    ax1.set_xlim(x_lim_orig)


    # --- D'. 最终卷积结果图 (ax2) - 逐步绘制 ---
    
    ax2.set_title(f'最终卷积结果 $f_1(t) * f_2(t)$', fontsize=10)
    ax2.set_xlabel('t')
    ax2.set_ylabel('幅度')
    ax2.grid(True, linestyle='--')
    ax2.set_ylim(min_y_conv, max_y_conv)
    ax2.set_xlim(conv_t[0], conv_t[-1])

    # 1. 找到当前时间 t_shift 对应的索引
    idx_max = np.searchsorted(conv_t, t_shift, side='right')
    
    if idx_max > 0:
        # 2. 提取并绘制从开始到当前索引的所有数据
        t_plot = conv_t[:idx_max]
        y_plot = conv_result[:idx_max]
        
        ax2.plot(t_plot, y_plot, label='$f_1(t) * f_2(t)$', color='blue')
        
        # 3. 绘制红点 (当前积分结果)
        conv_value = conv_result[idx_max - 1]
        current_t = conv_t[idx_max - 1]
    else:
        # 如果滑块在起始点之前，曲线为空
        conv_value = 0.0
        current_t = conv_t_start

    ax2.plot([current_t], [conv_value], 'ro', markersize=8, label='当前积分结果')
    ax2.legend(loc='upper right')
    
    # 4. 在 Streamlit 页面上显示图表
    st.pyplot(fig)
    plt.close(fig) # 避免内存泄漏

if __name__ == "__main__":
    main_convolution_app()