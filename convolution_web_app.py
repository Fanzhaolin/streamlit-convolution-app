import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. 健壮的函数解析器和离散化 (保持不变) ---

def evaluate_function(func_str, t):
    def u(x):
        return (x >= 0).astype(float)
    
    def rect(x, width=1):
        return (np.abs(x / width) <= 0.5).astype(float)

    env = {
        'np': np, 't': t, 'pi': np.pi, 'exp': np.exp, 'cos': np.cos,
        'sin': np.sin, 'abs': np.abs, 'sqrt': np.sqrt, 'u': u, 'rect': rect 
    }

    try:
        y_raw = eval(func_str, {"__builtins__": None}, env)
        if isinstance(y_raw, bool) or isinstance(y_raw, (int, float)):
            y = np.full_like(t, float(y_raw), dtype=float)
        elif isinstance(y_raw, np.ndarray):
            y = y_raw.astype(float)
        else:
            st.error(f"表达式返回了不支持的类型: {type(y_raw)}")
            return np.zeros_like(t)

        y[np.isnan(y)] = 0.0
        y[np.isinf(y)] = 0.0
        return y
    except Exception as e:
        st.error(f"解析表达式时出错: {e}")
        st.info("请检查表达式格式，例如：`u(t) * exp(-t)`")
        return np.zeros_like(t)

# --- 2. Streamlit 主应用函数 ---

def main_convolution_app():
    # 使用 wide 布局来充分利用水平空间
    st.set_page_config(layout="wide") 
    st.title("连续信号卷积运算智能体")
    
    # --- A. 输入控制区 ---
    st.sidebar.header("输入控制")
    f1_str = st.sidebar.text_input("f1(t) =", value="u(t) * exp(-t)")
    f2_str = st.sidebar.text_input("f2(t) =", value="rect(t, 2)")
    col1, col2 = st.sidebar.columns(2)
    t_start = col1.number_input("T_start:", value=-5.0, step=1.0)
    t_end = col2.number_input("T_end:", value=10.0, step=1.0)
    
    if t_start >= t_end:
        st.error("起始时间必须小于结束时间。")
        return

    # --- B. 数据计算 ---
    dt = 0.01 
    t = np.arange(t_start, t_end, dt)
    f1 = evaluate_function(f1_str, t)
    f2 = evaluate_function(f2_str, t)
    conv_result = signal.convolve(f1, f2, mode='full') * dt
    conv_t_len = len(t) + len(t) - 1
    conv_t = np.arange(conv_t_len) * dt + 2 * t_start
    conv_t_start = conv_t[0]
    conv_t_end = conv_t[-1]

    # --- C. 滑块控制 ---
    st.subheader("卷积过程控制")
    # 确保滑块在 Streamlit 应用首次运行时能拿到 conv_t_start 的值
    t_shift = st.slider(
        "时间平移 $t$: (当前值 $t = %.2f$)" % conv_t_start,
        min_value=conv_t_start,
        max_value=conv_t_end,
        value=conv_t_start,
        step=dt,
        format='t = %.2f'
    )
    
    # --- D. Matplotlib 绘图 (缩小 figsize) ---
    
    # *** 关键修改: 减小 figsize 高度到 (8, 8) ***
    fig, (ax0_1, ax0_2, ax1, ax2) = plt.subplots(
        4, 1, 
        figsize=(8, 8), 
        sharex=True,
        gridspec_kw={
            'hspace': 0.4, # 减小垂直间距
            'height_ratios': [1, 1, 2, 2] 
        }
    )
    
    plt.subplots_adjust(top=0.95, bottom=0.05) # 调整整体边界以节省空间

    max_y_orig = np.max([np.max(f1), np.max(f2), 1.0]) * 1.2
    min_y_orig = np.min([np.min(f1), np.min(f2), 0.0]) * 1.2
    x_lim_orig = (t_start, t_end)
    max_y_conv = np.max([np.max(conv_result), 1.0]) * 1.2
    min_y_conv = np.min([np.min(conv_result), 0.0]) * 1.2
    
    # 1. f1(t) 原始信号
    ax0_1.plot(t, f1, label='$f_1(t)$', color='red')
    ax0_1.set_title('$f_1(t)$ 原始信号', fontsize=10)
    ax0_1.set_ylabel('幅度', fontsize=8)
    ax0_1.set_ylim(min_y_orig, max_y_orig)
    ax0_1.set_xlim(x_lim_orig)
    ax0_1.grid(True, linestyle=':')
    ax0_1.legend(loc='upper right', fontsize=8)
    
    # 2. f2(t) 原始信号
    ax0_2.plot(t, f2, label='$f_2(t)$', color='green')
    ax0_2.set_title('$f_2(t)$ 原始信号', fontsize=10)
    ax0_2.set_ylabel('幅度', fontsize=8)
    ax0_2.set_ylim(min_y_orig, max_y_orig)
    ax0_2.set_xlim(x_lim_orig)
    ax0_2.grid(True, linestyle=':')
    ax0_2.legend(loc='upper right', fontsize=8)
    
    # 3. 卷积过程图 (动态)
    t_for_f2 = t_shift - t 
    f2_shifted = evaluate_function(f2_str, t_for_f2)
    product = f1 * f2_shifted
    
    ax1.plot(t, f1, label='$f_1(\\tau)$', color='red')
    ax1.plot(t, f2_shifted, label='$f_2(t-\\tau)$', color='green', linestyle='--')
    ax1.fill_between(t, 0, product, color='orange', alpha=0.3, label='$f_1(\\tau)f_2(t-\\tau)$ 乘积')
    ax1.set_title(f'卷积过程: $f_1(\\tau)$ 和 $f_2({t_shift:.2f}-\\tau)$', fontsize=10)
    ax1.set_xlabel('$\\tau$')
    ax1.set_ylabel('幅度', fontsize=8)
    ax1.set_ylim(min_y_orig, max_y_orig)
    ax1.set_xlim(x_lim_orig)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--')

    # 4. 最终卷积结果图 (逐步绘制)
    ax2.set_title(f'最终卷积结果 $f_1(t) * f_2(t)$', fontsize=10)
    ax2.set_xlabel('t')
    ax2.set_ylabel('幅度', fontsize=8)
    ax2.grid(True, linestyle='--')
    ax2.set_ylim(min_y_conv, max_y_conv)
    ax2.set_xlim(conv_t[0], conv_t[-1])

    idx_max = np.searchsorted(conv_t, t_shift, side='right')
    if idx_max > 0:
        t_plot = conv_t[:idx_max]
        y_plot = conv_result[:idx_max]
        ax2.plot(t_plot, y_plot, label='$f_1(t) * f_2(t)$', color='blue')
        
        conv_value = conv_result[idx_max - 1]
        current_t = conv_t[idx_max - 1]
    else:
        conv_value = 0.0
        current_t = conv_t_start

    ax2.plot([current_t], [conv_value], 'ro', markersize=8, label='当前积分结果')
    ax2.legend(loc='upper right', fontsize=8)

    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main_convolution_app()