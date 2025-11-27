import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import streamlit as st
import time

# --- 1. 缓存计算函数 (不变) ---

# 启动时的默认平移值
INITIAL_SHIFT_T = -6.0 

@st.cache_data
def calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt=0.01):
    t = np.arange(t_start, t_end, dt)
    def u(x):
        return (x >= 0).astype(float)
    def rect(x, width=1):
        return (np.abs(x / width) <= 0.5).astype(float)
        
    f1 = evaluate_function(f1_str, t)
    f2 = evaluate_function(f2_str, t)
    
    conv_result = signal.convolve(f1, f2, mode='full') * dt
    conv_t_len = len(t) + len(t) - 1
    conv_t = np.arange(conv_t_len) * dt + 2 * t_start
    
    max_y_orig = np.max([np.max(f1), np.max(f2), 1.0]) * 1.2
    min_y_orig = np.min([np.min(f1), np.min(f2), 0.0]) * 1.2
    max_y_conv = np.max([np.max(conv_result), 1.0]) * 1.2
    min_y_conv = np.min([np.min(conv_result), 0.0]) * 1.2
    
    return t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv

# --- 2. 健壮的函数解析器和离散化 (不变) ---
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
            return np.zeros_like(t) 

        y[np.isnan(y)] = 0.0
        y[np.isinf(y)] = 0.0
        return y
    except Exception as e:
        return np.zeros_like(t)


# --- 3. 状态管理函数 (不变) ---
def initialize_state(conv_t_start, conv_t_end, initial_t):
    if 'current_t' not in st.session_state or st.session_state.reset_flag:
        st.session_state.current_t = initial_t 
        st.session_state.conv_t_start = conv_t_start
        st.session_state.conv_t_end = conv_t_end
        st.session_state.reset_flag = False
        if 'is_running' not in st.session_state:
             st.session_state.is_running = False
        if st.session_state.is_running:
             st.session_state.is_running = False

def step_forward(dt_step):
    if st.session_state.current_t < st.session_state.conv_t_end:
        st.session_state.current_t = min(
            st.session_state.current_t + dt_step, 
            st.session_state.conv_t_end
        )
        return True 
    return False

def step_backward(dt_step):
    if st.session_state.current_t > st.session_state.conv_t_start:
        st.session_state.current_t = max(
            st.session_state.current_t - dt_step, 
            st.session_state.conv_t_start
        )
        return True
    return False

# --- 4. Streamlit 主应用函数 ---

def main_convolution_app():
    st.set_page_config(layout="wide") 
    
    st.markdown("### 连续信号卷积运算智能体", unsafe_allow_html=True)
    
    dt = 0.01 
    # *** 优化 1: 增大步长，让每帧的变化更明显 ***
    STEP_SIZE = 0.4
    # *** 优化 2: 增大延迟，让每帧之间更稳定 ***
    ANIMATION_DELAY = 0.05 

    # --- A. 输入控制区 (侧边栏) ---
    st.sidebar.header("输入控制")
    f1_str = st.sidebar.text_input("f1(t) =", value="u(t) * exp(-t)")
    f2_str = st.sidebar.text_input("f2(t) =", value="rect(t, 2)")
    col1, col2 = st.sidebar.columns(2)
    t_start = col1.number_input("T_start:", value=-6.0, step=1.0) 
    t_end = col2.number_input("T_end:", value=10.0, step=1.0)
    
    if t_start >= t_end:
        st.sidebar.error("起始时间必须小于结束时间。")
        return

    # --- B. 数据计算 (缓存调用) ---
    t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv = \
        calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt)
    
    conv_t_start = conv_t[0]
    conv_t_end = conv_t[-1]

    # 初始化/重置状态
    if st.sidebar.button("运行/更新卷积"):
        calculate_convolution_data.clear() 
        st.session_state.reset_flag = True
    
    initialize_state(conv_t_start, conv_t_end, INITIAL_SHIFT_T)

    # --- C. 动画控制区 (极致压缩) ---
    
    control_container = st.container()
    
    time_display = control_container.empty()
    time_display.markdown(f"**当前平移时间 $t = {st.session_state.current_t:.2f}$**")

    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = control_container.columns([1.5, 1.5, 1.5, 1.5, 4])
    
    if col_btn1.button("◀️ 后退一步"):
        st.session_state.is_running = False
        step_backward(STEP_SIZE)
        
    if st.session_state.is_running:
        button_label = "⏸️ 暂停"
    else:
        button_label = "▶️ 播放"

    if col_btn2.button(button_label):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun() 
        
    if col_btn3.button("▶️ 前进一步"):
        st.session_state.is_running = False
        step_forward(STEP_SIZE)

    if col_btn4.button("⏪ 重置"):
        st.session_state.is_running = False
        st.session_state.current_t = INITIAL_SHIFT_T
        

    # --- D. Matplotlib 绘图 (压缩) ---
    
    t_shift = st.session_state.current_t
    x_lim_orig = (t_start, t_end)
    
    fig, (ax0_1, ax0_2, ax1, ax2) = plt.subplots(
        4, 1, 
        figsize=(8, 6.5), 
        sharex=True,
        gridspec_kw={'hspace': 0.3, 'height_ratios': [1, 1, 2, 2] }
    )
    plt.subplots_adjust(top=0.98, bottom=0.04) 

    TINY_FONT = 8
    
    # 1. f1(t) 原始信号
    ax0_1.plot(t, f1, label='$f_1(t)$', color='red')
    ax0_1.set_title('$f_1(t)$ 原始信号', fontsize=TINY_FONT)
    ax0_1.set_ylabel('幅度', fontsize=TINY_FONT - 1)
    ax0_1.tick_params(axis='y', labelsize=TINY_FONT - 1)
    ax0_1.set_ylim(min_y_orig, max_y_orig)
    ax0_1.set_xlim(x_lim_orig)
    ax0_1.grid(True, linestyle=':')
    ax0_1.legend(loc='upper right', fontsize=TINY_FONT - 2)
    
    # 2. f2(t) 原始信号
    ax0_2.plot(t, f2, label='$f_2(t)$', color='green')
    ax0_2.set_title('$f_2(t)$ 原始信号', fontsize=TINY_FONT)
    ax0_2.set_ylabel('幅度', fontsize=TINY_FONT - 1)
    ax0_2.tick_params(axis='y', labelsize=TINY_FONT - 1)
    ax0_2.set_ylim(min_y_orig, max_y_orig)
    ax0_2.set_xlim(x_lim_orig)
    ax0_2.grid(True, linestyle=':')
    ax0_2.legend(loc='upper right', fontsize=TINY_FONT - 2)
    
    # 3. 卷积过程图 (动态更新)
    t_for_f2 = t_shift - t 
    f2_shifted = evaluate_function(f2_str, t_for_f2)
    product = f1 * f2_shifted
    
    ax1.plot(t, f1, label='$f_1(\\tau)$', color='red')
    ax1.plot(t, f2_shifted, label='$f_2(t-\\tau)$', color='green', linestyle='--')
    ax1.fill_between(t, 0, product, color='orange', alpha=0.3, label='$f_1(\\tau)f_2(t-\\tau)$ 乘积')
    ax1.set_title(f'卷积过程: $f_1(\\tau)$ 和 $f_2({t_shift:.2f}-\\tau)$', fontsize=TINY_FONT)
    ax1.set_xlabel('$\\tau$', fontsize=TINY_FONT - 1)
    ax1.set_ylabel('幅度', fontsize=TINY_FONT - 1)
    ax1.tick_params(axis='y', labelsize=TINY_FONT - 1)
    ax1.set_ylim(min_y_orig, max_y_orig)
    ax1.set_xlim(x_lim_orig)
    ax1.legend(loc='upper right', fontsize=TINY_FONT - 2)
    ax1.grid(True, linestyle='--')

    # 4. 最终卷积结果图 (状态累积绘制)
    idx_max = np.searchsorted(conv_t, t_shift, side='right')
    
    if idx_max > 0:
        t_plot = conv_t[:idx_max]
        y_plot = conv_result[:idx_max]
        ax2.plot(t_plot, y_plot, label='$f_1(t) * f_2(t)$', color='blue')
        
        conv_value = np.interp(t_shift, conv_t, conv_result) 
        current_t = t_shift 

    else:
        conv_value = 0.0
        current_t = st.session_state.conv_t_start

    ax2.plot([current_t], [conv_value], 'ro', markersize=8, label='当前积分结果')
    ax2.set_title(f'最终卷积结果 $f_1(t) * f_2(t)$', fontsize=TINY_FONT)
    ax2.set_xlabel('t', fontsize=TINY_FONT - 1)
    ax2.set_ylabel('幅度', fontsize=TINY_FONT - 1)
    ax2.tick_params(axis='both', labelsize=TINY_FONT - 1)
    ax2.set_ylim(min_y_conv, max_y_conv)
    ax2.set_xlim(conv_t[0], conv_t[-1])
    ax2.legend(loc='upper right', fontsize=TINY_FONT - 2)
    ax2.grid(True, linestyle='--')

    st.pyplot(fig)
    plt.close(fig)
    
    # --- E. 动画循环逻辑 ---
    if st.session_state.is_running:
        moved = step_forward(STEP_SIZE)
        
        if moved:
            time.sleep(ANIMATION_DELAY)
            st.rerun() 
        else:
            st.session_state.is_running = False
            st.toast("动画播放完毕！")

if __name__ == "__main__":
    main_convolution_app()