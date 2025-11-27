import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import streamlit as st
import time # 用于动画延迟

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


# --- 2. 状态管理函数 ---

# 启动时的默认平移值
INITIAL_SHIFT_T = -6.0 

def initialize_state(conv_t_start, conv_t_end, initial_t):
    """初始化或重置 Session State."""
    if 'current_t' not in st.session_state or st.session_state.reset_flag:
        st.session_state.current_t = initial_t 
        st.session_state.conv_t_start = conv_t_start
        st.session_state.conv_t_end = conv_t_end
        st.session_state.reset_flag = False
        # **新增：动画控制状态**
        if 'is_running' not in st.session_state:
             st.session_state.is_running = False
        # 如果是“运行/更新卷积”，则确保停止动画
        if st.session_state.is_running:
             st.session_state.is_running = False

def step_forward(dt_step):
    """向前推进时间 dt_step."""
    if st.session_state.current_t < st.session_state.conv_t_end:
        st.session_state.current_t = min(
            st.session_state.current_t + dt_step, 
            st.session_state.conv_t_end
        )
        return True # 表示成功推进
    return False # 表示到达终点

def step_backward(dt_step):
    """向后退回时间 dt_step."""
    if st.session_state.current_t > st.session_state.conv_t_start:
        st.session_state.current_t = max(
            st.session_state.current_t - dt_step, 
            st.session_state.conv_t_start
        )
        return True
    return False

# --- 3. Streamlit 主应用函数 ---

def main_convolution_app():
    st.set_page_config(layout="wide") 
    st.title("连续信号卷积运算智能体")
    
    dt = 0.01 
    # 动画速度：每次前进/后退的步长
    STEP_SIZE = 0.2
    # 动画延迟：每次重绘的最小间隔（秒）
    ANIMATION_DELAY = 0.05 

    # --- A. 输入控制区 ---
    st.sidebar.header("输入控制")
    f1_str = st.sidebar.text_input("f1(t) =", value="u(t) * exp(-t)")
    f2_str = st.sidebar.text_input("f2(t) =", value="rect(t, 2)")
    col1, col2 = st.sidebar.columns(2)
    t_start = col1.number_input("T_start:", value=-6.0, step=1.0) 
    t_end = col2.number_input("T_end:", value=10.0, step=1.0)
    
    if t_start >= t_end:
        st.error("起始时间必须小于结束时间。")
        return

    # --- B. 数据计算 ---
    t = np.arange(t_start, t_end, dt)
    f1 = evaluate_function(f1_str, t)
    f2 = evaluate_function(f2_str, t)
    conv_result = signal.convolve(f1, f2, mode='full') * dt
    conv_t_len = len(t) + len(t) - 1
    conv_t = np.arange(conv_t_len) * dt + 2 * t_start
    conv_t_start = conv_t[0]
    conv_t_end = conv_t[-1]

    # 初始化/重置状态
    if st.sidebar.button("运行/更新卷积"):
        st.session_state.reset_flag = True
    
    initialize_state(conv_t_start, conv_t_end, INITIAL_SHIFT_T)


    # --- C. 动画控制区 ---
    st.subheader("卷积过程控制")
    
    # 显示当前时间
    time_display = st.empty()
    time_display.markdown(f"**当前平移时间 $t = {st.session_state.current_t:.2f}$**")

    # 分割控制按钮区域
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1.5, 1.5, 1.5, 1.5, 4])
    
    # 按钮 1: 后退
    if col_btn1.button("◀️ 后退一步"):
        st.session_state.is_running = False # 停止自动播放
        step_backward(STEP_SIZE)
        
    # 按钮 2: 自动播放/暂停
    if st.session_state.is_running:
        button_label = "⏸️ 暂停"
    else:
        button_label = "▶️ 播放"

    if col_btn2.button(button_label):
        # 切换 is_running 状态并强制重新运行
        st.session_state.is_running = not st.session_state.is_running
        # **关键点：切换状态后，立即触发一次重跑**
        st.rerun() 
        
    # 按钮 3: 前进
    if col_btn3.button("▶️ 前进一步"):
        st.session_state.is_running = False # 停止自动播放
        step_forward(STEP_SIZE)

    # 按钮 4: 重置 (重置到 INITIAL_SHIFT_T)
    if col_btn4.button("⏪ 重置"):
        st.session_state.is_running = False
        st.session_state.current_t = INITIAL_SHIFT_T
        

    # --- D. Matplotlib 绘图 ---
    
    # 获取当前状态时间
    t_shift = st.session_state.current_t
    
    # ... (Matplotlib 绘图部分代码保持不变) ...
    
    # 设置 Matplotlib 子图
    fig, (ax0_1, ax0_2, ax1, ax2) = plt.subplots(
        4, 1, 
        figsize=(8, 8), 
        sharex=True,
        gridspec_kw={
            'hspace': 0.4,
            'height_ratios': [1, 1, 2, 2] 
        }
    )
    plt.subplots_adjust(top=0.95, bottom=0.05) 

    # 确定轴范围
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
    
    # 3. 卷积过程图
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

    # 4. 最终卷积结果图
    idx_max = np.searchsorted(conv_t, t_shift, side='right')
    
    if idx_max > 0:
        # 仅绘制到当前 t_shift 的数据点
        t_plot = conv_t[:idx_max]
        y_plot = conv_result[:idx_max]
        ax2.plot(t_plot, y_plot, label='$f_1(t) * f_2(t)$', color='blue')
        
        # 红点标记当前积分结果
        conv_value = np.interp(t_shift, conv_t, conv_result) 
        current_t = t_shift 

    else:
        # 如果当前时间小于卷积起始时间，红点在起始位置
        conv_value = 0.0
        current_t = st.session_state.conv_t_start

    ax2.plot([current_t], [conv_value], 'ro', markersize=8, label='当前积分结果')
    ax2.set_title(f'最终卷积结果 $f_1(t) * f_2(t)$', fontsize=10)
    ax2.set_xlabel('t')
    ax2.set_ylabel('幅度', fontsize=8)
    ax2.set_ylim(min_y_conv, max_y_conv)
    ax2.set_xlim(conv_t[0], conv_t[-1])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--')

    st.pyplot(fig)
    plt.close(fig)
    
    # --- E. 动画循环逻辑 ---
    if st.session_state.is_running:
        # 尝试向前推进
        moved = step_forward(STEP_SIZE)
        
        if moved:
            # 如果成功推进，短暂延迟后，强制重新运行脚本
            time.sleep(ANIMATION_DELAY)
            st.rerun() 
        else:
            # 到达终点，停止运行
            st.session_state.is_running = False
            st.toast("动画播放完毕！")

if __name__ == "__main__":
    main_convolution_app()