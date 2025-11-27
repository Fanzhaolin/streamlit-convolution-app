import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import time

# --- 1, 2, 3. 缓存计算、函数解析、状态管理 ---

# ************ 关键修改: 初始平移时间改为 -3.0 ************
INITIAL_SHIFT_T = -3.0 
# **********************************************************

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

def initialize_state(conv_t_start, conv_t_end, initial_t):
    # 重置逻辑保持不变，确保使用新的 initial_t
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

# --- 4. Plotly 绘图函数 (标题左对齐修正，保持不变) ---

def create_plotly_figure(t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv, t_start, t_end, f2_str):
    
    t_shift = st.session_state.current_t
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06, 
        row_heights=[0.18, 0.18, 0.32, 0.32],
        subplot_titles=('f₁(t) 原始信号', 'f₂(t) 原始信号', 
                        f'卷积过程: f₁(\u03c4) 和 f₂({t_shift:.2f}-\u03c4)',
                        'f₁(t) * f₂(t) 最终结果')
    )

    # --- 计算动态数据 ---
    t_for_f2 = t_shift - t 
    f2_shifted = evaluate_function(f2_str, t_for_f2)
    product = f1 * f2_shifted
    
    
    # 1. f1(t) 原始信号 (Row 1)
    fig.add_trace(go.Scatter(x=t, y=f1, mode='lines', name='f₁(t)', line=dict(color='red', width=2), showlegend=True), row=1, col=1)
    
    # 2. f2(t) 原始信号 (Row 2)
    fig.add_trace(go.Scatter(x=t, y=f2, mode='lines', name='f₂(t)', line=dict(color='green', width=2), showlegend=True), row=2, col=1)
    
    # 3. 卷积过程图 (Row 3)
    fig.add_trace(go.Scatter(x=t, y=f1, mode='lines', name='f₁(\u03c4)', line=dict(color='red', width=2), showlegend=True), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=f2_shifted, mode='lines', name=f'f₂({t_shift:.2f}-\u03c4)', line=dict(color='green', dash='dash', width=2), showlegend=True), row=3, col=1)
    
    x_fill = np.concatenate([t, t[::-1]])
    y_fill = np.concatenate([product, np.zeros_like(product)[::-1]]) 
    
    fig.add_trace(go.Scatter(
        x=x_fill, 
        y=y_fill, 
        fill='toself', 
        fillcolor='rgba(255, 165, 0, 0.3)', 
        line=dict(width=0),
        name='乘积', 
        showlegend=True
    ), row=3, col=1)

    # 4. 最终卷积结果图 (Row 4)
    idx_max = np.searchsorted(conv_t, t_shift, side='right')
    
    if idx_max > 0:
        t_plot = conv_t[:idx_max]
        y_plot = conv_result[:idx_max]
        
        fig.add_trace(go.Scatter(x=conv_t, y=conv_result, mode='lines', name='f₁(t) * f₂(t) (全)', line=dict(color='lightgray', width=1, dash='dot'), showlegend=False), row=4, col=1)
        
        fig.add_trace(go.Scatter(x=t_plot, y=y_plot, mode='lines', name='f₁(t) * f₂(t)', line=dict(color='blue', width=2), showlegend=True), row=4, col=1)
        
        conv_value = np.interp(t_shift, conv_t, conv_result) 
        current_t = t_shift 
        
        fig.add_trace(go.Scatter(x=[current_t], y=[conv_value], mode='markers', name='当前积分结果', 
                                 marker=dict(color='red', size=8), showlegend=True), row=4, col=1)
    else:
        fig.add_trace(go.Scatter(x=conv_t, y=conv_result, mode='lines', name='f₁(t) * f₂(t)', line=dict(color='blue', width=2), showlegend=True), row=4, col=1)
        fig.add_trace(go.Scatter(x=[conv_t[0]], y=[0.0], mode='markers', name='当前积分结果', 
                                 marker=dict(color='red', size=8), showlegend=True), row=4, col=1)
        
    # --- 布局和轴设置 ---
    
    fig.update_xaxes(
        range=[t_start, t_end], 
        showgrid=True, gridwidth=1, gridcolor='lightgray', 
        zeroline=False, 
        linecolor='black', mirror=True, 
        ticks='outside', ticklen=5,
        title_text='t'
    )
    
    fig.update_yaxes(
        title_text='幅度', 
        linecolor='black', mirror=True, 
        showgrid=True, gridwidth=1, gridcolor='lightgray', 
        zeroline=True, zerolinewidth=1, zerolinecolor='black',
        ticks='outside', ticklen=5 
    )

    fig.update_yaxes(range=[min_y_orig, max_y_orig], row=1, col=1)
    fig.update_yaxes(range=[min_y_orig, max_y_orig], row=2, col=1)
    fig.update_yaxes(range=[min_y_orig, max_y_orig], row=3, col=1)
    fig.update_yaxes(range=[min_y_conv, max_y_conv], row=4, col=1)
    fig.update_xaxes(title_text='\u03c4', row=3, col=1)
    fig.update_xaxes(title_text='t', row=4, col=1)

    fig.update_layout(
        height=600, 
        template="plotly_white", 
        font=dict(size=10),
        title_text=None,
        
        showlegend=True, 
        legend=dict(
            orientation="v", 
            yanchor="top", y=0.98, 
            xanchor="right", x=1.0, 
            bgcolor="rgba(255, 255, 255, 0.7)", 
            bordercolor="#000000", borderwidth=0.5, 
            font=dict(size=8)
        ),
        margin=dict(l=20, r=20, t=30, b=20) 
    )
    
    # 强制子图标题左对齐
    for annotation in fig['layout']['annotations']:
        annotation['x'] = 0.01 
        annotation['xanchor'] = 'left' 
        annotation['font']['size'] = 10
        
    return fig


# --- 5. Streamlit 主应用函数 (保持不变) ---

def main_convolution_app():
    st.set_page_config(layout="wide") 
    
    # 1. 标题移到侧边栏
    st.sidebar.markdown("### 连续信号卷积运算智能体", unsafe_allow_html=True)
    st.sidebar.markdown("---") 
    
    dt = 0.01 
    STEP_SIZE = 0.2
    ANIMATION_DELAY = 0.01 

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
    
    # 初始化时使用新的 INITIAL_SHIFT_T
    initialize_state(conv_t_start, conv_t_end, INITIAL_SHIFT_T)

    # --- 2. 当前平移时间移入侧边栏 ---
    st.sidebar.markdown(f"**当前平移时间 $t = {st.session_state.current_t:.2f}$**")
    st.sidebar.markdown("---")
    
    # --- C. 动画控制区 (位于主页面图表上方) ---
    
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1.5, 1.5, 1.5, 1.5, 4])
    
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
        # 重置按钮使用新的 INITIAL_SHIFT_T
        st.session_state.current_t = INITIAL_SHIFT_T
        

    # --- D. Plotly 绘图和显示 ---
    
    fig = create_plotly_figure(t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv, t_start, t_end, f2_str)

    st.plotly_chart(fig, use_container_width=True)
    
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