import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- 1. 缓存计算、函数解析 ---

@st.cache_data
def calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt=0.005):
    """
    计算卷积核心数据，并使用 Streamlit 缓存避免重复计算。
    """
    t = np.arange(t_start, t_end, dt)
    
    # 辅助函数（在作用域内定义或从外部导入，此处选择在外部定义，但为了保持自包含性，此处暂保留）
    def u(x):
        return (x >= 0).astype(float)
    def rect(x, width=1):
        return (np.abs(x / width) <= 0.5).astype(float)
        
    f1 = evaluate_function(f1_str, t, u, rect)
    f2 = evaluate_function(f2_str, t, u, rect)
    
    # 执行离散卷积，并乘以 dt 模拟积分
    conv_result = signal.convolve(f1, f2, mode='full') * dt
    
    # 计算卷积结果的时间轴
    conv_t_len = len(t) + len(t) - 1
    conv_t = np.arange(conv_t_len) * dt + 2 * t_start
    
    # 确定 Y 轴的显示范围
    max_y_orig = np.max([np.max(f1), np.max(f2), 1.0]) * 1.2
    min_y_orig = np.min([np.min(f1), np.min(f2), 0.0]) * 1.2
    max_y_conv = np.max([np.max(conv_result), 1.0]) * 1.2
    min_y_conv = np.min([np.min(conv_result), 0.0]) * 1.2
    
    return t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv

def evaluate_function(func_str, t, u, rect):
    """
    解析并计算输入的函数表达式
    """
    env = {
        'np': np, 't': t, 'pi': np.pi, 'exp': np.exp, 'cos': np.cos,
        'sin': np.sin, 'abs': np.abs, 'sqrt': np.sqrt, 'u': u, 'rect': rect 
    }
    try:
        # 使用自定义环境执行表达式
        y_raw = eval(func_str, {"__builtins__": None}, env)
        
        if isinstance(y_raw, bool) or isinstance(y_raw, (int, float)):
            y = np.full_like(t, float(y_raw), dtype=float)
        elif isinstance(y_raw, np.ndarray):
            y = y_raw.astype(float)
        else:
            return np.zeros_like(t) 

        # 清理 NaN/Inf 值
        y[np.isnan(y)] = 0.0
        y[np.isinf(y)] = 0.0
        return y
    except Exception as e:
        return np.zeros_like(t)

# --- 2. Plotly 绘图函数（仅显示最终结果） ---

def create_static_plotly_figure(t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv, t_start, t_end):
    """
    绘制 Plotly 图表：只显示 f1(t)，f2(t) 和最终的卷积结果。
    """
    
    # 创建 3 个子图：f1(t), f2(t), f1(t)*f2(t)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1, 
        row_heights=[0.3, 0.3, 0.4],
        subplot_titles=('f₁(t) 原始信号', 'f₂(t) 原始信号', 'f₁(t) * f₂(t) 最终结果')
    )

    # 1. f1(t) 原始信号 (Row 1)
    fig.add_trace(go.Scatter(x=t, y=f1, mode='lines', name='f₁(t)', line=dict(color='red', width=2), showlegend=True), row=1, col=1)
    
    # 2. f2(t) 原始信号 (Row 2)
    fig.add_trace(go.Scatter(x=t, y=f2, mode='lines', name='f₂(t)', line=dict(color='green', width=2), showlegend=True), row=2, col=1)
    
    # 3. 最终卷积结果图 (Row 3)
    fig.add_trace(go.Scatter(x=conv_t, y=conv_result, mode='lines', name='f₁(t) * f₂(t)', line=dict(color='blue', width=2), showlegend=True), row=3, col=1)

    # --- 布局和轴设置 ---
    fig.update_xaxes(
        range=[t_start, t_end], 
        showgrid=True, gridwidth=1, gridcolor='lightgray', 
        zeroline=False, linecolor='black', mirror=True, ticks='outside', 
        showticklabels=True
    )
    
    fig.update_yaxes(
        title_text='幅度', 
        linecolor='black', mirror=True, 
        showgrid=True, gridwidth=1, gridcolor='lightgray', 
        zeroline=True, zerolinewidth=1, zerolinecolor='black',
        ticks='outside', ticklen=5 
    )

    # 设置 Y 轴范围
    fig.update_yaxes(range=[min_y_orig, max_y_orig], row=1, col=1)
    fig.update_yaxes(range=[min_y_orig, max_y_orig], row=2, col=1)
    fig.update_yaxes(range=[min_y_conv, max_y_conv], row=3, col=1)
    
    # 设置 X 轴标题
    fig.update_xaxes(title_text='t', row=3, col=1)

    fig.update_layout(
        height=600, 
        template="plotly_white", 
        font=dict(size=10),
        title_text="连续信号卷积结果 $f_1(t) * f_2(t)$",
        showlegend=True, 
        legend=dict(
            orientation="h", yanchor="top", y=1.02, xanchor="left", x=0.01, 
            bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="#000000", borderwidth=0.5, 
            font=dict(size=8)
        ),
        margin=dict(l=20, r=20, t=50, b=20) 
    )
    
    # 强制子图标题左对齐
    for annotation in fig['layout']['annotations']:
        annotation['x'] = 0.01 
        annotation['xanchor'] = 'left' 
        annotation['font']['size'] = 12
        
    return fig


# --- 3. Streamlit 主应用函数 ---

def main_convolution_app():
    st.set_page_config(layout="wide") 
    
    st.markdown("# 连续信号卷积运算结果展示", unsafe_allow_html=True)
    st.markdown("输入 $f_1(t)$ 和 $f_2(t)$ 的表达式，系统将直接计算并展示最终卷积结果。")
    st.markdown("支持的函数：`u(t)` (单位阶跃)，`rect(t, width)` (矩形脉冲)，以及 `np.exp()`, `np.sin()`, `np.cos()` 等。")
    st.markdown("---")
    
    # 定义默认参数
    dt = 0.005
    
    # --- A. 输入控制区 ---
    st.header("输入控制")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # f1(t) 初始值: rect(t, 4)
        f1_str = st.text_input("f₁(t) =", value="rect(t, 4)")
    with col_f2:
        f2_str = st.text_input("f₂(t) =", value="rect(t, 2)")
        
    col_t_start, col_t_end, col_button = st.columns([1, 1, 1])
    with col_t_start:
        t_start = st.number_input("T_start:", value=-6.0, step=1.0) 
    with col_t_end:
        t_end = st.number_input("T_end:", value=10.0, step=1.0)
    
    with col_button:
        # 按钮触发计算，并清除缓存以强制更新
        st.write("---") # 垂直对齐按钮
        if st.button("▶️ 运行并更新结果"):
             calculate_convolution_data.clear() # 清除缓存，强制重新计算
             st.experimental_rerun() # 重新运行应用，使用新缓存值
    
    st.markdown("---")
    
    if t_start >= t_end:
        st.error("起始时间必须小于结束时间。")
        return

    # --- B. 数据计算 (缓存调用) ---
    t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv = \
        calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt)
    
    # --- C. Plotly 绘图和显示 ---
    
    st.header("卷积运算结果展示")
    
    fig = create_static_plotly_figure(t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv, t_start, t_end)

    # 在 Streamlit 中显示 Plotly 图表
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main_convolution_app()