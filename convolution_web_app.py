import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- 1. 缓存计算、函数解析 (保持不变) ---

@st.cache_data
def calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt=0.005):
    """
    计算卷积核心数据，并使用 Streamlit 缓存避免重复计算。
    """
    t = np.arange(t_start, t_end, dt)
    
    # 定义辅助函数
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

# --- 2. Plotly 绘图函数（保持不变） ---

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
        height=700, # 稍微增加高度以适应 3 个子图
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


# --- 3. Streamlit 主应用函数 (侧边栏布局) ---

def main_convolution_app():
    # 设置为 wide 布局以提供更多绘图空间
    st.set_page_config(layout="wide") 
    
    # 定义默认参数
    dt = 0.005
    
    # --- A. 左侧：侧边栏控制区 ---
    with st.sidebar:
        st.markdown("# 连续信号卷积运算展示", unsafe_allow_html=True)
        st.markdown("---")
        
        st.header("输入控制")
        st.markdown("支持的函数：`u(t)`, `rect(t, width)`, `np.exp()`, `np.sin()`, `np.cos()` 等。")
        
        # f1(t) 和 f2(t) 输入
        f1_str = st.text_input("f₁(t) =", value="rect(t, 4)")
        f2_str = st.text_input("f₂(t) =", value="rect(t, 2)")
        
        # T_start 和 T_end
        col_t_start, col_t_end = st.columns(2)
        with col_t_start:
            t_start = st.number_input("T_start:", value=-6.0, step=1.0) 
        with col_t_end:
            t_end = st.number_input("T_end:", value=10.0, step=1.0)
        
        st.markdown("---")
        
        # 按钮触发计算，并清除缓存以强制更新
        if st.button("▶️ 运行并更新结果", use_container_width=True):
             if t_start >= t_end:
                 st.error("起始时间必须小于结束时间。")
                 return
             calculate_convolution_data.clear() # 清除缓存
             st.rerun() # 重新运行应用，触发新计算
             
    # --- B. 数据校验 (防止 Streamlit 在侧边栏输入前就报错) ---
    if t_start >= t_end:
        # 如果不满足条件，只显示侧边栏，不显示主区域图表
        return

    # --- C. 数据计算 (主区域上方不放任何组件) ---
    t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv = \
        calculate_convolution_data(f1_str, f2_str, t_start, t_end, dt)
    
    # --- D. 右侧：Plotly 绘图和显示 ---
    
    # 主区域不放标题，标题已移至侧边栏
    
    fig = create_static_plotly_figure(t, f1, f2, conv_t, conv_result, max_y_orig, min_y_orig, max_y_conv, min_y_conv, t_start, t_end)

    # 在 Streamlit 主区域显示 Plotly 图表
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main_convolution_app()