import streamlit as st
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os

# 字体设置（保持不变）
font_path = os.path.join(os.path.dirname(__file__), 'NotoSansCJK-Regular.otf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="抛光垫稳态热传导分析", layout="wide")
st.title("🔬 抛光垫稳态热传导分析（含沟槽修正）")

# ==================== 核心计算函数 ====================
def compute_temperature(k_list, thickness_list, T_bottom, q_flux, num_nodes=1000):
    total_L = sum(thickness_list)
    z = np.linspace(0, total_L, num_nodes)
    k_node = np.zeros(num_nodes)
    layer_boundaries = np.cumsum([0] + thickness_list)
    for i, pos in enumerate(z):
        for layer_idx in range(3):
            if layer_boundaries[layer_idx] <= pos <= layer_boundaries[layer_idx + 1]:
                k_node[i] = k_list[layer_idx]
                break
    R_cumulative = np.zeros(num_nodes)
    for i in range(num_nodes - 2, -1, -1):
        dz = z[i + 1] - z[i]
        k_avg = (k_node[i + 1] + k_node[i]) / 2
        R_cumulative[i] = R_cumulative[i + 1] + (dz / k_avg)
    T_steady = T_bottom + q_flux * R_cumulative
    interfaces_z = [0] + np.cumsum(thickness_list).tolist()
    interfaces_T = []
    for z_val in interfaces_z:
        idx = np.argmin(np.abs(z - z_val))
        interfaces_T.append(T_steady[idx])
    return z, T_steady, interfaces_z, interfaces_T, layer_boundaries

# ==================== 绘制结构示意图 ====================
def draw_schematic(use_groove, depth_ratio=0.4, figsize=(4, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.axis('off')
    y0, y1, y2, y3 = 8.5, 5.5, 4.5, 1.0
    x0, x1 = 0.5, 5.5
    # 层3
    ax.add_patch(Rectangle((x0, y3), x1-x0, y2-y3, facecolor='#a6d96a', edgecolor='black', linewidth=2))
    ax.text((x0+x1)/2, (y2+y3)/2, '副垫层', ha='center', va='center', fontsize=10, weight='bold')
    # 层2
    ax.add_patch(Rectangle((x0, y2), x1-x0, y1-y2, facecolor='#f9d057', edgecolor='black', linewidth=2))
    ax.text((x0+x1)/2, (y2+y1)/2, '胶粘剂层', ha='center', va='center', fontsize=10, weight='bold')
    # 层1
    if use_groove:
        ax.add_patch(Rectangle((x0, y1), x1-x0, y0-y1, facecolor='#7fc0f0', edgecolor='black', linewidth=2))
        ax.text((x0+x1)/2, 0.42*(y0+y1), '抛光层', ha='center', va='center', fontsize=10, weight='bold')
        depth_px = depth_ratio * (y0 - y1) if 0 < depth_ratio < 1 else 0.3*(y0-y1)
        groove_bottom_y = y0 - depth_px
        num_grooves = 5
        groove_width = (x1 - x0) / (num_grooves * 1.5)
        spacing = groove_width * 0.5
        start_x = x0 + (x1 - x0 - (num_grooves * groove_width + (num_grooves-1)*spacing)) / 2
        for i in range(num_grooves):
            xg0 = start_x + i * (groove_width + spacing)
            xg1 = xg0 + groove_width
            ax.add_patch(Rectangle((xg0, groove_bottom_y), groove_width, depth_px,
                                   facecolor='white', edgecolor='gray', linewidth=1))
        ax.text((x0+x1)/2, groove_bottom_y + depth_px/2, '沟槽', ha='center', va='center',
                color='gray', fontsize=8)
    else:
        ax.add_patch(Rectangle((x0, y1), x1-x0, y0-y1, facecolor='#7fc0f0', edgecolor='black', linewidth=2))
        ax.text((x0+x1)/2, (y0+y1)/2, '抛光层', ha='center', va='center', fontsize=10, weight='bold')
    # 热流箭头
    ax.arrow(x1+0.3, y0, 0, y3-y0, head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=2)
    ax.text(x1+0.6, (y0+y3)/2, '热流方向', rotation=90, va='center', ha='center', fontsize=9, weight='bold')
    return fig

# ==================== 侧边栏输入 ====================
st.sidebar.header("⚙️ 参数设置")
st.sidebar.subheader("导热系数 (W/(m·K))")
k1 = st.sidebar.number_input("层1 (抛光层)", value=0.2, step=0.01, format="%.3f")
k2 = st.sidebar.number_input("层2 (胶粘剂层)", value=0.15, step=0.01, format="%.3f")
k3 = st.sidebar.number_input("层3 (副垫层)", value=1.0, step=0.01, format="%.3f")
k_list = [k1, k2, k3]

st.sidebar.subheader("厚度 (m)")
L1 = st.sidebar.number_input("层1 (抛光层)", value=1e-3, step=1e-5, format="%.6f")
L2 = st.sidebar.number_input("层2 (胶粘剂层)", value=0.05e-3, step=1e-6, format="%.7f")
L3 = st.sidebar.number_input("层3 (副垫层)", value=2e-3, step=1e-4, format="%.6f")
thick_list = [L1, L2, L3]

st.sidebar.subheader("边界条件")
T_bottom = st.sidebar.number_input("底部温度 (°C)", value=18.0, step=0.5)
mode = st.sidebar.radio("计算模式", options=["给定热流", "给定顶部温度"])
if mode == "给定热流":
    q_flux_input = st.sidebar.number_input("顶部热流 (W/m²)", value=5000.0, step=100.0)
    T_top_input = None
else:
    T_top_input = st.sidebar.number_input("顶部温度 (°C)", value=80.0, step=1.0)
    q_flux_input = None

st.sidebar.subheader("沟槽修正（仅作用于第一层）")
use_groove = st.sidebar.checkbox("启用沟槽修正", value=False)
if use_groove:
    depth_ratio = st.sidebar.number_input("沟槽深度 / 第一层厚度", value=0.4, min_value=0.01, max_value=0.99, step=0.01)
    area_ratio = st.sidebar.number_input("沟槽面积 / 第一层面积", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
    k_air = st.sidebar.number_input("介质导热系数 (W/(m·K))", value=0.6, step=0.01, format="%.3f")
else:
    depth_ratio = 0.4
    area_ratio = 0.5
    k_air = 0.6

# ==================== 计算与显示 ====================
if st.sidebar.button("🚀 计算", type="primary"):
    try:
        k_eff_list = k_list.copy()
        groove_info = ""

        if use_groove:
            if not (0 < depth_ratio < 1):
                st.error("深度比例必须在(0,1)之间")
                st.stop()
            if not (0 <= area_ratio <= 1):
                st.error("面积比例应在[0,1]之间")
                st.stop()
            if k_air <= 0:
                st.error("介质导热系数必须为正")
                st.stop()
            volume_fraction = area_ratio * depth_ratio
            if volume_fraction > 1.0:
                st.error("体积分数（面积比×深度比）不能超过1")
                st.stop()
            k_solid = k_eff_list[0]
            if volume_fraction == 0:
                k_eff = k_solid
            else:
                k_eff = 1.0 / ((1 - volume_fraction) / k_solid + volume_fraction / k_air)
            k_eff_list[0] = k_eff
            groove_info = f"深度比例={depth_ratio:.2f}, 面积比例={area_ratio:.2f}, 介质k={k_air:.3f}, 体积分数={volume_fraction:.3f}\n原始 k₁={k_solid:.3f} → 等效 k₁={k_eff:.3f} W/(m·K)"

        # 计算
        if mode == "给定热流":
            q_flux = q_flux_input
            if q_flux < 0:
                st.error("热流密度建议为非负值")
                st.stop()
            z, T, interfaces_z, interfaces_T, layer_boundaries = compute_temperature(
                k_eff_list, thick_list, T_bottom, q_flux
            )
            title_suffix = f"顶部热流 {q_flux:.0f} W/m²"
            names = ["抛光表面 (顶部)", "抛光层/胶层界面", "胶层/副垫层界面", "副垫层底部 (压板)"]
            total_drop = interfaces_T[0] - interfaces_T[-1]
            heat_flux_val = q_flux
            top_temp_val = interfaces_T[0]
        else:
            T_top = T_top_input
            R_total = sum(thick_list[i] / k_eff_list[i] for i in range(3))
            if R_total <= 0:
                st.error("总热阻计算异常，请检查参数")
                st.stop()
            q_flux = (T_top - T_bottom) / R_total
            z, T, interfaces_z, interfaces_T, layer_boundaries = compute_temperature(
                k_eff_list, thick_list, T_bottom, q_flux
            )
            title_suffix = f"顶部温度 {T_top:.1f}°C, 热流 {q_flux:.2f} W/m²"
            names = ["抛光表面 (顶部)", "抛光层/胶层界面", "胶层/副垫层界面", "副垫层底部 (压板)"]
            total_drop = T_top - T_bottom
            heat_flux_val = q_flux
            top_temp_val = T_top

        # ==================== 布局美化 ====================
        st.divider()  # 分隔线

        # 第一行：关键指标卡片
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.metric("抛光表面温度", f"{interfaces_T[0]:.1f} °C")
        with col_metric2:
            st.metric("底部（压板）温度", f"{interfaces_T[-1]:.1f} °C")
        with col_metric3:
            st.metric("总温差", f"{total_drop:.1f} °C", delta=None)
        with col_metric4:
            st.metric("热流密度", f"{heat_flux_val:.2f} W/m²")

        st.divider()

        # 第二行：温度图（左） + 结构示意图与表格（右）
        left_col, right_col = st.columns([3, 2], gap="medium")

        with left_col:
            st.subheader("📈 温度分布曲线")
            fig_temp, ax_temp = plt.subplots(figsize=(8, 5))
            z_mm = z * 1000
            ax_temp.plot(z_mm, T, 'b-', linewidth=2.5, label='稳态温度分布')
            for i in range(1, len(layer_boundaries) - 1):
                ax_temp.axvline(x=layer_boundaries[i] * 1000, color='gray', linestyle='--', alpha=0.6)
            for i in range(3):
                x_center = (layer_boundaries[i] + layer_boundaries[i + 1]) / 2 * 1000
                y_center = np.min(T) + (np.max(T) - np.min(T)) * (0.85 - i * 0.15)
                label = f'层{i+1}\n$k$={k_eff_list[i]:.3f}'
                if i == 0 and use_groove:
                    label += ' (修正)'
                ax_temp.text(x_center, y_center, label, horizontalalignment='center', fontsize=10, weight='bold')
            for idx, z_val in enumerate(interfaces_z):
                z_mm_val = z_val * 1000
                ax_temp.plot(z_mm_val, interfaces_T[idx], 'ro', markersize=8)
                ax_temp.text(z_mm_val + 0.02, interfaces_T[idx] + 0.5, f'{interfaces_T[idx]:.1f}°C', fontsize=9)
            ax_temp.set_xlabel('厚度方向位置 (mm)  [0=抛光表面 → 底部]', fontsize=12)
            ax_temp.set_ylabel('温度 (°C)', fontsize=12)
            ax_temp.set_title(f'稳态轴向温度分布\n{title_suffix}', fontsize=14, weight='bold')
            ax_temp.grid(True, alpha=0.3)
            ax_temp.legend(loc='best')
            st.pyplot(fig_temp)

            st.caption(
                "**核心控制方程**：稳态一维热传导 $\\frac{d}{dz}\\left(k\\frac{dT}{dz}\\right)=0$，"
                "边界条件：顶部热流 $q=-k\\frac{dT}{dz}\\big|_{z=0}$，底部温度 $T(L)=T_{\\text{bottom}}$。"
                "（数值离散采用串联热阻法求解）"
            )

        with right_col:
            st.subheader("📐 结构示意图")
            fig_schem = draw_schematic(use_groove, depth_ratio if use_groove else 0.4, figsize=(4.5, 3.5))
            st.pyplot(fig_schem)

            st.subheader("📋 界面温度结果")
            df = pd.DataFrame({
                '界面': names,
                '位置 (mm)': [z * 1000 for z in interfaces_z],
                '温度 (°C)': interfaces_T
            })
            df['位置 (mm)'] = df['位置 (mm)'].map(lambda x: f"{x:.3f}")
            df['温度 (°C)'] = df['温度 (°C)'].map(lambda x: f"{x:.2f}")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # 沟槽修正详情（折叠）
            if use_groove:
                with st.expander("🔧 沟槽修正详情"):
                    st.info(groove_info)

    except Exception as e:
        st.error(f"计算错误: {str(e)}")
else:
    st.info("👈 请先在左侧侧边栏设置参数，然后点击『计算』按钮查看结果。")
    # 初始显示占位
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write("等待计算...")
    with col2:
        st.subheader("📐 结构示意图")
        fig_schem = draw_schematic(False)
        st.pyplot(fig_schem)

st.sidebar.markdown("---")
st.sidebar.caption("提示：修改参数后点击『计算』更新结果。")