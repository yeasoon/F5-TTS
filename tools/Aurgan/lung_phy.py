#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
respiratory_acoustics.py
A minimal 1-D acoustic transmission-line model of the respiratory tract
(0–6 generations, Weibel morphology) -> mouth frequency response & time-domain sound
< 220 lines, pure Python, ~5 s on laptop
"""

import numpy as np
from scipy.sparse import lil_matrix, linalg
import matplotlib.pyplot as plt
# import sounddevice as sd   # optional: pip install sounddevice

# -------------------------------------------------
# 1. 物理常数
# -------------------------------------------------
rho  = 1.14e-3      # g/cm^3
c    = 3.54e4       # cm/s
mu   = 1.86e-4      # g/cm/s
gamma = 1.4
kappa_over_Cp = 2.5e-5  # cm^2/s  (approx air)
E    = 1.0e6        # dyn/cm^2  cartilage
h_over_r = 0.1      # wall thickness ratio
Re_crit = 2000
K_src = 1.2e-3      # Pa, turbulent source strength

# -------------------------------------------------
# 2. Weibel 0–6 代数据  (FRC)
# -------------------------------------------------
weibel = {
    'L':  np.array([12.0, 4.8, 2.0, 0.8, 0.5, 0.4, 0.3]),      # cm
    'r':  np.array([0.90, 0.63, 0.47, 0.34, 0.25, 0.18, 0.13]), # cm
    'N':  2**np.arange(7)                                         # number of segments
}

# -------------------------------------------------
# 3. 分段 T-段参数
# -------------------------------------------------
def alpha_wall(omega, r):
    """黏-热-壁顺从衰减系数 alpha [Np/cm]"""
    visc = (omega**2/(2*rho*c**3))*(4/3*mu + (gamma-1)*kappa_over_Cp)
    Cw   = 2*np.pi*r**3/(E*h_over_r)          # cm^5/dyn
    Yw   = 1j*omega*Cw                        # 简化为纯顺从
    wall = omega/2 * np.imag(Yw)/(np.pi*r**2)
    return visc + wall

def T_section(w, L, r):
    """返回单位段：Zs = Rs + 1j*w*Ls , Yp = Gp + 1j*w*Cp"""
    S   = np.pi*r**2
    Lp  = rho/S
    Cp  = S/(rho*c**2)
    alp = alpha_wall(w, r)
    Rs  = 2*alp*rho*c/S
    Gp  = 0.0                         # 忽略并联电导
    return Rs, Lp, Cp, Gp

# -------------------------------------------------
# 4. 自动构建节点列表 & 连接
# -------------------------------------------------

nodes = []       # (gen, idx, x/L)  0=prox, 1=dist
edges = []       # (n1,n2, L_seg, r, Nseg)
node_of = {}     # (gen,idx,side) -> global node id
def build_network():


    # 根节点
    node_of[(0,0,0)] = 0
    nodes.append((0,0,0))
    node_id = 1

    for gen in range(7):
        Nseg = weibel['N'][gen]
        Lseg = weibel['L'][gen]
        r    = weibel['r'][gen]
        for idx in range(Nseg):
            # 远端节点
            node_of[(gen,idx,1)] = node_id
            nodes.append((gen,idx,1))
            node_id += 1
            # 连接：prox->dist
            edges.append((node_of[(gen,idx,0)], node_of[(gen,idx,1)], Lseg, r, 1))
            # 分叉到下一代
            if gen < 6:
                for child in range(2):
                    child_idx = 2*idx + child
                    node_of[(gen+1, child_idx, 0)] = node_id
                    nodes.append((gen+1, child_idx, 0))
                    node_id += 1
                    edges.append((node_of[(gen,idx,1)], node_of[(gen+1,child_idx,0)], 0, r, 0))
    return nodes, edges, node_id

nodes, edges, n_nodes = build_network()
print('total nodes:', n_nodes, 'edges:', len(edges))

# -------------------------------------------------
# 5. 装配 MNA 矩阵  (节点压力法)
# -------------------------------------------------
def assemble_MNA(omega):
    w = omega
    Y = lil_matrix((n_nodes, n_nodes), dtype=complex)
    # 遍历所有 edges
    for n1,n2, Lseg, r, Nseg in edges:
        if Lseg==0: continue      # 零长跳过
        Rs,Lp,Cp,Gp = T_section(w, Lseg, r)
        # 导纳
        Zs = Rs + 1j*w*Lp
        Yp = Gp + 1j*w*Cp
        # T 段：串联 Zs/2  each端，并联 Yp 中段
        Y[n1,n1] += 1/(Zs/2)
        Y[n2,n2] += 1/(Zs/2)
        Y[n1,n2] -= 1/(Zs/2)
        Y[n2,n1] -= 1/(Zs/2)
        # 并联支路挂在中点 -> 用平均近似挂到 n1,n2
        Y[n1,n1] += Yp/2
        Y[n2,n2] += Yp/2
    # 嘴辐射 (节点 0)
    a_rad = weibel['r'][0]
    Z_rad = 1j*w * 0.613*rho*a_rad / (np.pi*a_rad**2)
    Y[0,0] += 1/Z_rad
    # 终端 6 代 -> 肺实质特征阻抗
    S6 = np.pi*weibel['r'][6]**2
    c_p = 3500                     # cm/s
    rho_p = 0.3*rho + 0.7*1.0e-3   # g/cm^3
    Z_p = rho_p*c_p / S6
    for n in range(n_nodes):
        if nodes[n][0]==6 and nodes[n][2]==1:  # 远端
            Y[n,n] += 1/Z_p
    return Y

# -------------------------------------------------
# 6. 湍流源  (放在气管第一分叉)
# -------------------------------------------------
def source_vector(omega):
    """返回节点电流源向量 [n_nodes]"""
    Is = np.zeros(n_nodes, dtype=complex)
    w = omega
    # 选气管末端节点作为源节点
    src_node = node_of[(0,0,1)]
    # 计算 Re
    r0 = weibel['r'][0]
    v_mean = 400                 # cm/s  静息吸气
    Re = 2*r0*v_mean/(mu/rho)
    if Re > Re_crit:
        Prms = K_src * (Re**2 - Re_crit**2)
        # 体积速度源  (简单：同相位)
        Usrc = np.pi*r0**2 * np.sqrt(2)*Prms/(rho*c)
        Is[src_node] = Usrc
    return Is

# -------------------------------------------------
# 7. 扫频
# -------------------------------------------------
f = np.logspace(np.log10(50), np.log10(3000), 200)
w = 2*np.pi*f
H = np.zeros_like(w, dtype=complex)

for i, wi in enumerate(w):
    Y = assemble_MNA(wi)
    I = source_vector(wi)
    # 稀疏 LU 求解
    p = linalg.spsolve(Y.tocsr(), I)
    H[i] = p[0]          # 嘴压

# -------------------------------------------------
# 8. 绘图
# -------------------------------------------------
plt.figure(figsize=(6,4))
plt.semilogx(f, 20*np.log10(np.abs(H)/20e-6))
plt.xlabel('Frequency [Hz]')
plt.ylabel('SPL [dB re 20 µPa]')
plt.title('Mouth frequency response (0–6 gen transmission line)')
plt.grid(True, which='both')
plt.tight_layout()
# plt.show()
plt.savefig("respiratory_freq_response.png", dpi=300, bbox_inches="tight")
print("Figure saved → respiratory_freq_response.png")

# -------------------------------------------------
# 9. 时域合成 & 播放
# -------------------------------------------------
fs = 16000
N  = fs*4                      # 4 s
t  = np.arange(N)/fs
# 构造白噪声源 -> 用 H 滤波
rng = np.random.default_rng(1)
white = rng.standard_normal(N)
WHITE = np.fft.rfft(white)
# 插值到 freq 轴
H_interp = np.interp(np.fft.rfftfreq(N, 1/fs), f, H, left=0, right=0)
sig = np.fft.irfft(WHITE * H_interp, N)
sig = sig / np.max(np.abs(sig)) * 0.3

# sd.play(sig, samplerate=fs, blocking=True)
from scipy.io import wavfile
wavfile.write("respiratory_sound.wav", fs, (sig * 32767).astype(np.int16))
print("Audio saved → respiratory_sound.wav")