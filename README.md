# 复杂大气环境下激光弱目标回波仿真范例

这是一个可以直接运行的物理仿真范例。它不依赖实际实验数据，也不使用“给理想脉冲叠加任意高斯噪声”的做法，而是依次计算大气衰减、目标散射、背景光、散粒噪声、暗计数和读出噪声，再生成 H0/H1 回波波形并计算信噪比。

## 运行方法

当前环境只需要 Python 和 NumPy：

```powershell
python simulate.py
```

运行后会重新生成：

- `scenario_results.csv`：72 个大气—距离—湍流组合的 SNR 数据
- `representative_waveforms.npz`：一组代表性 H0/H1 波形
- `representative_waveform.svg`：代表性波形图
- `snr_vs_range.svg`：中等湍流下不同能见度的 SNR—距离曲线
- `snr_vs_visibility.svg`：5 km 距离下 SNR—能见度曲线
- `summary.json`：代表场景、随机种子、样本数及关键结果

## 范例采用的系统与场景

| 参数 | 范例值 |
|---|---:|
| 激光波长 | 1064 nm |
| 脉冲能量 | 100 μJ |
| 脉冲宽度 | 20 ns FWHM |
| 光束发散角 | 0.30 mrad |
| 接收孔径 | 150 mm |
| 滤光片带宽 | 2 nm |
| 接收视场半角 | 0.50 mrad |
| 探测器量子效率 | 0.70 |
| 暗计数率 | 1000 counts/s |
| 读出噪声 | 12 e− rms |
| 目标模型 | 小型朗伯漫反射面 |
| 目标反射率 | 0.20 |
| 目标投影面积 | 0.05 m² |
| 距离 | 1–10 km |
| 能见度 | 2–20 km |
| 湍流 | 弱、中、强三档 |
| 每个场景蒙特卡罗次数 | 300 |

这些数字只是示范参数，不是某台真实设备的技术指标。正式研究中必须根据课题指定的激光器、探测器、目标和大气条件替换。

## 核心计算

小型朗伯目标接收到的脉冲能量为：

$$
E_r=
E_t\eta_t\eta_r\eta_pT_{\mathrm{atm}}^2
\frac{4\rho A_p\cos\theta_sA_r}
{\pi^2\theta^2R^4}
$$

其中大气单程透射率为：

$$
T_{\mathrm{atm}}=\exp(-\gamma R)
$$

本范例用 Koschmieder 关系和 Angstrom 波长修正近似消光系数：

$$
\gamma(\lambda)=
\frac{3.912}{V}
\left(\frac{\lambda}{550\ \mathrm{nm}}\right)^{-q}
$$

正式研究应使用 MODTRAN、Py6S 或可引用文献中的消光系数替代这个简化式。

闪烁因子 \(\xi\) 采用均值为 1 的对数正态模型：

$$
\xi=\exp(\sigma_{\ln I}z-\sigma_{\ln I}^2/2),
\qquad z\sim\mathcal N(0,1)
$$

目标信号电子数和背景电子数为：

$$
N_s=\eta_q\frac{E_r}{h\nu},
\qquad
N_b=\eta_q\frac{P_b\tau_{\mathrm{win}}}{h\nu}
$$

其中：

$$
P_b=L_\lambda\Delta\lambda A_r\Omega_{\mathrm{FOV}}\eta_r
$$

每个时间采样点的观测值由信号、背景、暗计数、泊松散粒噪声和读出噪声共同组成。归一化匹配滤波器为：

$$
\mathbf h=\frac{\mathbf g}{\|\mathbf g\|_2},
\qquad
z=\mathbf h^{\mathsf T}\mathbf y
$$

波形信噪比按匹配滤波后的信号电平与噪声标准差之比计算：

$$
\mathrm{SNR}_{\mathrm{dB}}
=20\log_{10}
\frac{z_s}{\sigma_z}
$$

## 结果示例

代表场景为 8 km 距离、10 km 能见度、中等湍流。结果显示：

- 平均接收脉冲能量约为 \(1.24\times10^{-19}\) J
- 平均目标信号约为 0.45 个光电子
- 100 ns 观测窗内的背景光电子约为 833 个
- 理论匹配滤波 SNR 约为 −46.14 dB
- 300 次蒙特卡罗得到的平均 SNR 约为 −46.16 dB
- 由于信号远弱于背景和噪声，单脉冲几乎无法可靠检测

这说明在范例参数下，需要采用更强的发射能量、更大的目标有效散射截面、更窄的滤光片和视场、更多脉冲累积或更低的探测器噪声。可以直接修改 `TargetScene` 和 `LaserSystem` 参数进行对比。

## 如何在课题报告中定位

建议表述为：

> 本文建立了激光大气传输、目标散射及光电探测噪声模型，通过蒙特卡罗方法生成复杂大气环境下的弱目标回波仿真数据，并对匹配滤波后的信噪比和检测性能进行了数值分析。

不要表述为“实验结果”或“实测验证”。模型可信性可以通过以下方式补充：

1. 检查自由空间、零反射率和零噪声等极限情况。
2. 检查小目标回波是否近似满足 \(R^{-4}\) 距离衰减。
3. 增加蒙特卡罗次数，确认 SNR 均值和分位数收敛。
4. 将晴空和雾天衰减趋势与公开文献或 MODTRAN 结果对比。
5. 做能见度、湍流强度、反射率和探测器噪声的参数敏感性分析。

## 文件说明

- `simulate.py`：完整可运行代码
- `scenario_results.csv`：每个场景一行，可直接用 Excel 打开
- `representative_waveforms.npz`：NumPy 波形数据，可用 `numpy.load()` 读取
- `summary.json`：关键结果摘要
- 三个 `.svg` 文件：浏览器可直接打开的参数扫描和波形图
