# 均匀电子气 RDMFT 程序：理论、公式与实现说明

本文档汇总本仓库中 **简约密度矩阵泛函理论（RDMFT）** 在 **三维顺磁均匀电子气（HEG）** 上的实现，对应源码主要在 `include/*.hpp` 与 `src/main.cpp`。可与各头文件中的块注释对照阅读。

---

## 1. 项目定位与数据流

可执行文件 `rdmft_heg`（[`src/main.cpp`](../src/main.cpp)）完成以下流程：

1. 解析命令行：`--rs`（$r_s$ 列表）、**必填** `--funcs`（泛函键列表）、`--N`（$k$ 网格点数，须为奇数）、`--kmax`（$k_{\max}=f\cdot k_F$ 的因子 $f$）、`--out-dir`、`--nk-out`、`--force`、`--init-uniform`、`--verbose` 等。
2. 对每个泛函键构造 `Functional` 实例（见 [`include/Functional.hpp`](../include/Functional.hpp) 的 `make_functional` / `main.cpp` 的 `make`）。
3. 对每个 $r_s$：由 [`HEG::kF`](../include/HEG.hpp) 得 $k_F$，取 $k_{\max}=f\,k_F$，用 [`Grid::uniform_trapezoid`](../include/Grid.hpp) 生成 $[0,k_{\max}]$ 上的均匀节点与梯形权重；调用 [`ExchangeKernel::build`](../include/ExchangeKernel.hpp) 预计算矩阵 $W$。
4. 调用 [`solve_rdmft`](../include/Solver.hpp) 自洽求占据数 $n_i=n(k_i)$，再算动能、交换–关联能密度等。
5. **能量输出**：每个泛函一个 TSV（默认 `data/<stem>.tsv`）。每行给出 $E/N$、**关联能**（相对 HF）
   $$
   E_c^{\mathrm{RDMFT}} = \frac{E^{\mathrm{RDMFT}}}{N} - \frac{E^{\mathrm{HF}}}{N},
   $$
   其中 $E^{\mathrm{HF}}/N$ 用 [`HEG::HF_per_electron`](../include/HEG.hpp) 的解析式。**QMC 参考列**为 PW92 参数化的 $e_c(r_s)$（[`PW92::ec_per_electron`](../include/QMC.hpp)），用于对比而非变分目标。
6. **可选** `--nk-out`：在收敛时将 $k,\,n(k)$ 写入单独 TSV。

**源码映射**：驱动循环在 `main.cpp`；物理量在 [`Energy.hpp`](../include/Energy.hpp)；求解器在 [`Solver.hpp`](../include/Solver.hpp)。

---

## 2. 均匀电子气几何与 Hartree–Fock 基准

Wigner–Seitz 半径 $r_s$ 与数密度（原子单位，见 [`HEG.hpp`](../include/HEG.hpp)）：

$$
\rho = \frac{3}{4\pi r_s^3}.
$$

顺磁情况下费米波矢：

$$
k_F = \frac{(9\pi/4)^{1/3}}{r_s}.
$$

Hartree–Fock 极限下每电子动能与交换能（用于定义 $E_c$ 的减项）：

$$
t_s = \frac{3}{10}k_F^2,\qquad e_x = -\frac{3}{4\pi}k_F,\qquad e_{\mathrm{HF}} = t_s + e_x.
$$

---

## 3. RDMFT 能量与密度（每体积，Hartree）

对自旋非极化 HEG，自然轨道为平面波，**每自旋轨道**动量占据 $n(k)\in[0,1]$ 只依赖 $k=|{\bf k}|$。程序采用的每体积能量与密度为（[`Energy.hpp`](../include/Energy.hpp) 文件头注释；自旋因子已并入常数）：

$$
\frac{T}{V} = \frac{1}{2\pi^2}\int_0^\infty k^4\, n(k)\,dk,
$$

$$
\frac{E_{xc}}{V} = -\frac{1}{2\pi^3}\int_0^\infty\!\!\int_0^\infty k\,k'\, K\bigl(n(k),n(k')\bigr)\,\ln\left|\frac{k+k'}{k-k'}\right|\,dk\,dk',
$$

$$
\rho = \frac{1}{\pi^2}\int_0^\infty k^2\, n(k)\,dk.
$$

离散化：$k_i$ 与梯形权重 $w_i$ 满足 $\int_0^{k_{\max}} F(k)\,dk \approx \sum_i w_i F(k_i)$（[`Grid.hpp`](../include/Grid.hpp)）。

---

## 4. 交换核离散化：乘积积分与矩阵 $W$

### 4.1 困难与思路

被积函数含 $\ln|(k+k')/(k-k')|$，在 $k'=k$ 处为**可积对数奇异性**；若用朴素梯形在奇点附近采样，收敛极慢。

### 4.2 乘积积分（product integration）

在 $k'$ 轴分段网格上，将 $u(k') = k'\,g(k')$（程序里 $g$ 由 $K$ 与 $n$ 决定）在每一小区间上视为**分段线性**（帽函数叠加）。对 $\ln(k_i+k')$ 与 $\ln|k_i-k'|$ 与帽函数的卷积用**解析原函数**计算，从而：

- 当 $u(k')$ 在网格上分段线性时，内层积分在机器精度意义下与连续公式一致（对 HF 阶跃且 $k_F$ 对齐网格时交换能尤其稳）；
- 对光滑 $u$ 为二阶精度量级、且无对数奇异性引起的离散误差尖峰。

### 4.3 原函数（与 `ExchangeKernel.hpp` 一致）

采用约定 $0\cdot\ln 0=0$。记

$$
F_0(t) = t\ln|t| - t \quad\text{对应}\quad \int \ln|t|\,dt,
$$

$$
F_1(t) = \frac{t^2}{2}\ln|t| - \frac{t^2}{4} \quad\text{对应}\quad \int t\ln|t|\,dt\ (t\neq 0).
$$

对区间 $[a,b]$ 上 $\ln|c-k'|$ 与 $k'\ln|c-k'|$ 的积分，通过变量平移化为 $F_0,F_1$ 在端点之差；$\ln(k_i+k')$ 用 $t=k_i+k'$ 的同样原函数。

### 4.4 矩阵 $W$ 与能量组装

预计算稠密矩阵 $W_{ij}$（行主序存储），使得内层「交换势」型量可写为（因子化核时有快速路径，见下节）

$$
V_i = \sum_j W_{ij}\,k_j\,(\cdots)_j,
$$

其中 $(\cdots)_j$ 由 $K(n_i,n_j)$ 的结构决定：若 $K(n_i,n_j)=f(n_i)f(n_j)$，则 $V_i=f(n_i)\sum_j W_{ij} k_j f(n_j)$（[`EnergyEvaluator::V_inner`](../include/Energy.hpp)）。

交换–关联能密度：

$$
\frac{E_{xc}}{V} = -\frac{1}{2\pi^3}\sum_i w_i\,k_i\,V_i.
$$

同时构造转置存储 `Wt`，使 $\partial E_{xc}/\partial n_i$ 的计算中固定外标 $i$ 时对 $j$ 的访问连续（缓存友好），见 `deps_xc`。可选 **OpenMP** 并行 $O(N^2)$ 循环（编译宏 `_OPENMP`）。

---

## 5. Euler–Lagrange 方程与准粒子能量 $\varepsilon_i$

在固定 $\rho$ 下，对 $n_i\equiv n(k_i)$ 变分 $T+E_{xc}-\mu\rho$。离散权重下动能与密度对 $n_i$ 的导数比为 $\frac{1}{2\pi^2}w_i k_i^4$ 与 $\frac{1}{\pi^2}w_i k_i^2$，化简后得到用于投影梯度分支的**准粒子能量**（[`pseudo_energy`](../include/Energy.hpp)）：

$$
\varepsilon_i = \frac{k_i^2}{2} + \frac{1}{\mathrm{pref}_i}\frac{\partial (E_{xc}/V)}{\partial n_i},\qquad \mathrm{pref}_i = \frac{w_i k_i^2}{\pi^2}.
$$

对称核 $K(a,b)=K(b,a)$ 时，

$$
\frac{\partial (E_{xc}/V)}{\partial n_i} = -\frac{k_i}{\pi^3}\sum_j w_j\,W_{ji}\,k_j\,\frac{\partial K}{\partial n_i}(n_j,n_i),
$$

其中 $\partial K/\partial n_i$ 由 `Functional::kernel_grad(n_i,n_j)` 给出（因子化情形有专用快速路径）。

**物理约束**：内点 $0<n_i<1$ 时 $\varepsilon_i=\mu$；$n_i=0$ 对应 $\varepsilon_i>\mu$，$n_i=1$ 对应 $\varepsilon_i<\mu$（KKT 型互补）。

---

## 6. 各泛函的两体核 $K(n_i,n_j)$

以下均满足程序中 **JK 约定**下的 $E_{xc}$ 形式（见 [`Functional.hpp`](../include/Functional.hpp) 类注释）。默认可分离核 $K(n_i,n_j)=f(n_i)f(n_j)$，由 `f`、`df` 描述；否则覆盖 `kernel` / `kernel_grad`。

| 泛函 | $f(n)$ 或核 $K$ | 说明 |
|------|-----------------|------|
| **HF** | $f(n)=n$ | 标准 Hartree–Fock 交换在 RDMFT 写法下 |
| **Mueller / BB** | $f(n)=\sqrt{n}$ | Müller 泛函 |
| **GU** | 同 Müller | 有限体系中去自相互作用；**HEG 平面波下与 Müller 数值一致** |
| **Power($\alpha$)** | $f(n)=n^\alpha$ | Sharma 等；常用 $\alpha\sim 0.55$–$0.58$ |
| **CGA** | $K=\frac12\bigl[n_in_j+\sqrt{n_i(2-n_i)}\sqrt{n_j(2-n_j)}\bigr]$ | Csányi–Goedecker–Arias |
| **CHF** | $K=n_in_j+\sqrt{n_i(1-n_i)n_j(1-n_j)}$ | 实现为 `BetaFunctional(0.5)` |
| **Beta($\beta$)** | $K=n_in_j+\bigl[n_i(1-n_i)n_j(1-n_j)\bigr]^\beta$ | $\beta\to\infty$ 趋向 HF 空穴关闭；$\beta=\frac12$ 即 CHF |
| **GEO** | 见下式 **(实现)** | 三通道非乘积核 |
| **OptGeo** | $K=w_1 n_in_j+w_2(n_in_j)^{1/2}+w_3(n_in_j)^{3/4}$ | $w_k$ 由球面角 $(a,b,c)$ 平方归一化：$w_1=a^2/\|\cdot\|^2$ 等 |
| **HybOpt** | $(1-\lambda)n_in_j+\lambda n_i^\alpha n_j^\alpha$ | $\lambda\in[0,1]$；$\alpha\in(0,1)$（边界数值钳制） |
| **BBC1** | 弱–弱占据对 $\sqrt{n_in_j}$ 带**光滑**符号翻转 | 演示非乘积、梯度难；`smooth` 默认 $0.05$ |
| **BBC3** | 弱–弱 / 强–强 / 其余三种规则 + 光滑化 | HEG 上等价 BBC2 型配对规则 |

**GEO 实现（与 `GEOFunctional::kernel` 一致）**：令 $p=n_in_j$，

$$
K_{\mathrm{GEO}} = \frac{1}{4}\,p + \frac{1}{2}\,p^{1/2} + \frac{1}{4}\,p^{3/4}\quad (p>0),\quad K_{\mathrm{GEO}}(0,\cdot)=0.
$$

即三条通道 $(n_in_j,\,(n_in_j)^{1/2},\,(n_in_j)^{3/4})$ 的权重为 **$(w_1,w_2,w_3)=(1/4,\,1/2,\,1/4)$**，且 $K(1,1)=1$。**注意**：`GEOFunctional` 类上方个别英文注释块曾写另一种系数顺序，以 **上述代码公式** 为准。

**HybOpt** 与 **OptGeo** 的 CLI：`HybOpt@lambda;alpha`、`OptGeo@a;b;c`（分号分隔，shell 中常需引号）。

---

## 7. 自洽求解算法（[`Solver.hpp`](../include/Solver.hpp)）

### 7.1 密度约束与 $\mu$ 二分

目标密度 $\rho_{\mathrm{target}}=\rho(r_s)$ 由 [`HEG::density`](../include/HEG.hpp) 给出。对每个 SCF 步，在固定当前 $n$ 下构造辅助场后，对化学势 $\mu$ 在区间 $[\mu_{\mathrm{lo}},\mu_{\mathrm{hi}}]$ 上 **二分** `bisect_iter` 次：利用 $\rho(\mu)$ 的**单调性**（更大 $\mu$ 填充更多态），使 $\rho(\mu)\approx\rho_{\mathrm{target}}$。默认 $\mu_{\mathrm{lo}}=-50$、$\mu_{\mathrm{hi}}=50$（`SolveOptions`）。

### 7.2 占据更新：按泛函分岔

1. **因子化 Power 族**（HF、Müller、GU、Power）：  
   - 对 $\alpha\neq 1$，由 EL 得闭式反解 $n_i$（`update_occupations_power`）。  
   - **HF（$\alpha\to 1$）**：$\varepsilon_i=k_i^2/2-U_i/(\pi k_i)$ 上为阶跃；程序在**最接近 $\mu$ 的单个网格点**上做能量空间**线性插值**赋予分数占据，使 $\rho(\mu)$ 对 $\mu$ 连续，外层 $\mu$ 二分稳定。

2. **可加型核**（CGA、CHF、Beta）：写 $K=n_in_j+g(n_i)g(n_j)$（CGA 带整体因子 $\frac12$，在占据更新里用 `el_scale=0.5`）。EL 化为
   $$
   U^{HF}_i + g'(n_i)\,U^{g}_i = \pi k_i\left(\frac{k_i^2}{2}-\mu\right)
   $$
   （尺度与代码中 `update_occupations_additive` 一致）。  
   - CGA：$g(n)=\sqrt{n(2-n)}$，$g'(n)$ 的反解有闭式 $n=1-s/\sqrt{1+s^2}$。  
   - Beta：对 $u=1-2n$ 一维单调方程用二分 `invert_dgbeta`。

3. **GEO / OptGeo**：对三条通道 $f_1=n,\,f_2=\sqrt{n},\,f_3=n^{3/4}$ 分别算 $U_{1},U_{2},U_{3}$；EL 左侧对 $n_i$ **单调递减**（在 $U_\alpha\ge 0$ 时），对 $n_i\in(0,1)$ **二分**求根；若 $U$ 全为零则退回 HF 型阶跃规则。

4. **HybOpt**：两通道 HF + Power，LHS 对 $n$ 单调时用二分（`update_occupations_hf_power_mix` + 专用 $\mu$ 二分）。

5. **通用分支（如 BBC1）**：**投影梯度**（步长代码中固定 `0.10`）：$n_i\leftarrow\mathrm{clip}\bigl(n_i-\mathrm{step}\cdot(\varepsilon_i-\mu)\bigr)$，内层仍配合 $\mu$ 二分以满足 $\rho$。

### 7.3 线性混合

$$
n_i \leftarrow (1-\eta)\,n_i + \eta\, n_i^{\mathrm{target}},\qquad \eta=\texttt{mix}.
$$

- `SolveOptions` 默认：`mix=0.4`，`max\_iter=800`，`tol\_n=10^{-9}`。  
- **`main.cpp` 调用处**覆盖为：`tol_n=10^{-8}`，`max_iter=1200`，`mix=0.4`。**以驱动为准**。

收敛判据：$\max_i|n_i^{\mathrm{new}}-n_i|$ 小于 `tol_n`。

### 7.4 多起点与亚稳态

对 **可加型**（CGA/Beta，含 CHF）、**GEO/OptGeo**、**HybOpt**、**BBC3** 等，`needs_multistart` 为真：在「HF 阶跃」与多种宽度参数的 **Fermi 型抹阶** `initial_smeared` 之间多起点尝试；**保留每电子总能量 $E/N$ 最低**的收敛解（变分择优）。

**可加型技巧**：计算空穴通道 $U_g$ 时对 $n$ 用软钳制 $n\in[n_{\mathrm{floor}},1-n_{\mathrm{floor}}]$（代码中 $n_{\mathrm{floor}}=10^{-6}$）计算 $g(n)$，使 $g'$ 在饱和端不发散、帮助离开 HF 型局域极小；**能量仍用未钳制的 $n$**。

**`--init-uniform`**：若给定，则只用均匀初值、单起点。

---

## 8. $k$ 空间网格（[`Grid.hpp`](../include/Grid.hpp)）

- **生产默认**（`main.cpp`）：[`uniform_trapezoid(k_{\max},N)`](../include/Grid.hpp)，$k_i=i\,k_{\max}/(N-1)$，$N$ 为奇数（默认 $801$），$k_{\max}=f\,k_F$（默认 $f=3$）。
- **其他已实现网格**（供实验 / 遗留）：`uniform_simpson`、`trapezoid_with_node_at`（$k_F$ 对齐节点）、`graded_fermi_trapezoid`（费米面附近加密）、`log_trapezoid`（$\ln k$ 均匀）。当前驱动未调用后几种。

---

## 9. 附录：PW92 与 Ortiz–Ballone（1997）

### 9.1 PW92 关联能（程序用于 `Ec_QMC` 列）

Perdew–Wang 1992 参数化（[`PW92`](../include/QMC.hpp)）：

$$
e_c(r_s) = -2A(1+\alpha_1 r_s)\ln\!\left(1+\frac{1}{2A\bigl(\beta_1\sqrt{r_s}+\beta_2 r_s+\beta_3 r_s^{3/2}+\beta_4 r_s^2\bigr)}\right),
$$

系数 $A,\alpha_1,\beta_1,\ldots,\beta_4$ 见 `QMC.hpp` 中 `constexpr` 常量表。

### 9.2 Ortiz–Ballone $n(q)$（仅头文件参考）

[`OrtizBallone1997`](../include/QMC.hpp) 给出 DMC 拟合的动量分布分段多项式及离散 $r_s$ 表。**当前 `main.cpp` 不调用**；若需与文献 $n(k)$ 对比，需在脚本或其他驱动中自行读取系数。

---

## 10. Python 脚本与单元测试（简述）

| 路径 | 作用 |
|------|------|
| `scripts/plot_common.py` | 曲线列表与绘图样式 |
| `scripts/plot_results.py` | 读取 `data/*.tsv` 画 $E_c(r_s)$ 与 PW92 |
| `scripts/plot_nk*.py` | 读取 `data/nk/*.tsv` 画 $n(k)$ |
| `scripts/optimize_optGeo.py` / `optimize_hybopt.py` | 在 PW92 参考下数值拟合参数 |
| `tests/test_hf_exchange.cpp` | HF 极限下动能、交换、自洽能与解析值对比（容差约 $5\times 10^{-3}$） |

---

## 11. 参考文献（与代码注释一致）

- J. P. Perdew & Y. Wang, *Phys. Rev. B* **45**, 13244 (1992) — PW92。  
- A. M. K. Müller, *Phys. Lett. A* **105**, 446 (1984)。  
- A. Sharma, J. K. Dewhurst, N. N. Lathiotakis, E. K. U. Gross, *Phys. Rev. B* **78**, 201103(R) (2008) — Power。  
- S. Goedecker, C. J. Umrigar, *Phys. Rev. Lett.* **81**, 866 (1998) — GU。  
- G. Csányi, S. Goedecker, T. A. Arias, *Phys. Rev. A* **65**, 032510 (2002) — CGA。  
- G. Csányi, T. A. Arias, *Phys. Rev. B* **61**, 7348 (2000) — CHF。  
- O. V. Gritsenko, K. Pernal, E. J. Baerends, *J. Chem. Phys.* **122**, 204102 (2005) — BBC1。  
- N. N. Lathiotakis, N. Helbig, E. K. U. Gross, *Phys. Rev. B* **75**, 195120 (2007) — BBC3/BBC2 层次。  
- G. Ortiz & P. Ballone, *Phys. Rev. B* **56**, 9970 (1997) — 动量分布参数化（头文件表）。

---

*文档版本与仓库实现同步；若实现变更，请以对应头文件为准。*
