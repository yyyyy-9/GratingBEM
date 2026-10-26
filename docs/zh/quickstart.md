# 从安装到第一组 TE/TM 衍射效率

## 安装

建议 Python 3.10 或更新版本，以及单独虚拟环境。进入解压后的项目目录（包含 `pyproject.toml`）：

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
python -m gratingbem --version
```

也可从 [GitHub 预发布](https://github.com/yyyyy-9/GratingBEM/releases) 下载匹配版本的 wheel，运行 `python -m pip install 路径/gratingbem-0.1.0a1-py3-none-any.whl`。当前不使用 PyPI 同名包作为本项目的发布来源。

不需要安装 MATLAB、COMSOL、S4 或 PCGrate。主要依赖只有 NumPy、SciPy。尚未验证所有 Windows Python/BLAS 组合；Windows 上建议先用 `--workers 1`。

## 实例：79 线/mm，−98 级 Littrow

生成配置模板：

```bash
python -m gratingbem init --shape echelle -o example.json
```

默认模板设定波长 248.33 nm、79 线/mm、74° 闪耀角、90° 顶角、圆角顶 20 nm、底 50 nm；模型为**理想裸 Al 半空间**，不是 Newport 厂家的加工截面数据。

先检查：

```bash
python -m gratingbem inspect example.json
```

这会打印周期、波长、入射角（约 74.004934°）、铝复介电常数、槽段数、预计积分节点数。若节点超过配置的 `max_nodes` 会显式报错。

再计算：

```bash
# macOS/Linux 可选：OPENBLAS_NUM_THREADS=1 python -m gratingbem run example.json -o results/79line --workers 4
python -m gratingbem run example.json -o results/79line --workers 1
```

`--workers 4` 会在具有合适多进程支持的环境中加速曲线积分矩阵装配（线性方程求解仍使用 BLAS）。

### 查看结果

`results/79line/` 中含：

| 文件 | 内容 |
|---|---|
| `result.json` | 输入问题、求解参数、耗时、诊断信息和所有传播级次 |
| `orders.csv` | 每个 `order` 的 TE/TM 复振幅与效率（0–1 比例） |
| `profile.svg` | 光栅轮廓预览，几何等比例 |
| `profile_segments.json` | 解析直线/圆弧段参数 |
| `profile_vertices.csv` | 关键顶点（圆弧不是折线采样） |
| `run_info.json` | 包版本、NumPy/SciPy、配置 SHA256 和系统信息 |

注意：输出的目标级次 `*_target_pct` 是**百分数**；CSV `*_efficiency` 是 0–1 的**比例**。

## 用 Python 直接调用

```python
from gratingbem import Profile, Material, Incidence, Simulation, Numerics, calculate, efficiency

profile = Profile.echelle(period_um=1000/79, blaze_deg=74, apex_deg=90,
                          tip_radius_um=0.02, valley_radius_um=0.05)
sim = Simulation(profile, wavelength_nm=248.33,
                 incidence=Incidence.littrow(-98), target_order=-98,
                 material=Material.aluminum())
result = calculate(sim, Numerics(p_order=4, max_panel_wavelengths=1.0), workers=1)
print(result.summary())
print('目标级次效率 %', efficiency(result,-98,'TE'), efficiency(result,-98,'TM'))
result.save('results/from_python')
```

若需要自定义铝介电常数，例如来自椭偏测量：`Material.constant(-8+1j)`，注意假定 `exp(-iωt)`。它代表整块均匀半空间，而非薄膜。

## Python 直接查看特定级次

```python
order = -98
idx, = (result.modes == order).nonzero()
if len(idx):
    print('TE (%)', 100*result.efficiencies['TE'][idx[0]])
    print('TM (%)', 100*result.efficiencies['TM'][idx[0]])
else:
    print('该级次在空气侧不传播')
```

**计算结果不是默认收敛结果。** 真正用于比较器件性能之前，请继续阅读精度验证章节。
