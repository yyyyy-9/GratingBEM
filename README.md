# GratingBEM

[![Tests](https://github.com/yyyyy-9/GratingBEM/actions/workflows/ci.yml/badge.svg)](https://github.com/yyyyy-9/GratingBEM/actions/workflows/ci.yml)

**Bare-interface diffraction-grating calculations using a 2D periodic Maxwell boundary integral equation (KM1 / high-order Nyström).**

面向计算光栅的研究型 Python 包。支持金属反射光栅的 TE/TM、全部传播级次、复杂槽形以及几何参数扫描，**不包含涂层算法和未经充分验证的通用 RCIP**。公开接口不再使用历史 v9–v16 模块名。

> **Research alpha (0.1.0a1)** — 可安装、可重复计算、具备独立测试的源码发行包；不是已通过所有材料和轮廓验证的商业求解器，也未发布到 PyPI。所有衍射效率需做独立网格与积分阶数检查。

## 最快上手（3 条命令）

```bash
python -m pip install .
python -m gratingbem init --shape echelle -o my_grating.json
python -m gratingbem run my_grating.json -o results/echelle --workers 4
```

结果：`results/echelle/result.json`（计算摘要和全部级次）、`orders.csv`（逐级 TE/TM 效率）、`profile.svg`（轮廓）、`run_info.json`（软件版本和运行环境）。

若只想核对几何、材料、入射角和预计未知量，不必运行求解：

```bash
python -m gratingbem inspect my_grating.json
```

## Python 示例

```python
from gratingbem import (Profile, Material, Incidence, Simulation,
                        Numerics, calculate, scan, efficiency)

# Newport 402E 参数的理想化几何；圆角和材料是模型假设，不是厂家测量
profile = Profile.echelle(
    period_um=1000 / 79,
    blaze_deg=74,
    apex_deg=90,
    tip_radius_um=0.020,
    valley_radius_um=0.050,
)
sim = Simulation(
    profile=profile,
    wavelength_nm=248.33,
    incidence=Incidence.littrow(-98),
    target_order=-98,
    material=Material.aluminum(),
)
settings = Numerics(p_order=4, max_panel_wavelengths=1.0)
result = calculate(sim, settings, workers=1)
print(result.summary())
print('目标级次 TE (%)', efficiency(result, -98, 'TE'))
result.save('results/one_run')

radius_scan = scan(sim, 'profile.tip_radius_um',
                   [0.005, 0.01, 0.02], settings, workers=1)
radius_scan.save('results/radius_scan')
```

结果中的 `TE_target_pct`、`TM_target_pct` 是目标级次**绝对入射功率百分比**，不是相对镜面效率。单位：`wavelength_nm` 为 nm，`period_um` / 高度 / 圆角半径为 µm，角度为度，复介电常数无量纲。损耗约定为 `exp(-iωt)`，被动材料应有 `Im(ε)≥0`。

Python 脚本使用多进程时，请将计算和扫描放在 `if __name__ == '__main__':` 保护下，再设置 `workers=4`；上面的 Python 示例先用单进程，CLI 已处理启动保护。级次必须为整数，Littrow 的正负号按 `kx(m)=kx(入射)+2πm/d` 定义，回返时 `sinθ=-mλ/(2d)`。角度和 Littrow 条件只能指定一种。

## 已支持的功能

- 单界面周期金属 / 分段均匀介质半空间，TE/TM，固定波长或 Littrow 条件。
- 解析直线、解析圆弧、矩形、梯形、多级阶梯、中阶梯、圆角与用户自定义混合路径。
- 双密度周期 KM1 传输积分方程、准周期 Green 函数、解析对数奇异项和高阶 Nyström。
- 多进程矩阵装配、单次计算、单参数扫描、二维扫描与经验自适应加密。
- 输出传播级次、边界矩阵残差、被动性检查、面元数和装配/求解耗时。

**不支持：** 氧化膜或涂层、多层、有限厚度透射结构、3D 或锥形入射、沿 x 回折的悬垂轮廓、Wood/Rayleigh 奇点处的稳定计算、通用 RCIP。平滑圆角采用真实曲线求积；尖角默认采用分级网格而非已验证 RCIP。

## 质量与风险

程序从历史 BEM v16 裸光栅研究内核整理而来。发布审查修正了输入歧义、正 Littrow 级次的角度符号及扫描退出状态，原 −98 级参考计算保持。测试包括平面 Fresnel、记录值回归、轮廓合法性、并行一致性、CLI/扫描/收敛和文件输出；详见 [验证记录](docs/zh/verification.md)及[精度边界](docs/zh/accuracy.md)。**总反射率不超过 100% 是必要而非充分条件；小矩阵残差也不代表真实 Maxwell 效率已收敛。** 研究数据应保存网格/多项式阶数、Ewald 截断与独立数值参考。

## 使用文档

- [安装与 10 分钟教程](docs/zh/quickstart.md)
- [矩形、阶梯、圆弧、自定义轮廓](docs/zh/profiles.md)
- [参数扫描、数值设置及结果](docs/zh/workflows.md)
- [精度验证、模型边界与常见错误](docs/zh/accuracy.md)
- [开发与正式发布检查表](docs/zh/release.md)
- [本地测试与参考数值记录](docs/zh/verification.md)

本包的核心为 Python + NumPy + SciPy。`pip install '.[test]'` 后运行 `python -m pytest`，额外积分测试用 `python -m pytest -m slow`。完整集合用 `python -m pytest -o addopts=''`。GitHub [Releases](https://github.com/yyyyy-9/GratingBEM/releases) 提供 wheel、源码包及校验摘要；当前未上传 PyPI。

## Citation / reference

Research implementation of the 2D periodic Kleinman–Martin-style transmission BIE. Foundational papers include Helsing & Karlsson, [*On a Helmholtz Transmission Problem in Planar Domains with Corners*](https://arxiv.org/abs/1711.09796). Material dispersion: A. D. Rakić *et al.*, *Applied Optics* **37**, 5271–5283 (1998), [DOI:10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271). These identify formulation/material background, not an independent validation of every implementation path. No claim is made that the research solver reproduces a manufacturer's proprietary groove shape or certified diffraction efficiency.

MIT license. See [LICENSE](LICENSE).
