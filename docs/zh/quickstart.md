# Python API 与命令行

交互式使用从根目录 [main.py](../../main.py) 开始，设置与数组读取见 [README](../../README.md)。下面提供库调用和文件输出方式。

## Python 调用

```python
from gratingbem import (
    Profile, Material, Incidence, Simulation, Numerics, calculate, efficiency,
)

profile = Profile.echelle(
    period_um=1000 / 79, blaze_deg=74, apex_deg=90,
    tip_radius_um=0.02, valley_radius_um=0.05,
)
simulation = Simulation(
    profile, wavelength_nm=248.33, incidence=Incidence.littrow(-98),
    target_order=-98, material=Material.aluminum(),
)
numerics = Numerics(
    p_order=4, max_panel_wavelengths=1.0,
    modes_air=155, modes_metal=25, oversample=16, max_nodes=1800,
)
result = calculate(simulation, numerics, workers=1)
print(result.summary())
print(efficiency(result, -98, 'TE'), efficiency(result, -98, 'TM'))

# 需要时显式保存
result.save('results/example')
```

`result.efficiencies` 中的效率为 0–1 比例，`efficiency()` 返回百分数。全部传播级次在 `result.modes`，复振幅在 `result.amplitudes`。示例参数来源见 [Newport 79 线示例](newport79.md)。

## 命令行

```bash
python -m gratingbem init --shape echelle -o grating.json
python -m gratingbem inspect grating.json
python -m gratingbem run grating.json -o results/run --workers 1
python -m gratingbem scan grating.json -o results/scan --workers 1
python -m gratingbem converge grating.json -o results/converge --workers 1
```

`inspect` 查看几何、材料和预计节点数。`run`、`scan`、`converge` 写入显式指定的目录；结果包含效率、复振幅、轮廓和计算设置。`scan` 读取配置的 `scan.parameter` / `scan.values`，也支持 `scan.axes` 组合网格。

计算或物理检查失败时 CLI 返回非零退出码，并在可写时保留诊断输出。脚本应同时检查退出码和结果状态。

## 经验加密

```python
from gratingbem import calculate_adaptive, AdaptiveOptions

adaptive = calculate_adaptive(
    simulation, numerics,
    AdaptiveOptions(target_abs_tol=0.001, min_levels=3, max_levels=5),
    workers=1,
)
print(adaptive.status)
```

`empirically_converged` 表示满足所设经验阈值。网格、求积阶数和核截断仍需分别考虑，见 [数值设置](accuracy.md)。

## 多进程

Windows/macOS 的 Python 脚本使用多进程时，将计算放在 `if __name__ == '__main__':` 内；CLI 已处理入口保护。可先使用 `workers=1`，再根据 CPU 和内存设置并行数。扫描设置见 [扫描用法](scanning.md)。
