# 扫描用法

## 交互脚本

在 [main.py](../../main.py) 顶部选择 `MODE`：

| 模式 | 含义 | 主要设置 |
| --- | --- | --- |
| `single` | 单次计算 | 轮廓、波长、入射条件 |
| `angle` | 入射角扫描 | `ANGLE_VALUES` |
| `wavelength` | 波长扫描 | `WAVELENGTH_VALUES` |
| `blaze` | 闪耀角扫描 | `BLAZE_VALUES`，需要中阶梯轮廓 |
| `grid` | 全部参数组合 | `SCAN_AXES` |
| `paired` | 按位置配对 | `SCAN_AXES`，数组等长 |

```python
MODE = 'grid'
SCAN_AXES = {
    'wavelength': np.linspace(247, 249, 5),
    'angle': np.linspace(73.5, 74.5, 7),
}
```

此例计算 5 × 7 个点，`data['TE_map']` 与 `data['TM_map']` 的形状为 `(5, 7)`，单位为百分数。还可扫描 `blaze`、`apex`、`tip_radius`、`valley_radius`、`period` 或完整的 `profile.*` 参数名。`grid` 轴数组须严格单调。

三个及更多参数仍保留完整数组；用 `PLOT_AXES` 选择显示的两个轴，`PLOT_SLICE` 选择其他轴的切片。`paired` 只计算等长数组中对应位置的组合。`MAX_CALCULATIONS` 控制单次扫描的计算次数上限。

## 入射条件

波长扫描默认 `WAVELENGTH_INCIDENCE = 'fixed'`，保持基准波长对应的实验室入射角。设置为 `'littrow'` 时，每个波长重新求解指定级次的 Littrow 角度，要求 `SOURCE = Incidence.littrow(...)`。

显式扫描 `angle` 时不能同时逐波长跟踪 Littrow。改变闪耀角只改变轮廓，不改变入射角。

## 提取数组

```python
data['axes']              # 轴名和取值
data['TE_map']            # 目标级次 TE 效率
data['TM_map']
data['results']           # 每个计算点的完整结果
data['orders_each']       # 各点传播级次，允许随参数变化
data['efficiencies_each']

# 提取其他级次，无需重新求解；该级次不传播时取零
eta = order_efficiency_map(data, order=-97, polarization='TE')
```

在 Jupyter/IPython 中 `%run -i main.py` 后可直接调用上述函数。脚本默认显示图形、不写文件；需要时自行调用 `plt.savefig()`、`np.save()` 或结果对象的 `save()`。

## Python API

```python
from gratingbem import scan, scan_grid

one_axis = scan(simulation, 'incidence.angle_deg', [73.5, 74.0, 74.5], numerics)
grid = scan_grid(
    simulation,
    {'wavelength_nm': [247, 248.33, 249],
     'incidence.angle_deg': [73.5, 74.0, 74.5]},
    numerics,
    on_error='record',
)
```

`on_error='record'` 保留失败点及其他已完成结果。配置文件的 `scan` 项也可指定同样的扫描参数，配合 `python -m gratingbem scan` 输出文件。

## 并行设置

`SCAN_WORKERS` 与 `SCAN_BACKEND` 控制扫描任务的并行数和 `'thread'` / `'process'` 后端。Python API 对应 `scan_workers` 和 `scan_backend`。扫描级并行与单点 `workers>1` 不叠加；Windows/macOS 使用进程时，计算入口放在 `if __name__ == '__main__':` 中。

根据 CPU 和内存从较小的并行数开始，可在启动前设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`。完整结果顺序与输入顺序一致。参数点仍需按 [数值设置](accuracy.md) 判断收敛。
