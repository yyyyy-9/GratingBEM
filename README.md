# GratingBEM

GratingBEM 是面向二维周期光栅的 Python 边界元计算工具，支持 TE/TM 衍射效率、解析直线与圆弧轮廓，以及波长、入射角和几何参数扫描。结果可以直接作为 NumPy 数组使用，并通过 Matplotlib 显示。

## 安装与运行

从 [Releases](https://github.com/yyyyy-9/GratingBEM/releases) 下载 `GratingBEM.zip`，完整解压并进入项目目录：

```bash
python -m pip install -e ".[plot]"
python main.py
```

需要 Python 3.10+。数值计算依赖 NumPy、SciPy，绘图使用 Matplotlib；不需要另外安装商业仿真软件。也可克隆本仓库后执行相同命令。

## 设置计算

修改根目录 [main.py](main.py) 顶部的设置：

```python
PROFILE = Profile.echelle(
    period_um=1000 / 79, blaze_deg=74, apex_deg=90,
    tip_radius_um=0.020, valley_radius_um=0.050,
)
WAVELENGTH_NM = 248.33
SOURCE = Incidence.littrow(-98)  # 或 Incidence.angle(74.0)
TARGET_ORDER = -98
METAL = Material.aluminum()
MODE = 'single'
```

几何长度用 µm，波长用 nm，角度用度。示例参考 Newport 79 线/mm、74° 中阶梯的名义参数；各参数的来源和假设见 [Newport 79 线示例](docs/zh/newport79.md)。轮廓也可设为矩形、梯形、台阶、自定义折线或直线/圆弧路径，见 [轮廓设置](docs/zh/profiles.md)。

## 扫描

`MODE` 可选择单次计算、单参数扫描、组合网格或配对扫描：

```python
MODE = 'angle'       # 入射角；另支持 wavelength、blaze
ANGLE_VALUES = np.linspace(73.5, 74.5, 7)

# 组合网格扫描，计算全部参数组合
MODE = 'grid'
SCAN_AXES = {
    'wavelength': np.linspace(247, 249, 5),
    'angle': np.linspace(73.5, 74.5, 7),
}
# 改为 MODE = 'paired' 时，等长数组按位置配对
```

波长扫描默认固定实验室入射角；需要每个波长重新满足 Littrow 时，设置 `WAVELENGTH_INCIDENCE = 'littrow'`。详见 [扫描用法](docs/zh/scanning.md)。

## 读取结果

`main.py` 在终端打印效率并显示图窗，默认不写结果文件。在 Jupyter/IPython 中执行 `%run -i main.py` 后，可直接取数组：

```python
data['orders']           # 最后一个计算点的全部传播级次
data['TE_orders_pct']    # 各级次 TE 效率，百分数
data['TM_orders_pct']    # 各级次 TM 效率，百分数
data['TE_map']           # 扫描目标级次效率，形状对应扫描轴
data['TM_map']
data['results']          # 各点的完整求解结果
```

也可在其他脚本中 `import main`，设置 `main.SHOW_PLOTS = False` 后调用 `main.run()`。Python API、CLI 和显式文件输出见 [快速使用](docs/zh/quickstart.md)。

## 计算方法与数值设置

裸界面求解采用周期 KM1 型双密度边界积分方程、Ewald 准周期 Green 函数与高阶 Nyström 离散。扫描支持单参数、组合网格和配对模式，并根据物理结构与 Bloch 相位自动组织计算。

对于同一结构、材料和波长下满足 Bloch 等价条件的入射方向，程序可共享边界积分算子的装配，并对 TE、TM 分别复用矩阵分解；各入射方向仍独立构造右端项，计算自己的衍射复振幅与效率。几何网格缓存和组间并行进一步减少重复工作。

[方法说明](docs/zh/methods.md)介绍方程、扫描复用条件、数值设置与结果读取；[相关参考文献](docs/zh/references.md)按主题列出周期光栅、边界积分、Green 函数及多入射求解方面的论文。

适用二维周期、沿槽线不变的 TE/TM 问题；不包含三维锥形入射或粗糙度模型。单层介质膜可显式使用 [实验性涂层接口](docs/zh/coating.md)。

网格、求积阶数和核截断由 `NUMERICS` 设置。实际使用应检查这些设置的独立收敛，详见 [数值设置与适用范围](docs/zh/accuracy.md)。

## 源码与许可

求解器位于 `src/gratingbem`，使用示例位于 `examples`。仓库中的开发测试可安装 `.[test]` 后运行 `python -m pytest -q`。

项目采用 [MIT 许可](LICENSE)。功能变化见 [更新记录](CHANGELOG.md)；问题与建议可提交至 [Issues](https://github.com/yyyyy-9/GratingBEM/issues)。
