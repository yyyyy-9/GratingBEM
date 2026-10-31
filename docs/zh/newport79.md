# Newport 79 线示例与参数来源

示例使用二维周期中阶梯轮廓，参考 Newport / Richardson Gratings 的 `53-*-402E` 名义槽密度与闪耀角。

## 结构与设置

| 参数 | 示例值 | 来源与含义 |
| --- | --- | --- |
| 槽密度 | 79 线/mm | Newport 官方目录 Table T3 的名义值 |
| 周期 | `1000 / 79 = 12.658227848… µm` | 由示例采用的槽密度换算 |
| 闪耀角 | 74° | 官方目录的名义闪耀角 |
| 顶角 | 90° | 理想槽形假设，非厂家实测截面 |
| 槽顶 / 槽底圆角 | 20 / 50 nm | 示例建模值，非厂家公差或测量数据 |
| 波长 | 248.33 nm | 示例计算条件，可自行修改 |
| 目标级次 | −98 | 示例采用的衍射级次与程序符号约定 |
| 入射角 | 约 74.004934° | 由所设周期、波长与级次的 Littrow 条件求得 |
| 材料 | 裸铝半空间 | 使用 Rakić Lorentz–Drude 体材料模型 |

几何由每周期两条槽面及可选的相切圆弧构成；金属位于轮廓下方，空气位于上方。设置见 [main.py](../../main.py) 与 [JSON 示例](../../examples/echelle_79line.json)。

## 厂家来源

[Newport / Richardson Gratings 官方衍射光栅目录](https://api.p1.mks.com/mam/celum/MKS-Newport-Gratings-Catalog.pdf#page=18) 的 Table T3 在 `53-*-402E` 项下列出 79 线/mm、74°。同目录 Table A4 的激光应用表列为 79.01 线/mm；本示例按 Table T3 的 79 线/mm 设置，使用具体器件时应按其实际规格调整。

目录的名义参数不能确定实际槽顶、槽底、氧化层或镀膜截面。示例的顶角和圆角仅用于定义可计算的理想模型。

## 材料来源

铝介电常数采用 A. D. Rakić 等人的 Lorentz–Drude 参数，见 [Optical properties of metallic films for vertical-cavity optoelectronic devices](https://doi.org/10.1364/AO.37.005271)，Applied Optics 37, 5271–5283 (1998)。实现位于 [materials.py](../../src/gratingbem/_core/materials.py)。

体材料模型不自动包含实际器件的表面氧化物、介质膜或粗糙度；计算效率不代表厂家实测效率。
