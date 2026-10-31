# 定义周期光栅轮廓

GratingBEM 将材料界面定义为一条从 `(0,z0)` 走到 `(period,z0)` 的周期曲线。金属在界面下方，空气在上方；正法向指向空气。允许垂直壁，但**不支持 x 倒退的悬垂轮廓**。所有长度为 µm。

## 内置形状

```python
from gratingbem import Profile

p1 = Profile.rectangle(period_um=0.8, height_um=0.1, duty_cycle=0.5,
                       corner_radius_um=0.008)
p2 = Profile.trapezoid(period_um=0.8, height_um=0.1,
                       bottom_duty=0.65, top_duty=0.40)
p3 = Profile.staircase(period_um=1.2, height_um=0.2, steps=4,
                       duty_cycle=0.7, corner_radius_um=0.003)
p4 = Profile.echelle(period_um=1000/79, blaze_deg=74, apex_deg=90,
                     tip_radius_um=0.02, valley_radius_um=0.05)

for i,p in enumerate([p1,p2,p3,p4]):
    p.save_svg(f'profile_{i}.svg')
```

`tip_radius_um` 和 `valley_radius_um` 分别对应中阶梯槽顶、跨周期槽底的几何圆角半径；构造器会检查圆角半径是否能够被当前几何容纳。圆角是真圆弧，求解器在圆弧上积分，并非线段折线。

中阶梯示例的厂家名义参数与建模假设见 [Newport 79 线示例](newport79.md)。其他形状用于说明通用轮廓接口。

## 任意折线与多级台阶

```python
p = Profile.custom(period_um=1.2, vertices=[
    (0,0), (.2,0), (.2,.12), (.6,.12), (.6,.05), (.8,.05), (1.2,0)
])
# 对折线顶点做圆角（也可传每个顶点的半径列表）
rounded = p.rounded(0.008)
```

`Profile.steps(period_um=..., heights_um=[0,.1,.2,.1,0], widths_fraction=[.15,.2,.3,.2,.15])` 支持按每级宽度/高度定义阶梯。

## 直线与圆弧混合路径

```python
path = Profile.path(period_um=1.0)
path.line_to(.2,0)
path.line_to(.3,.1)
path.arc_to(.45,.1, radius=.1, turn='cw')
path.line_to(1.,0)
p = path.build(name='my_profile')
```

`arc_to()` 不会自动确保与前后直线相切；有相切要求时，建议先用预设轮廓的 `.rounded()` 或自行构造精确圆心和连接点。即使几何上合法，未相切点仍是数学尖角，应配合更细网格收敛检查。

## 导出与扫描

```python
p.save_svg('shape.svg')
# 预设轮廓支持修改 recipe 参数：
p2 = p4.with_parameter('apex_deg', 80)
```

由无 recipe 的任意自定义路径修改参数时，应每次重新建立完整 `Profile`，或 `scan(..., parameter='profile', values=[profile1, profile2,...])`。建议先做轮廓 SVG 与节点检查，再开始昂贵的 Maxwell 求解。
