# 计算、扫描、收敛与结果文件

## 扫描一个几何参数

```python
from gratingbem import Profile,Material,Incidence,Simulation,Numerics,scan
p=Profile.echelle(period_um=1000/79, blaze_deg=74, apex_deg=90,
                  tip_radius_um=.02, valley_radius_um=.05)
sim=Simulation(p,248.33,Incidence.littrow(-98),-98,Material.aluminum())
settings=Numerics(p_order=4,max_panel_wavelengths=1.0,max_nodes=1800)
sweep=scan(sim,'profile.tip_radius_um',[.01,.02,.03],settings,workers=4,on_error='record')
sweep.save('results/radii')
```

每个点有独立 `case_000/...` 文件，汇总在 `scan.csv`。扫描中若某个轮廓几何非法，可选 `on_error='record'` 保留错误，不丢弃其他案例。**扫描本身不会保证每个点都收敛**。

## 二维扫描

```python
from gratingbem import scan_grid
runs = scan_grid(sim, {
    'profile.apex_deg': [85,90,95],
    'profile.tip_radius_um': [.01,.02],
}, settings,workers=4, on_error='record')
runs.save('results/grid')
```

这会独立求解 3 × 2 个组合，遇到几何不合法点会记录错误。当前没有跨几何矩阵缓存复用。

## 自动经验加密

```python
from gratingbem import calculate_adaptive, AdaptiveOptions
ar=calculate_adaptive(sim,settings,AdaptiveOptions(
    target_abs_tol=.001, # 目标级次效率相邻网格差异 <= 0.1 百分点（无量纲比值）
    min_levels=3,max_levels=5,
), workers=4,checkpoint_dir='results/adaptive')
print(ar.status)
print(ar.summary())
ar.save('results/adaptive')
```

`empirically_converged` 只代表约定的网格/独立审计在经验阈值内；**并非严格 Maxwell 误差上界**。`not_converged`、时间限制或审计未完成都应保留为不通过。

## 命令行操作

```bash
python -m gratingbem run example.json -o results/run --workers 4
python -m gratingbem scan example.json -o results/scan --workers 4
python -m gratingbem converge example.json -o results/adapt --workers 4
```

`scan` 从配置中的 `scan.parameter`、`scan.values` 读取参数；也可用 `scan.axes` 定义笛卡儿积。`on_error` 仅接受 `raise` 或 `record`。CLI 的 `run`／`scan` 遇到被动性检查失败会返回 2，同时保留计算和诊断；扫描摘要分别记录执行失败的 `failed_cases` 与求解完成但被动性失败的 `passivity_failed_cases`。`converge` 只有达到所选审计条件的经验收敛才返回 0，其余终态返回 2。

单段平面无法构造不同的分段分配，因此默认独立分布审计会报告 `distribution_check_incomplete`，不能当作已通过；用户可明确设置 `independent_distribution_check=False`，此时结论仅覆盖其余启用的审计。任何配置均不将 `accuracy_certified` 变为 true。

## 加速与耗时

主要耗时在边界积分矩阵装配。对于有圆弧的轮廓，使用 `workers=4` 多进程行块装配；若内存紧张或 Windows spawn 存在限制，先用 `workers=1`。建议设置 `OPENBLAS_NUM_THREADS=1` 降低多进程与 BLAS 抢核。输出分别记录 `assembly_s`、`solve_s`、`total_s`。机器硬件、处理器数量、Ewald 截断和并行策略均会影响耗时。
