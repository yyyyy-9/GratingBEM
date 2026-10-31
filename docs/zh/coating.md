# 实验性单层介质膜

`gratingbem.coating` 提供单层共形介质膜的双界面边界积分计算，与裸界面入口分开。材料和厚度需显式给出；深槽、高阶与薄膜条件应独立检查收敛。

## 命令行

```bash
python -m gratingbem.coating init -o coated.json
python -m gratingbem.coating inspect coated.json
python -m gratingbem.coating run coated.json -o results/coated --workers 2 --block-rows 12 --timeout 120
```

默认在输出目录保存矩阵装配检查点。意外中断后可执行相同命令接续；输入不一致的检查点不会复用。`--fresh` 重新计算，`--checkpoint-dir` 可指定保存目录。

`--timeout` 监控装配工作进程的完成时间；超时后使用较小行块重试，仍无法完成时明确失败并保留已完成检查点。它不限制整个 Python 程序或稠密线性求解的总时长。

## Python API

```python
from gratingbem.coating import Film, CoatedSimulation, AssemblyOptions, calculate

coated = CoatedSimulation(simulation, Film(thickness_nm=5, epsilon=2.25))
options = AssemblyOptions(
    checkpoint_dir='results/coated/checkpoint',
    workers=2, block_rows=12, block_timeout_s=120, resume=True,
)
result = calculate(coated, numerics, assembly_options=options)
print(result.summary())
```

`simulation` 和 `numerics` 的定义见 [Python API](quickstart.md)。交互脚本中也可设置 `COATED = True`、`FILM_THICKNESS_NM` 和 `FILM_EPSILON`；该入口采用内存内串行装配，不保存检查点。

此接口为研究性模型，不提供多层膜、通用 RCIP 或严格误差上界。实际器件的材料和截面参数应另行确认。
