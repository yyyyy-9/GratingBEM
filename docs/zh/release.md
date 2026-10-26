# 发布与维护指南

本项目以 **0.1.0a1 科研 alpha** 在 [GitHub](https://github.com/yyyyy-9/GratingBEM) 维护，版本标签为 `v0.1.0a1`，Release 标记为预发布。它提供 wheel、源码发行包、完整 ZIP 与 SHA256SUMS；未发布到 PyPI。原始输入 ZIP 保持不变，发布包从审查后的源码构建。

## 推荐代码结构

```text
src/gratingbem/
    __init__.py      稳定公开接口
    model.py         几何/波长/材料/入射定义及校验
    api.py           calculate / scan / convergence
    cli.py           init / inspect / run / scan / converge
    _core/           求解器数值核心（内部实现，不承诺稳定）
examples/            用户示例与配置（不是日志）
docs/zh/             用户教程及误差说明
tests/               自动回归与接口测试
```

不同研究版本的实验日志、失败原型和 RCIP、涂层模块均不放入发行源码包。旧物理内核的代码被整理进单一 `_core` 命名空间，公开 API 不再暴露历史版本号。

## 本地测试与构建

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m pip wheel . --no-deps -w dist
python -m gratingbem init -o grating.json
python -m gratingbem inspect grating.json
```

安装后从项目外的目录试用 `python -m gratingbem run`，排除源码目录污染导入。CI覆盖Linux/macOS/Windows、Python3.10/3.12；慢积分基准只在一组环境执行，避免全矩阵重复。实际执行状态见工作流与验证说明。MIT 许可及 GratingBEM contributors 署名按输入项目保留。将来推送 PyPI 前仍需单独确认包名和发布授权。

本次发布检查先复现并修复了小数级次截断、入射/材料定义冲突、正 Littrow 角度符号、未知扫描错误策略及被动性失败扫描返回成功；新增测试进入原 pytest 集合。版本重构夹具、Fresnel 基准和 Windows 双进程装配分别验证，不将它们混称独立 RCWA 或实验验证。发布时核对源码、wheel/sdist、ZIP及线上附件摘要，测试输出只作当前批次临时数据。

## 将来镀膜包的隔离

镀膜算法尚未通过 79 线/mm 大周期金属光栅高精度验收，因此必须另设实验分支或未来的 `gratingbem.multilayer` 模块，不应混入当前的稳定入口。不要把材料 `Material.constant` 当作介质膜求解。
