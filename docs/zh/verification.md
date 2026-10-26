# 0.1.0a1 发布验证

验证日：2026-10-26。当前本地环境为 Windows、Python 3.13.12、NumPy 2.5.2、SciPy 1.17.1、pytest 9.1.1；科学依赖复用现有安装，测试工具放在独立临时环境。原 ZIP 的 SHA256 为 `312a0d1eeedb4468a924e39bf983863a300023ea930c1e4b119d3ab97e5e9a56`，原文件不修改。

## 本次实际执行

- 原默认集合：11项通过，1项慢测试未选择。
- 审查反例先失败：小数级次截断、入射条件冲突、命名材料与固定介电常数冲突、缺失实部、未知扫描错误策略、正Littrow角度符号，以及真实粗网格扫描被动性失败仍返回成功。
- 修复后完整集合：`python -m pytest -q -o addopts=''`，**31项全部通过**（29项默认、2项慢积分场景），本次约22.75秒。计时只代表这一台机器。
- 独立解析Fresnel：有损铝正/负入射角、无损介质正入射，TE/TM反射率均在测试要求的0.002绝对比例误差内。
- 原echelle记录值：79线/mm、74°闪耀、90°顶角、248.33nm、−98级Littrow、20/50nm圆角，276节点，目标效率TE 67.2150396872%、TM 63.6508251293%在固定容差内复现。这是旧记录值回归，本次没有重新取得前身v16源码逐元素对照。
- 实际Windows双进程曲线装配与单进程传播级次复振幅一致，rtol=1e-11、atol=1e-12。
- CLI覆盖init/inspect/run/scan/converge和输出。真实粗网格总反射率超限时scan返回2，分别记录执行失败与被动性失败。单段平面无法改变独立分布时保留审计未完成；显式关闭该审计的测试只检查其余启用条件，accuracy_certified仍为false。

## 公开材料和安装

语法、依赖、MIT许可、构建清单及内部文档链接纳入发布核对。输入wheel中21个共同Python模块与输入源码相符，但额外含5个未列入源码的历史模块：_circle、_legacy_geometry、_legacy_kernel、_legacy_kernel_batched、_legacy_material。本次不沿用该旧wheel，从当前源码重建并严格核对安装模块集合；原ZIP保留。

安装验证从源码目录外运行公开CLI，核对实际安装位置、元数据版本和安装代码与冻结源码字节。CI使用固定官方Actions提交，覆盖Linux/macOS/Windows及Python3.10/3.12；Linux/Python3.10固定NumPy1.23.5和SciPy1.10.1核对最低依赖系列，其余使用可解析的新版本。两项较慢积分场景只在Linux/Python3.12执行。真实线上状态见[GitHub工作流](https://github.com/yyyyy-9/GratingBEM/actions/workflows/ci.yml)，配置存在不等于已运行。

## 结果适用范围

没有独立高精度RCWA/FMM或实验效率对照，没有严格后验误差上界、多层涂层或通用RCIP验证。参考夹具、Fresnel解析核对、并行一致性和经验收敛分别报告。MIT许可保留输入项目的GratingBEM contributors署名；GitHub为科研alpha预发布，PyPI不在本次发布范围。

临时日志、矩阵和安装环境完成本批判断后清理；测试源码和必要结论随项目维护，不另积存逐次运行报告。
