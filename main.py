"""GratingBEM 主脚本：单次、一维、组合网格和配对扫描，直接看图与数组。

安装: python -m pip install -e '.[plot]'
运行: python main.py
交互: Jupyter/IPython 中 %run -i main.py，结果在变量 data 中。

默认不写任何结果文件。网格扫描 data['TE_map'] 的形状严格对应 SCAN_AXES 的顺序。
"""
from __future__ import annotations

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from gratingbem import Profile, Material, Incidence, Simulation, Numerics, calculate, calculate_many

# ===================== 1. 只需修改这里 =====================
PROFILE = Profile.echelle(
    period_um=1000 / 79, blaze_deg=74, apex_deg=90,
    tip_radius_um=0.020, valley_radius_um=0.050,
)
WAVELENGTH_NM = 248.33
SOURCE = Incidence.littrow(-98)         # 或 Incidence.angle(74.0)
TARGET_ORDER = -98
METAL = Material.aluminum()

# single / angle / wavelength / blaze / grid / paired
MODE = 'single'
ANGLE_VALUES = np.linspace(73.5, 74.5, 7)
WAVELENGTH_VALUES = np.linspace(246, 251, 7)
BLAZE_VALUES = np.linspace(73.5, 74.5, 7)

# 多参数用这一处定义：键顺序就是 NumPy 多维数组的维度顺序。
# grid: 笛卡儿积（下例共 5 x 7=35 次）; paired: 同位置一一配对（需等长）。
SCAN_AXES = {
    'wavelength': np.linspace(247, 249, 5),  # nm
    'angle': np.linspace(73.5, 74.5, 7),     # degree
    # 'blaze': np.linspace(73.5, 74.5, 3),
}
# 波长变化而没有显式 angle 轴时：fixed 固定实验室入射角；
# littrow 在每个波长重新调整入射角。两者是不同的物理实验。
WAVELENGTH_INCIDENCE = 'fixed'       # 'fixed' / 'littrow'

# >=3 维网格绘图时，显示其中两个维度；其余维度取中间值或 PLOT_SLICE 指定的实际值。
PLOT_AXES = None                     # 例 ('wavelength', 'angle')
PLOT_SLICE = {}                      # 例 {'blaze': 74.0}（实际角度，取最近采样值）
MAX_CALCULATIONS = 500               # 防止无意触发上千次耗时仿真，可自行修改

NUMERICS = Numerics(p_order=4, max_panel_wavelengths=0.9,
                    modes_air=155, modes_metal=25, oversample=12, max_nodes=1800)
WORKERS = 4                         # 每次求解内部的矩阵装配进程数
SCAN_WORKERS = 1                    # >1: 多参数点同时计算，单点 workers=1
SCAN_BACKEND = 'thread'              # 'thread' / 'process' (spawn)
BLOCH_REUSE = 'auto'                 # 自动分组/有限Ewald核预检/失败后独立计算；可设 False 关闭
POLARIZATIONS = ('TE', 'TM')

# 单层介质涂层属于实验性求解器，深槽高阶光栅需独立验证数值精度。
COATED = False
FILM_THICKNESS_NM = 5.0
FILM_EPSILON = 2.25 + 0j

SHOW_PLOTS = True                    # 本地 / Notebook 直接 plt.show()
PRINT_ALL_ORDERS = True             # 输出最后一个计算点的所有传播级次
# ===========================================================

# 可用的扫描名；也允许 Simulation.with_parameter() 支持的 profile.xxx 完整键。
_AXIS_NAMES = {
    'angle': 'incidence.angle_deg', 'wavelength': 'wavelength_nm',
    'blaze': 'profile.blaze_deg', 'apex': 'profile.apex_deg',
    'tip_radius': 'profile.tip_radius_um',
    'valley_radius': 'profile.valley_radius_um',
    'period': 'profile.period_um',
}
_AXIS_LABELS = {
    'angle': 'Incidence angle (deg)', 'wavelength': 'Wavelength (nm)',
    'blaze': 'Blaze angle (deg)', 'apex': 'Apex angle (deg)',
    'tip_radius': 'Tip radius (µm)', 'valley_radius': 'Valley radius (µm)',
    'period': 'Period (µm)',
}


def _axis_parameter(name):
    if name in _AXIS_NAMES:
        return _AXIS_NAMES[name]
    if name in ('wavelength_nm', 'incidence.angle_deg') or name.startswith('profile.'):
        return name
    raise ValueError(f'不认识的扫描参数 {name!r}；可使用 angle/wavelength/blaze/apex 等，或 profile.xxx')


def _axis_label(name):
    return _AXIS_LABELS.get(name, name)


def _check_values(name, raw):
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f'{name} 必须是一维非空的有限实数数组')
    return values


def _scan_definition(mode, base):
    """返回 (轴名到数值的映射, 各计算点参数, 多维 shape)。"""
    if mode == 'single':
        return {}, [{}], (1,)
    if mode in ('angle', 'wavelength', 'blaze'):
        values = {'angle': ANGLE_VALUES, 'wavelength': WAVELENGTH_VALUES,
                  'blaze': BLAZE_VALUES}[mode]
        axis = {mode: _check_values(mode, values)}
    elif mode in ('grid', 'paired'):
        if not isinstance(SCAN_AXES, dict) or len(SCAN_AXES) < 2:
            raise ValueError('组合扫描的 SCAN_AXES 至少需要两个轴，例如 wavelength 与 angle')
        axis = {key: _check_values(key, values) for key, values in SCAN_AXES.items()}
    else:
        raise ValueError('MODE 应为 single / angle / wavelength / blaze / grid / paired')

    params = [_axis_parameter(k) for k in axis]
    if len(params) != len(set(params)):
        raise ValueError('SCAN_AXES 包含重复的物理参数别名')
    if 'profile.blaze_deg' in params and base.profile.name != 'echelle':
        raise ValueError('闪耀角扫描需要 Profile.echelle() 轮廓')
    if WAVELENGTH_INCIDENCE not in ('fixed', 'littrow'):
        raise ValueError("WAVELENGTH_INCIDENCE 应为 'fixed' 或 'littrow'")
    if 'wavelength_nm' in params and WAVELENGTH_INCIDENCE == 'littrow':
        if 'incidence.angle_deg' in params:
            raise ValueError('扫描显式入射角时不能同时逐点跟踪 Littrow：请选择 fixed')
        if SOURCE.littrow_order is None:
            raise ValueError('WAVELENGTH_INCIDENCE=littrow 要求 SOURCE=Incidence.littrow(...)')

    if mode == 'grid':
        for key,values in axis.items():
            delta = np.diff(values)
            if len(delta) and not (np.all(delta > 0) or np.all(delta < 0)):
                raise ValueError(f'grid 的 {key} 数组必须严格单调，才能正确绘制有物理坐标的热力图')
        shape = tuple(len(v) for v in axis.values())
        combos = [dict(zip(axis, values)) for values in product(*axis.values())]
    elif mode == 'paired':
        lengths = {len(v) for v in axis.values()}
        if len(lengths) != 1:
            raise ValueError('paired 扫描要求各扫描数组长度完全相同')
        shape = (next(iter(lengths)),)
        combos = [dict(zip(axis,values)) for values in zip(*axis.values())]
    else:
        shape = (len(next(iter(axis.values()))),)
        combos = [{key: float(v)} for key, arr in axis.items() for v in arr]
    if len(combos) > MAX_CALCULATIONS:
        raise ValueError(f'扫描需要 {len(combos)} 次计算，超过 MAX_CALCULATIONS={MAX_CALCULATIONS}')
    return axis, combos, shape


def _make_simulation(base, params):
    """保持物理语义：先设置波长/形状，最后设置显式入射角。"""
    param = {_axis_parameter(k): float(v) for k,v in params.items()}
    wavelength_changing = 'wavelength_nm' in param
    explicit_angle = 'incidence.angle_deg' in param
    if explicit_angle or (wavelength_changing and WAVELENGTH_INCIDENCE == 'fixed'):
        # 即使原始 SOURCE 是 Littrow，此后角度也固定为初始实验室角，除非 angle 轴再覆盖。
        base = base.with_parameter('incidence.angle_deg', base.incidence_deg)
    for key, value in param.items():
        if key != 'incidence.angle_deg':
            base = base.with_parameter(key, value)
    if explicit_angle:
        base = base.with_parameter('incidence.angle_deg', param['incidence.angle_deg'])
    return base


def _edges(x):
    """从不等距扫描的坐标中心计算热力图边界，避免将非等距间隔画成等宽。"""
    x = np.asarray(x,dtype=float)
    if len(x)==1:
        h = max(abs(x[0])*0.01, 0.1)
        return x + np.array([-h,h])
    midpoint = (x[:-1]+x[1:])/2
    return np.r_[x[0]-(midpoint[0]-x[0]), midpoint, x[-1]+(x[-1]-midpoint[-1])]


def _plot_grid(axis, efficiencies, title):
    keys = list(axis)
    if len(keys) < 2:
        raise ValueError('热力图需要两个轴')
    if PLOT_AXES is None:
        ykey, xkey = keys[:2]
    else:
        if len(PLOT_AXES)!=2 or PLOT_AXES[0] == PLOT_AXES[1] or any(k not in axis for k in PLOT_AXES):
            raise ValueError('PLOT_AXES 必须指定 SCAN_AXES 中两个不同的键')
        xkey,ykey = PLOT_AXES
    ix,iy = keys.index(xkey),keys.index(ykey)
    fix = {k: int(np.argmin(np.abs(axis[k]-float(PLOT_SLICE[k]))))
           if k in PLOT_SLICE else len(axis[k])//2
           for k in keys if k not in (xkey,ykey)}
    if any(k not in axis or k in (xkey,ykey) for k in PLOT_SLICE):
        raise ValueError('PLOT_SLICE 只能指定不在 PLOT_AXES 的扫描轴')
    info = ', '.join(f'{k}={axis[k][i]:g}' for k,i in fix.items())
    drawn = [p for p in POLARIZATIONS if p in efficiencies]
    fig,panels = plt.subplots(1,len(drawn),figsize=(6.4*len(drawn),4.6),squeeze=False)
    for pol,ax in zip(drawn,panels[0]):
        arr = efficiencies[pol]
        heat = np.empty((len(axis[ykey]),len(axis[xkey])),float)
        for iyv in range(len(axis[ykey])):
            for ixv in range(len(axis[xkey])):
                sl = [fix.get(k,0) for k in keys]
                sl[iy],sl[ix] = iyv,ixv
                heat[iyv,ixv] = arr[tuple(sl)]
        pcm = ax.pcolormesh(_edges(axis[xkey]),_edges(axis[ykey]),heat,shading='flat',cmap='viridis')
        fig.colorbar(pcm,ax=ax,label='Absolute efficiency (%)')
        ax.set(xlabel=_axis_label(xkey),ylabel=_axis_label(ykey),
               title=f'{pol} | order {TARGET_ORDER}')
    fig.suptitle(title + (f' | slice: {info}' if info else ''))
    fig.tight_layout()
    return fig, {k: axis[k][i] for k,i in fix.items()}


def order_efficiency_map(data, order, polarization='TE'):
    """任何级次的效率数组，形状与 grid/paired 扫描一致，单位 %。

    级次若在某扫描点不传播，该点效率定义为 0。
    """
    p = polarization.upper()
    values = []
    for modes,etas in zip(data['orders_each'],data['efficiencies_each']):
        if p not in etas:
            raise ValueError(f'没有计算偏振 {p!r}')
        index = np.flatnonzero(modes == int(order))
        values.append(float(etas[p][index[0]]) if len(index) else 0.0)
    return np.asarray(values).reshape(data['scan_shape'])


def run(mode=None):
    """执行计算；全部结果保留内存。mode 传值可临时覆盖顶部的 MODE。"""
    mode = MODE if mode is None else mode
    if not POLARIZATIONS or any(p not in ('TE','TM') for p in POLARIZATIONS):
        raise ValueError('POLARIZATIONS 应为 TE/TM 的非空子集')
    sim0 = Simulation(profile=PROFILE, wavelength_nm=WAVELENGTH_NM,
                      incidence=SOURCE, target_order=TARGET_ORDER, material=METAL)
    axis, cases, scan_shape = _scan_definition(mode,sim0)
    results, angles, modes_each, efficiencies_each = [],[],[],[]
    te_values,tm_values = [],[]
    if SCAN_BACKEND not in ('thread','process'):
        raise ValueError("SCAN_BACKEND 必须为 'thread' 或 'process'")
    if BLOCH_REUSE not in (True, False, 'auto', 'on', 'off'):
        raise ValueError('BLOCH_REUSE 必须为 auto/on/off 或 True/False')
    if COATED and SCAN_WORKERS > 1:
        raise ValueError('实验性涂层求解暂不支持 scan_workers>1，请改为 1')
    print(f'\nGratingBEM {mode} | {len(cases)} calculations | λ₀={WAVELENGTH_NM:g} nm | '
          f'd={PROFILE.period_um:.6f} µm | target m={TARGET_ORDER}')
    if mode=='grid':
        print('数组维度顺序：' + ' × '.join(f'{name}({len(v)})' for name,v in axis.items()))
    print(f"{'progress':>10} {'parameters':>48} {'incidence°':>11} {'TE (%)':>10} {'TM (%)':>10} {'time (s)':>9}")
    batch=None
    if not COATED and len(cases)>1:
        simulations=[_make_simulation(sim0, params) for params in cases]
        def progress(done,total,index,result,error):
            if error: return
            s=simulations[index]
            values=[]
            for pol in POLARIZATIONS:
                idx=np.flatnonzero(result.modes==TARGET_ORDER)
                eff=result.efficiencies[pol]
                values.append(f'{100*float(eff[idx[0]]) if len(idx) else 0.:.4f}%')
            print(f'完成 {done}/{total} (原索引 {index+1}) | θ={s.incidence_deg:.5f}° | '+
                  ', '.join(f'{p}={v}' for p,v in zip(POLARIZATIONS,values)),flush=True)
        batch=calculate_many(simulations, NUMERICS, polarizations=POLARIZATIONS,
                             scan_workers=SCAN_WORKERS,workers=1,progress=progress,
                             scan_backend=SCAN_BACKEND,bloch_reuse=BLOCH_REUSE)
        if batch.errors and any(e is not None for e in batch.errors):
            raise RuntimeError(f'扫描出现求解错误: {batch.errors}')
    for i,params in enumerate(cases):
        try:
            s = _make_simulation(sim0,params)
            if COATED:
                from gratingbem.coating import Film, CoatedSimulation
                from gratingbem.coating import calculate as calculate_coated
                r = calculate_coated(CoatedSimulation(s,Film(FILM_THICKNESS_NM,FILM_EPSILON)),
                                     NUMERICS,polarizations=POLARIZATIONS)
                if r.status.startswith('failed'):
                    raise RuntimeError(f'涂层物理检查失败：{r.summary()}')
                modes = np.asarray(r.modes,int)
                etas = {p:np.asarray(r.raw['polarizations'][p]['efficiencies'])*100
                        for p in POLARIZATIONS}
                elapsed = r.raw['total_seconds']
            else:
                r = (batch.results[i] if batch is not None else
                     calculate(s,NUMERICS,polarizations=POLARIZATIONS,workers=WORKERS))
                if any(not r.diagnostics[p]['passes_passivity'] for p in POLARIZATIONS):
                    raise RuntimeError(f'被动性检查失败：{r.summary()}')
                modes = np.asarray(r.modes,int)
                etas = {p:np.asarray(r.efficiencies[p])*100 for p in POLARIZATIONS}
                elapsed = r.timings['total_s']
        except Exception as exc:
            raise RuntimeError(f'组合扫描在点 {i+1}/{len(cases)} {params!r} 失败：{exc}') from exc
        vals = {}
        for pol in POLARIZATIONS:
            idx = np.flatnonzero(modes==TARGET_ORDER)
            vals[pol] = float(etas[pol][idx[0]]) if len(idx) else 0.0
        param_text = ', '.join(f'{k}={v:g}' for k,v in params.items()) or 'baseline'
        if batch is None:
            print(f'{i+1:>4}/{len(cases):<5} {param_text:>48} {s.incidence_deg:11.5f} '
                  f'{vals.get("TE",np.nan):10.4f} {vals.get("TM",np.nan):10.4f} {elapsed:9.2f}',flush=True)
        results.append(r);angles.append(s.incidence_deg)
        te_values.append(vals.get('TE',np.nan));tm_values.append(vals.get('TM',np.nan))
        modes_each.append(modes.copy());efficiencies_each.append({p:a.copy() for p,a in etas.items()})

    TE_target,TM_target = np.asarray(te_values),np.asarray(tm_values)
    shape = scan_shape if mode == 'grid' else (len(cases),)
    TE_map,TM_map = TE_target.reshape(shape),TM_target.reshape(shape)
    order_last = modes_each[-1]
    TE_last = efficiencies_each[-1].get('TE',np.full(len(order_last),np.nan))
    TM_last = efficiencies_each[-1].get('TM',np.full(len(order_last),np.nan))
    if PRINT_ALL_ORDERS:
        print('\n最后一个计算点：传播反射级次效率（%）')
        print(f"{'m':>6} {'TE':>12} {'TM':>12}")
        for j,m in enumerate(order_last):
            print(f'{int(m):6d} {TE_last[j]:12.6f} {TM_last[j]:12.6f}')
    print('最后一个计算点总反射：' + ', '.join(
        f'{p}={np.sum(efficiencies_each[-1][p]):.5f}%'
        for p in POLARIZATIONS))
    if COATED:
        print('注意：涂层通过基础检查不等于高阶深槽结果已网格收敛。')

    # 图 1：最后一个参数点的全部传播级次，不读写任何文件。
    fig_orders,ax = plt.subplots(figsize=(10,4.2))
    if 'TE' in POLARIZATIONS:ax.plot(order_last,TE_last,'.-',label='TE')
    if 'TM' in POLARIZATIONS:ax.plot(order_last,TM_last,'.-',label='TM')
    ax.axvline(TARGET_ORDER,ls='--',color='0.5',lw=1)
    ax.set(xlabel='Diffraction order',ylabel='Absolute efficiency (%)',
           title='All propagating reflected orders | last calculation point')
    ax.grid(alpha=.25);ax.legend();fig_orders.tight_layout()

    fig_scan=None;plot_slice={}
    if mode=='grid' and len(axis)>=2:
        fig_scan,plot_slice = _plot_grid(axis,{'TE':TE_map,'TM':TM_map},'Parameter grid')
    elif mode!='single':
        first_key = next(iter(axis))
        x = np.asarray(axis[first_key])
        fig_scan,ax = plt.subplots(figsize=(9,4.2))
        if 'TE' in POLARIZATIONS:ax.plot(x,TE_target,'o-',label='TE')
        if 'TM' in POLARIZATIONS:ax.plot(x,TM_target,'s-',label='TM')
        ax.set(xlabel=_axis_label(first_key),ylabel=f'Order {TARGET_ORDER} efficiency (%)',
               title=f'{mode} scan'+(' | other parameters paired pointwise' if mode=='paired' else ''))
        ax.grid(alpha=.25);ax.legend();fig_scan.tight_layout()

    data = {
        'mode':mode,'axes':axis,'scan_shape':shape,
        'cases':cases,'angles_deg':np.asarray(angles),
        'scan_values':(np.asarray(next(iter(axis.values()))) if axis else np.asarray([sim0.incidence_deg])),
        'TE_target_pct':TE_target,'TM_target_pct':TM_target,
        'TE_map':TE_map,'TM_map':TM_map,
        'orders':order_last,'TE_orders_pct':TE_last,'TM_orders_pct':TM_last,
        'orders_each':modes_each,'efficiencies_each':efficiencies_each,
        'results':results,'results_grid':np.asarray(results,dtype=object).reshape(shape),
        'batch_elapsed_s':None if batch is None else batch.elapsed_s,
        'scan_backend':SCAN_BACKEND,'bloch_reuse':BLOCH_REUSE,
        'last_result':results[-1],'figures':(fig_orders,fig_scan),'plot_slice':plot_slice,
    }
    print('\n全部结果仍在 data 中，未默认保存 CSV/PNG/JSON。')
    if SHOW_PLOTS:
        if plt.get_backend().lower()=='agg':
            print('当前后端为 Agg，无法弹出图窗；请在桌面 Python 或 Jupyter 中运行。')
        else:
            plt.show()
    return data


if __name__ == '__main__':
    data = run()
