"""Shared empirical convergence metrics (not rigorous a posteriori bounds)."""
import numpy as np

def _get_metrics(result,previous,polarizations):
    values={};failures=[];passivity=True
    for pol in polarizations:
        cur=result.diagnostics[pol];eta=cur['target_efficiency']
        if eta is None:failures.append(f'{pol}: target order not propagating')
        if cur['reflection_sum']>1+1e-8:
            # Final passivity tolerance is checked separately.
            pass
        values[pol+'_target_pct']=None if eta is None else 100*eta
        values[pol+'_reflection_pct']=100*cur['reflection_sum']
        values[pol+'_residual']=cur['matrix_residual']
        if previous is None:continue
        prv=previous.diagnostics[pol];old=prv['target_efficiency']
        if eta is not None and old is not None:
            values[pol+'_target_delta_pp']=100*abs(eta-old)
        values[pol+'_reflected_delta_pp']=100*abs(cur['reflection_sum']-prv['reflection_sum'])
        om=result.modes;pm=previous.modes
        union=np.union1d(om,pm)
        a=np.zeros(len(union));b=np.zeros(len(union))
        a[np.searchsorted(union,om)]=result.efficiencies[pol]
        b[np.searchsorted(union,pm)]=previous.efficiencies[pol]
        values[pol+'_all_orders_l1_pp']=100*float(np.sum(np.abs(a-b)))
    return values,failures


def _meets(runs,steps,settings,pols):
    if len(runs)<settings.min_levels or len(runs)<3:
        return False # require TWO successive differences; min_levels=2 never enough
    for row in steps[-2:]:
        for pol in pols:
            if (row.get(pol+'_target_delta_pp') is None or
                row[pol+'_target_delta_pp'] > 100*settings.target_abs_tol or
                row[pol+'_reflected_delta_pp']>100*settings.reflection_abs_tol):return False
            if (settings.order_l1_tol is not None and
                row[pol+'_all_orders_l1_pp']>100*settings.order_l1_tol):return False
    for pol in pols:
        x=runs[-1].diagnostics[pol]
        if x['reflection_sum']>1+settings.passivity_tolerance:return False
        if x['matrix_residual']>settings.residual_tolerance:return False
        if x['target_efficiency'] is None:return False
    return True
