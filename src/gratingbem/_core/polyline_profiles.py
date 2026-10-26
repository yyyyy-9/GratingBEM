"""Periodic piecewise-linear profiles (micrometre units, metal below interface)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

@dataclass(frozen=True)
class Profile:
    period_um: float
    vertices: tuple[tuple[float,float],...]
    name: str = 'custom'
    recipe: dict | None = None

    def __post_init__(self):
        d=float(self.period_um);v=np.asarray(self.vertices,dtype=float)
        if not np.isfinite(d) or d<=0:raise ValueError('period_um must be positive')
        if v.ndim!=2 or v.shape[1]!=2 or len(v)<2 or not np.isfinite(v).all():
            raise ValueError('vertices must be an Nx2 finite array')
        tol=d*1e-10
        if abs(v[0,0])>tol or abs(v[-1,0]-d)>tol or abs(v[0,1]-v[-1,1])>tol:
            raise ValueError('vertices must span [0,period] with equal periodic end heights')
        if np.any(np.diff(v[:,0])<-tol):
            raise ValueError('overhangs (decreasing x) unsupported; vertical walls allowed')
        if np.any(np.linalg.norm(np.diff(v,axis=0),axis=1)<max(1e-14,d*1e-13)):
            raise ValueError('zero-length profile edge')
        object.__setattr__(self,'period_um',d)
        object.__setattr__(self,'vertices',tuple(tuple(map(float,t)) for t in v))
        if self.recipe is not None:object.__setattr__(self,'recipe',dict(self.recipe))

    @property
    def lengths_um(self):
        return np.linalg.norm(np.diff(np.asarray(self.vertices),axis=0),axis=1)

    def as_dict(self):
        return dict(name=self.name,period_um=self.period_um,
                    vertices_um=[list(t) for t in self.vertices],recipe=self.recipe)

    def with_parameter(self,key,value):
        if self.recipe is None:raise ValueError('Custom profiles: scan Profile objects or use a callback')
        d=dict(self.recipe)
        if key not in d:raise ValueError(f'Unknown parameter {key}; available: {list(d)}')
        d[key]=value
        return from_dict(dict(shape=self.name,**d))

    def svg(self,width=760,height=260):
        """Self-contained profile preview (equal physical x/z scale)."""
        from html import escape
        v=np.asarray(self.vertices,float)
        zlo=float(min(v[:,1]));zhi=float(max(v[:,1]))
        scale=min((width-80)/self.period_um,(height-80)/max(zhi-zlo,1e-12))
        mid=(zlo+zhi)/2
        xx=width/2+(v[:,0]-self.period_um/2)*scale
        yy=height/2-(v[:,1]-mid)*scale
        points=' '.join((('M' if i==0 else 'L')+f'{x:.3f},{y:.3f}')
                        for i,(x,y) in enumerate(zip(xx,yy)))
        bot=height+20
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                f'width="{width}" height="{height}">'
                f'<rect width="100%" height="100%" fill="#eef6fc"/>'
                f'<path d="{points} L{xx[-1]:.3f},{bot} L{xx[0]:.3f},{bot} Z" '
                f'fill="#c7d5db" stroke="#275d80" stroke-width="2"/>'
                f'<text x="18" y="22" font-size="14" fill="#173d54">'
                f'{escape(self.name)} | d={self.period_um:.6g} µm</text></svg>')

    def save_svg(self,path):
        from pathlib import Path
        Path(path).write_text(self.svg(),encoding='utf8')
        return Path(path)

    @classmethod
    def custom(cls,period_um,vertices,name='custom'):
        return cls(period_um,tuple(tuple(v) for v in vertices),name)

    @classmethod
    def from_function(cls,period_um,fn:Callable,samples=32):
        if samples<2:raise ValueError('samples>=2 required')
        x=np.linspace(0,period_um,int(samples)+1);z=np.asarray(fn(x),float)
        if z.shape!=x.shape:raise ValueError('fn must return an array of heights')
        return cls.custom(period_um,np.column_stack((x,z)),name='sampled_function')

    @classmethod
    def flat(cls,period_um,z_um=0.):
        return cls(period_um,((0,z_um),(period_um,z_um)),'flat',dict(period_um=period_um,z_um=z_um))

    @classmethod
    def rectangle(cls,period_um,height_um,duty_cycle=.5,center=.5,kind='ridge'):
        _duty(duty_cycle,center)
        if height_um<=0 or kind not in ('ridge','groove'):raise ValueError('height>0, kind ridge/groove')
        a=(center-duty_cycle/2)*period_um;b=(center+duty_cycle/2)*period_um
        if a<=0 or b>=period_um:raise ValueError('need nonzero baseline margins')
        h=height_um*(1 if kind=='ridge' else -1)
        return cls(period_um,((0,0),(a,0),(a,h),(b,h),(b,0),(period_um,0)),
                   'rectangle',dict(period_um=period_um,height_um=height_um,
                                    duty_cycle=duty_cycle,center=center,kind=kind))

    @classmethod
    def trapezoid(cls,period_um,height_um,bottom_duty=.6,top_duty=.4,center=.5,kind='ridge'):
        _duty(bottom_duty,center)
        if not (0<top_duty<=bottom_duty and height_um>0 and kind in ('ridge','groove')):
            raise ValueError('require 0<top_duty<=bottom_duty, height>0, kind ridge/groove')
        bl=(center-bottom_duty/2)*period_um;br=(center+bottom_duty/2)*period_um
        tl=(center-top_duty/2)*period_um;tr=(center+top_duty/2)*period_um
        if bl<=0 or br>=period_um:raise ValueError('need nonzero baseline margins')
        h=height_um*(1 if kind=='ridge' else -1)
        v=((0,0),(bl,0),(tl,h),(tr,h),(br,0),(period_um,0))
        return cls(period_um,v,'trapezoid',dict(period_um=period_um,height_um=height_um,
                   bottom_duty=bottom_duty,top_duty=top_duty,center=center,kind=kind))

    @classmethod
    def steps(cls,period_um,heights_um:Sequence[float],widths_fraction=None):
        hs=np.asarray(heights_um,float)
        ws=np.full(len(hs),1/len(hs)) if widths_fraction is None else np.asarray(widths_fraction,float)
        if len(hs)<2 or ws.shape!=hs.shape or not np.isfinite(hs).all() or not np.isfinite(ws).all():
            raise ValueError('need matching finite height and width arrays, at least 2 each')
        if np.any(ws<=0) or not np.isclose(ws.sum(),1,atol=1e-12) or not np.isclose(hs[0],hs[-1]):
            raise ValueError('positive widths must sum to 1; first/last step heights must match')
        xx=np.r_[0,np.cumsum(ws)*period_um];xx[-1]=period_um
        v=[(0.,float(hs[0]))]
        for j,h in enumerate(hs):
            v.append((xx[j+1],float(h)))
            if j<len(hs)-1 and not np.isclose(hs[j+1],h):v.append((xx[j+1],float(hs[j+1])))
        return cls(period_um,tuple(v),'steps',dict(period_um=period_um,
                   heights_um=list(map(float,hs)),widths_fraction=list(map(float,ws))))

    @classmethod
    def staircase(cls,period_um,height_um,steps=4,duty_cycle=.7,center=.5,direction='rising'):
        if not isinstance(steps,int) or steps<2:raise ValueError('steps must be integer>=2')
        _duty(duty_cycle,center)
        if height_um<=0 or direction not in ('rising','falling'):raise ValueError('height>0; rising/falling')
        ml=center-duty_cycle/2;mr=1-center-duty_cycle/2
        if ml<=0 or mr<=0:raise ValueError('need nonzero baseline margins')
        h=np.linspace(height_um/steps,height_um,steps)
        if direction=='falling':h=h[::-1]
        profile=cls.steps(period_um,[0.,*h,0.],[ml,*([duty_cycle/steps]*steps),mr])
        return cls(period_um,profile.vertices,'staircase',dict(period_um=period_um,
            height_um=height_um,steps=steps,duty_cycle=duty_cycle,center=center,direction=direction))

    @classmethod
    def echelle(cls,period_um,blaze_deg=76.156,apex_deg=90.,reversed=False):
        if not (0<blaze_deg<90 and 0<apex_deg<180-blaze_deg):raise ValueError('invalid angles')
        a=np.deg2rad(blaze_deg);b=np.deg2rad(180-blaze_deg-apex_deg)
        w=period_um*np.tan(b)/(np.tan(a)+np.tan(b));h=w*np.tan(a)
        v=np.array([[0.,0.],[w,h],[period_um,0.]])
        if reversed:v=np.column_stack((period_um-v[::-1,0],v[::-1,1]))
        return cls(period_um,tuple(map(tuple,v)),'echelle',dict(period_um=period_um,
                   blaze_deg=blaze_deg,apex_deg=apex_deg,reversed=reversed))


def _duty(d,c):
    if not 0<d<1 or not 0<c<1 or c-d/2<0 or c+d/2>1:raise ValueError('bad duty_cycle/center')


def from_dict(obj):
    d=dict(obj);kind=d.pop('shape',d.pop('name',None))
    if kind=='custom':return Profile.custom(d['period_um'],d['vertices_um'])
    f=dict(flat=Profile.flat,rectangle=Profile.rectangle,trapezoid=Profile.trapezoid,
           steps=Profile.steps,staircase=Profile.staircase,echelle=Profile.echelle)
    if kind not in f:raise ValueError(f'unknown shape {kind!r}; use {list(f)} or custom')
    return f[kind](**d)
