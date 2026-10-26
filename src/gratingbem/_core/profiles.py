"""v13 periodic grating profiles: exact straight/arc segments, rounded presets.

Solver geometry is not replaced by a polygon when it contains arcs. A polygonal
vertex approximation is provided only for legacy plotting or geometry export.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
from html import escape
import math
import numpy as np
from gratingbem._core.polyline_profiles import Profile as _V12Profile
from .segments import Line,Arc,fillet_polyline,arc_through_radius,from_dict as segment_from_dict


def _r(x):return float(x)


@dataclass(frozen=True)
class Profile(_V12Profile):
    segments:tuple[Line|Arc,...]|None=None

    def __post_init__(self):
        super().__post_init__()
        if self.segments is None:return
        seg=tuple(self.segments)
        if not seg or not all(isinstance(s,(Line,Arc)) for s in seg):
            raise ValueError('segments must be a nonempty collection of Line/Arc')
        tol=max(1e-10,1e-9*self.period_um)
        for left,right in zip(seg[:-1],seg[1:]):
            if np.linalg.norm(np.array(left.end)-right.start)>tol:
                raise ValueError('path segments are not connected')
        if np.linalg.norm(np.array(seg[0].start)-self.vertices[0])>tol or np.linalg.norm(np.array(seg[-1].end)-self.vertices[-1])>tol:
            raise ValueError('segments/vertices endpoints mismatch')
        for s in seg:
            if s.end[0]<s.start[0]-tol:
                raise ValueError('profile segments must be x-monotone (no overhang)')
        object.__setattr__(self,'segments',seg)

    @classmethod
    def from_segments(cls,period_um,segments,name='path',recipe=None):
        segments=tuple(segments)
        if not segments:raise ValueError('empty path')
        vertices=[segments[0].start]+[s.end for s in segments]
        return cls(period_um,tuple(vertices),name,recipe,segments)

    @property
    def has_arcs(self):return self.segments is not None and any(isinstance(s,Arc) for s in self.segments)

    @property
    def geometric_segments(self):
        if self.segments is not None:return self.segments
        return tuple(Line(a,b) for a,b in zip(self.vertices[:-1],self.vertices[1:]))

    @property
    def lengths_um(self):return np.asarray([s.length for s in self.geometric_segments],float)

    def as_dict(self):
        d=super().as_dict()
        if self.segments is not None:d['segments']=[s.as_dict() for s in self.segments]
        return d

    def svg(self,width=760,height=260):
        parts=[]
        for seg in self.geometric_segments:
            n=(max(8,math.ceil(abs(seg.sweep)*28)) if isinstance(seg,Arc) else 2)
            xyz=seg.eval(np.linspace(-1,1,n))
            if parts:xyz=xyz[1:]
            parts.extend(map(tuple,xyz))
        v=np.asarray(parts,float)
        zlo=float(v[:,1].min());zhi=float(v[:,1].max())
        sc=min((width-80)/self.period_um,(height-80)/max(zhi-zlo,1e-12))
        xx=width/2+(v[:,0]-self.period_um/2)*sc
        yy=height/2-(v[:,1]-(zlo+zhi)/2)*sc
        points=' '.join(('M' if i==0 else 'L')+f'{x:.4f},{y:.4f}' for i,(x,y) in enumerate(zip(xx,yy)))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
                f'<rect width="100%" height="100%" fill="#edf5fa"/>'
                f'<path d="{points} L{xx[-1]:.4f},{height+20} L{xx[0]:.4f},{height+20} Z" '
                f'fill="#becfd9" stroke="#1c6587" stroke-width="1.6"/>'
                f'<text x="16" y="20" font-size="13" fill="#1d4663">'
                f'{escape(self.name)} | period={self.period_um:.5g} µm</text></svg>')

    def save_svg(self,path):
        p=Path(path);p.write_text(self.svg(),encoding='utf8');return p

    def rounded(self,radii_um:float|Sequence[float],name=None,recipe=None):
        """Apply exact tangent circular fillets at all vertices incl. periodic seam.

        radii_um can be a single radius or per-vertex radii in original order.
        Each radius is geometric, not a small-segment approximation.
        """
        if self.has_arcs:raise ValueError('round() needs a piecewise-linear input; already has arcs')
        if self.segments is not None and any(not isinstance(s,Line) for s in self.segments):
            raise ValueError('cannot fillet curved path twice')
        r=np.asarray(radii_um,float)
        if r.ndim==0:r=np.full(len(self.vertices),float(r))
        if len(r)!=len(self.vertices):raise ValueError('radii_um must contain one radius per vertex')
        if np.all(r<1e-15):
            return Profile(self.period_um,self.vertices,name or self.name,recipe or self.recipe)
        seg=fillet_polyline(self.vertices,r)
        if recipe is None and self.recipe is None:
            recipe={'period_um':self.period_um,
                    'vertices_um':[list(x) for x in self.vertices],
                    'corner_radii_um':list(map(float,r))}
            if name is None:name='rounded_custom'
        return Profile.from_segments(self.period_um,seg,name or self.name,recipe)

    def with_parameter(self,key,value):
        if self.recipe is None:raise ValueError('Custom path scan: supply Profile values or an update callback')
        d=dict(self.recipe)
        if key not in d:raise ValueError(f'Unknown parameter {key}; available {list(d)}')
        d[key]=value
        return from_dict(dict(shape=self.name,**d))

    @classmethod
    def echelle(cls,period_um,blaze_deg=76.156,apex_deg=90.,reversed=False,
                tip_radius_um=0.,valley_radius_um=0.):
        orig=super().echelle(period_um,blaze_deg,apex_deg,reversed=reversed)
        recipe=dict(orig.recipe,tip_radius_um=float(tip_radius_um),valley_radius_um=float(valley_radius_um))
        # v12 tip always at index 1 even in the reversed orientation.
        r=[valley_radius_um,tip_radius_um,valley_radius_um]
        return orig.rounded(r,name='echelle',recipe=recipe) if any(r) else cls(orig.period_um,orig.vertices,'echelle',recipe)

    @classmethod
    def rectangle(cls,period_um,height_um,duty_cycle=.5,center=.5,kind='ridge',
                  corner_radius_um=0.,corner_radii_um=None):
        orig=super().rectangle(period_um,height_um,duty_cycle,center,kind)
        rad=(corner_radius_um if corner_radii_um is None else corner_radii_um)
        recipe=dict(orig.recipe,corner_radius_um=float(corner_radius_um),
                    corner_radii_um=(None if corner_radii_um is None else list(map(float,corner_radii_um))))
        return orig.rounded(rad,name='rectangle',recipe=recipe) if np.any(np.asarray(rad)!=0) else cls(orig.period_um,orig.vertices,'rectangle',recipe)

    @classmethod
    def trapezoid(cls,period_um,height_um,bottom_duty=.6,top_duty=.4,center=.5,kind='ridge',
                  corner_radius_um=0.,corner_radii_um=None):
        orig=super().trapezoid(period_um,height_um,bottom_duty,top_duty,center,kind)
        rad=(corner_radius_um if corner_radii_um is None else corner_radii_um)
        recipe=dict(orig.recipe,corner_radius_um=float(corner_radius_um),
                    corner_radii_um=(None if corner_radii_um is None else list(map(float,corner_radii_um))))
        return orig.rounded(rad,name='trapezoid',recipe=recipe) if np.any(np.asarray(rad)!=0) else cls(orig.period_um,orig.vertices,'trapezoid',recipe)

    @classmethod
    def steps(cls,period_um,heights_um,widths_fraction=None,corner_radius_um=0.,corner_radii_um=None):
        orig=super().steps(period_um,heights_um,widths_fraction)
        rad=(corner_radius_um if corner_radii_um is None else corner_radii_um)
        recipe=dict(orig.recipe,corner_radius_um=float(corner_radius_um),
                    corner_radii_um=(None if corner_radii_um is None else list(map(float,corner_radii_um))))
        return orig.rounded(rad,name='steps',recipe=recipe) if np.any(np.asarray(rad)!=0) else cls(orig.period_um,orig.vertices,'steps',recipe)

    @classmethod
    def staircase(cls,period_um,height_um,steps=4,duty_cycle=.7,center=.5,direction='rising',
                  corner_radius_um=0.,corner_radii_um=None):
        orig=super().staircase(period_um,height_um,steps,duty_cycle,center,direction)
        rad=(corner_radius_um if corner_radii_um is None else corner_radii_um)
        recipe=dict(orig.recipe,corner_radius_um=float(corner_radius_um),
                    corner_radii_um=(None if corner_radii_um is None else list(map(float,corner_radii_um))))
        return orig.rounded(rad,name='staircase',recipe=recipe) if np.any(np.asarray(rad)!=0) else cls(orig.period_um,orig.vertices,'staircase',recipe)

    @classmethod
    def path(cls,period_um,start_z_um=0.):return PathBuilder(period_um,start_z_um)


class PathBuilder:
    """Fluent exact path: line_to; arc_to using radius/direction; build.

    arc_to picks a circular arc with specified radius and sweep orientation.
    Tangency to surrounding lines is not implicit; use .rounded(...) for that.
    """
    def __init__(self,period_um,start_z_um=0.):
        self.period_um=float(period_um)
        self.pos=(0.,float(start_z_um));self.start=self.pos;self.segments=[]

    def line_to(self,x,z):
        seg=Line(self.pos,(float(x),float(z)))
        self.segments.append(seg);self.pos=seg.end;return self

    def arc_to(self,x,z,radius=None,turn='cw',large_arc=False,*,radius_um=None):
        rr=(radius if radius_um is None else radius_um)
        if rr is None:raise ValueError('arc_to requires radius or radius_um')
        if radius is not None and radius_um is not None:raise ValueError('supply one radius')
        seg=arc_through_radius(self.pos,(float(x),float(z)),float(rr),turn,large_arc)
        self.segments.append(seg);self.pos=seg.end;return self

    def arc_center_to(self,x,z,center,turn='cw'):
        a=np.asarray(self.pos);b=np.array([x,z],float);cc=np.asarray(center,float)
        radius=float(np.linalg.norm(a-cc))
        if abs(np.linalg.norm(b-cc)-radius)>1e-8*max(1,radius):
            raise ValueError('start/end points must have same distance to arc center')
        t0=float(np.arctan2(a[1]-cc[1],a[0]-cc[0]))
        t1=float(np.arctan2(b[1]-cc[1],b[0]-cc[0]))
        if turn=='ccw':sw=(t1-t0)%(2*np.pi)
        elif turn=='cw':sw=-((t0-t1)%(2*np.pi))
        else:raise ValueError('turn cw/ccw')
        seg=Arc(tuple(cc),radius,t0,sw)
        self.segments.append(seg);self.pos=seg.end;return self

    def build(self,name='path'):
        return Profile.from_segments(self.period_um,self.segments,name=name)


def from_dict(obj):
    d=dict(obj)
    if 'recipe' in d and isinstance(d['recipe'],dict):
        recipe=dict(d['recipe']);recipe['shape']=d.get('shape',d.get('name',None))
        if recipe['shape'] is not None:
            return from_dict(recipe)
    kind=d.pop('shape',d.pop('name',None))
    if 'segments' in d:
        return Profile.from_segments(d['period_um'],[segment_from_dict(s) for s in d['segments']],name=kind)
    if kind=='rounded_custom':
        base=Profile.custom(d['period_um'],d['vertices_um'])
        return base.rounded(d['corner_radii_um'],name='rounded_custom',recipe={
            'period_um':d['period_um'],'vertices_um':d['vertices_um'],
            'corner_radii_um':d['corner_radii_um']})
    if kind=='custom':return Profile.custom(d['period_um'],d['vertices_um'])
    lookup={'flat':Profile.flat,'rectangle':Profile.rectangle,
            'trapezoid':Profile.trapezoid,'steps':Profile.steps,
            'staircase':Profile.staircase,'echelle':Profile.echelle}
    if kind not in lookup:raise ValueError(f'unknown shape {kind!r}')
    return lookup[kind](**d)
