"""Validation shared by the public model and the legacy problem adapter."""
from numbers import Real


def integer_order(value, *, name='order', nonzero=False):
    """Normalize exact integral values without silently truncating diffraction orders."""
    if isinstance(value,bool) or not isinstance(value,(Real,str)):
        raise ValueError(f'{name} must be an integer')
    try:
        parsed=int(value)
    except (ValueError,OverflowError) as exc:
        raise ValueError(f'{name} must be a finite integer') from exc
    if not isinstance(value,str) and value!=parsed:
        raise ValueError(f'{name} must be an integer')
    if nonzero and parsed==0:
        raise ValueError(f'{name} must be a nonzero integer')
    return parsed
