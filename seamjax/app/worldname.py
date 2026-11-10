"""Plain-language reading of a world's name (shown in the debug overlay).
'4.8.10' lists the regular polygons that meet at every corner (a square, an octagon and a decagon); 5 squares is '4.4.4.4.4'.
Their corner angles add up to exactly 360 degrees on a flat floor, to more on a hyperbolic one (space curves away, rooms multiply fast) and to
less on a sphere (the world closes up on itself).  'grid:WxH:x:y' is a W by H block of square rooms whose left/right (x) and top/bottom (y)
edges are a wall (none), wrap around (plain) or wrap around mirrored (flip)."""
from collections import Counter

POLY = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon', 9: 'nonagon', 10: 'decagon', 12: 'dodecagon'}
SUM = {'flat': 'add up to exactly 360 deg', 'hyper': 'add up to more than 360 deg, so space curves away', 'sphere': 'add up to less than 360 deg, so the world closes up on itself'}
EDGE = {'none': 'a wall', 'plain': 'wraps around', 'flip': 'wraps around mirrored'}


def explain(name, cat):
    if name.startswith('grid:'):
        _, size, wx, wy = name.split(':'); w, h = size.split('x')
        return 'wrapped world: %s x %s block of square rooms; left/right edge %s, top/bottom edge %s' % (w, h, EDGE[wx], EDGE[wy])
    cnt = Counter(int(x) for x in name.split('.')); parts = [('%d %s%s' % (c, POLY.get(n, '%d-gon' % n), 's' if c > 1 else '')) if c > 1 else ('an ' if n == 8 else 'a ')+POLY.get(n, '%d-gon' % n)
                                                             for n, c in sorted(cnt.items(), key=lambda kv: name.split('.').index(str(kv[0])))]
    what = parts[0] if len(parts) == 1 else ', '.join(parts[:-1])+' and '+parts[-1]
    return '%s: %s meet at every corner; their angles %s' % (name, what, SUM[cat])
