import random
import json
import math
from sympy import symbols, Function, Eq, Derivative, S, simplify
from sympy.printing.latex import LatexPrinter

x, t, y, z = symbols('x t y z')   # z added for 3D directional shuffling
u = Function('u')

# SymPy prints 0.45 as 0.450000000000000. This overrides that.
class _Printer(LatexPrinter):
    def _print_Float(self, expr):
        return str(float(expr))

def sym_latex(expr):
    return _Printer().doprint(expr)

def _fmt_num(expr):
    return str(int(expr)) if expr.is_Integer else str(float(expr))


# ============================================================
# DIALECT CONVERTERS
# Walk the SymPy expression tree and produce prefix/postfix strings.
# These handle individual terms — the top-level equation joining
# is handled separately by build_dialect() to allow shuffled ordering.
# ============================================================

def _derivative_to_str(expr, mode):
    # args[0] = function, args[1:] = (variable, order) specs
    r = _to_dialect(expr.args[0], mode)
    for var_spec in expr.args[1:]:
        try:
            var, order = var_spec[0], int(var_spec[1])
        except (TypeError, IndexError):
            var, order = var_spec, 1
        for _ in range(order):
            r = f'd({r}, {var})' if mode == 'prefix' else f'{r} {var} d'
    return r

def _to_dialect(expr, mode):
    from sympy import Eq, Add, Mul, Derivative, Pow
    if expr.is_Number:   return _fmt_num(expr)
    if expr.is_Symbol:   return str(expr)
    if expr.is_Function: return str(expr)
    if expr.func == Eq:
        l, r = _to_dialect(expr.lhs, mode), _to_dialect(expr.rhs, mode)
        return f'=({l}, {r})' if mode == 'prefix' else f'{l} {r} ='
    if expr.func == Add:
        args = [_to_dialect(a, mode) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f'+({result}, {a})' if mode == 'prefix' else f'{result} {a} +'
        return result
    if expr.func == Mul:
        args = [_to_dialect(a, mode) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = f'*({result}, {a})' if mode == 'prefix' else f'{result} {a} *'
        return result
    if expr.func == Pow:
        b, e = _to_dialect(expr.args[0], mode), _to_dialect(expr.args[1], mode)
        return f'^({b}, {e})' if mode == 'prefix' else f'{b} {e} ^'
    if expr.func == Derivative:
        return _derivative_to_str(expr, mode)
    return str(expr)

def to_prefix(eq):  return _to_dialect(eq, 'prefix')
def to_postfix(eq): return _to_dialect(eq, 'postfix')


# ============================================================
# SPATIAL DIRECTION UTILITIES
#
# Directional shuffling: instead of always using x, each instance
# randomly picks WHICH spatial variable(s) the PDE acts on.
#
# This forces the model to learn the ABSTRACT PATTERN
# (e.g. "first-order time + second-order spatial") rather than
# memorising "Heat always has d(d(...,x),x)".
# ============================================================

# All non-empty subsets of {x, y, z}
ALL_SPATIAL_SUBSETS = [
    ['x'], ['y'], ['z'],
    ['x', 'y'], ['x', 'z'], ['y', 'z'],
    ['x', 'y', 'z'],
]

# Laplace always needs exactly 2 directions (cross-term u_st requires a pair)
LAPLACE_SPATIAL_PAIRS = [['x', 'y'], ['x', 'z'], ['y', 'z']]

# Burgers stays 1D in one direction (2D Burgers is a vector PDE — different structure)
BURGERS_SPATIAL_SINGLES = [['x'], ['y'], ['z']]

def format_dirs(svars_names):
    """['x'] → 'x',  ['x','y'] → 'x and y',  ['x','y','z'] → 'x, y, and z'"""
    if len(svars_names) == 1: return svars_names[0]
    if len(svars_names) == 2: return f"{svars_names[0]} and {svars_names[1]}"
    return f"{svars_names[0]}, {svars_names[1]}, and {svars_names[2]}"


# ============================================================
# POSITIONAL SHUFFLING UTILITIES
#
# Positional shuffling: randomly rearrange where terms sit in
# the equation.  Two types combined in one operation:
#   1. Cross-side movement: terms move across the = sign,
#      flipping their sign when they do.
#   2. Within-side reordering: terms on each side appear in
#      a random order.
#
# SymPy canonicalises Add expressions internally, so we CANNOT
# get different orderings by writing a+b vs b+a in Python.
# Solution: keep terms as a Python list, shuffle the list,
# and build the dialect string directly from that list —
# bypassing SymPy's ordering entirely.
#
# Correctness invariant:
#   sum(lhs_terms) - sum(rhs_terms) == original lhs - original rhs
# i.e. every rearrangement is algebraically equivalent.
# ============================================================

def positional_shuffle(eq):
    """
    Returns (lhs_terms, rhs_terms) as randomly shuffled Python lists.

    Steps:
      1. total = lhs - rhs  (everything moved to one side, = 0)
      2. Extract additive terms of total via as_ordered_terms()
      3. Shuffle the list randomly (within-side reordering)
      4. Pick a random split point n (cross-side movement):
           lhs_terms = terms[:n]         (kept as-is)
           rhs_terms = [-t for t in terms[n:]]  (negated — crossed the = sign)
    """
    total = eq.lhs - eq.rhs
    terms = [t for t in total.as_ordered_terms() if not t.is_zero]

    if len(terms) <= 1:
        return ([eq.lhs], [eq.rhs])

    random.shuffle(terms)                              # within-side reorder
    n_left = random.randint(0, len(terms))             # cross-side split

    lhs_terms = terms[:n_left]                         # go to LHS unchanged
    rhs_terms = [-t for t in terms[n_left:]]           # negated when moved to RHS

    return lhs_terms, rhs_terms

def _terms_to_str(terms, mode):
    """
    Build a dialect string from a list of SymPy terms IN THAT ORDER.
    This is the key function that bypasses SymPy's canonical Add ordering.
    Joins with left-associative + chain.
    """
    non_zero = [t for t in terms if not t.is_zero]
    if not non_zero:
        return '0'

    if mode == 'latex':
        # Emit +/- between terms with proper signs
        parts = [sym_latex(non_zero[0])]
        for term in non_zero[1:]:
            if term.could_extract_minus_sign():
                parts.append(f'- {sym_latex(-term)}')
            else:
                parts.append(f'+ {sym_latex(term)}')
        return ' '.join(parts)

    elif mode == 'prefix':
        # Left-associative: +(+(a, b), c)
        result = _to_dialect(non_zero[0], 'prefix')
        for term in non_zero[1:]:
            result = f'+({result}, {_to_dialect(term, "prefix")})'
        return result

    else:  # postfix
        # Left-associative: a b + c +
        result = _to_dialect(non_zero[0], 'postfix')
        for term in non_zero[1:]:
            result = f'{result} {_to_dialect(term, "postfix")} +'
        return result

def build_dialect(lhs_terms, rhs_terms, mode):
    """Assemble a full equation string from shuffled term lists."""
    lhs = _terms_to_str(lhs_terms, mode)
    rhs = _terms_to_str(rhs_terms, mode)
    if mode == 'prefix':  return f'=({lhs}, {rhs})'
    if mode == 'postfix': return f'{lhs} {rhs} ='
    return f'{lhs} = {rhs}'   # latex


# ============================================================
# BURGERS:  u_t + u·u_s = ν·u_ss   (s = single spatial direction)
#
# Single direction only — 2D Burgers is a vector equation with
# a completely different structure.
#
# Regime based on ν (viscosity):
#   shock     (ν < 0.2):  nonlinear term dominates → tanh, polynomial
#   diffusion (ν > 0.6):  diffusion term dominates → exp, polynomial
#   mixed     (0.2–0.6):  both effects             → tanh, exp, polynomial
# ============================================================

def sample_burgers():
    svar = random.choice(BURGERS_SPATIAL_SINGLES)[0]
    return {'nu': round(random.uniform(0.05, 1.5), 2), 'svar': svar, 'dirs': svar}

def burgers_regime(nu):
    if nu < 0.2:   return 'shock'
    elif nu > 0.6: return 'diffusion'
    else:          return 'mixed'

def burgers_equation(nu, svar):
    s = symbols(svar)
    uf = u(t, s)
    return Eq(Derivative(uf, t) + uf * Derivative(uf, s), nu * Derivative(uf, s, 2))

BURGERS_NL = [
    "The time derivative of u plus u times its spatial derivative in {svar} equals {nu} times the second derivative in {svar}.",
    "u self-advects in {svar}: temporal change plus u times its {svar}-gradient equals {nu} times its second {svar}-derivative.",
    "u_t plus the nonlinear term u times u_{svar} equals {nu} times u_{svar}{svar}.",
    "The first-order temporal derivative plus u times the first-order {svar}-derivative equals {nu} times the second-order {svar}-derivative.",
    "Nonlinear transport and diffusion in {svar}: time derivative plus u times its {svar}-gradient balances {nu} times its second {svar}-derivative.",
    "The rate of change of u in time plus u multiplied by its {svar}-rate of change equals {nu} times the second {svar}-derivative.",
]

BURGERS_OPERATORS = {
    'shock':     ['tanh', 'polynomial'],
    'diffusion': ['exp', 'polynomial'],
    'mixed':     ['tanh', 'exp', 'polynomial'],
}

BURGERS_REASONING = {
    'shock': [
        "ν={nu} is below 0.2 in the {svar}-direction; nonlinear advection dominates, forming shocks — tanh captures the steep front.",
        "Low viscosity ν={nu}: diffusion is too weak to prevent shock formation in {svar}; the nonlinear term controls the structure.",
        "With ν={nu} in the inviscid regime along {svar}, steep gradients form; tanh basis functions are needed.",
    ],
    'diffusion': [
        "ν={nu} exceeds 0.6 in {svar}; diffusion dominates and the solution decays smoothly — exponential basis functions apply.",
        "High viscosity ν={nu} along {svar} overwhelms nonlinear advection; u_{svar}{svar} controls behavior.",
        "At ν={nu}, diffusion in {svar} suppresses shock formation entirely; exponentially smooth profile.",
    ],
    'mixed': [
        "ν={nu} in the transitional range along {svar}; both nonlinear advection and diffusion are active — tanh and exp both relevant.",
        "Intermediate viscosity ν={nu} in {svar}: neither shock formation nor pure diffusion dominates.",
        "With ν={nu} between regimes in {svar}, nonlinear and diffusive terms are comparable in magnitude.",
    ],
}

def generate_burgers():
    coeffs  = sample_burgers()
    nu, svar = coeffs['nu'], coeffs['svar']
    regime  = burgers_regime(nu)
    eq      = burgers_equation(nu, svar)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = BURGERS_OPERATORS[regime]
    rsn = random.choice(BURGERS_REASONING[regime]).format(**coeffs)
    return {
        'family': 'Burgers', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': random.choice(BURGERS_NL).format(**coeffs),
        },
        'target': f"family: Burgers | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# WAVE:  u_tt = c²·∑(u_ss)  for s in spatial_vars
#
# c  = wave speed (physical quantity, used in NL)
# c_sq = c² = actual PDE coefficient (appears in all dialects)
#
# Directional: any subset of {x, y, z} — the wave propagates
# in those directions.
#
# Regime based on wave speed c:
#   slow (c < 1.0):  long-wavelength → sin, cos, polynomial
#   fast (c ≥ 1.0):  short-wavelength with exponential envelope → sin, cos, exp
# ============================================================

def sample_wave():
    svars = random.choice(ALL_SPATIAL_SUBSETS)
    c = round(random.uniform(0.1, 3.0), 2)
    return {'c': c, 'c_sq': round(c**2, 2), 'svars': svars, 'dirs': format_dirs(svars)}

def wave_regime(c):
    return 'slow' if c < 1.0 else 'fast'

def wave_equation(c_sq, svars):
    syms = [symbols(s) for s in svars]
    uf   = u(t, *syms)
    return Eq(Derivative(uf, t, 2), c_sq * sum(Derivative(uf, s, 2) for s in syms))

WAVE_NL = [
    "The second time derivative of u equals {c_sq} times the second spatial derivative in {dirs}.",
    "u oscillates: its second-order temporal change equals {c_sq} times its second-order spatial change in {dirs}.",
    "The acceleration of u in time equals {c_sq} times its spatial curvature in {dirs}.",
    "u propagates at speed {c} in {dirs}: second temporal derivative equals {c_sq} times second spatial derivative.",
    "Second-order temporal change equals {c_sq} times second-order spatial change in {dirs}.",
    "The curvature of u in time is proportional to {c_sq} times its curvature in {dirs}.",
]

WAVE_OPERATORS = {
    'slow': ['sin', 'cos', 'polynomial'],
    'fast': ['sin', 'cos', 'exp'],
}

WAVE_REASONING = {
    'slow': [
        "Wave speed c={c} in {dirs} is below 1.0; long-wavelength oscillations — sin, cos, and polynomial apply.",
        "Low wave speed c={c} in {dirs}: long-period oscillations; sinusoidal and polynomial basis functions dominate.",
        "c={c} gives a slow wave in {dirs}; long wavelength means polynomial terms contribute alongside sin and cos.",
    ],
    'fast': [
        "Wave speed c={c} in {dirs} exceeds 1.0; short-wavelength oscillations with exponential amplitude variation.",
        "High wave speed c={c} in {dirs}: rapid oscillations with exponential envelope — sin, cos, exp all relevant.",
        "c={c} gives a fast wave in {dirs}; short wavelength and rapid oscillation require exp alongside sin and cos.",
    ],
}

def generate_wave():
    coeffs = sample_wave()
    c, svars = coeffs['c'], coeffs['svars']
    regime  = wave_regime(c)
    eq      = wave_equation(coeffs['c_sq'], svars)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = WAVE_OPERATORS[regime]
    rsn = random.choice(WAVE_REASONING[regime]).format(**coeffs)
    return {
        'family': 'Wave', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': random.choice(WAVE_NL).format(**coeffs),
        },
        'target': f"family: Wave | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# LAPLACE:  A·u_s1s1 + B·u_s1s2 + C·u_s2s2 = 0
#
# Always 2 spatial directions (a pair from {x,y}, {x,z}, {y,z}).
# The cross-term B·u_s1s2 only makes sense between two directions.
#
# Elliptic guarantee: B² < 4AC.  We cap B at 90% of max.
#
# Regime based on ratio = B²/(4AC):
#   isotropic (< 0.1):  B ≈ 0, separable  → sin, cos, polynomial
#   mixed     (0.1–0.6): moderate coupling → sin, cos, exp, polynomial
#   skewed    (> 0.6):  heavy coupling     → exp, polynomial
# ============================================================

def sample_laplace():
    s1, s2  = random.choice(LAPLACE_SPATIAL_PAIRS)
    A = round(random.uniform(0.1, 2.0), 2)
    C = round(random.uniform(0.1, 2.0), 2)
    b_limit = round(0.9 * 2 * (A * C) ** 0.5, 2)
    B = round(random.uniform(-b_limit, b_limit), 2)
    return {'A': A, 'B': B, 'C': C, 's1': s1, 's2': s2, 'dirs': f"{s1} and {s2}"}

def laplace_regime(A, B, C):
    ratio = (B ** 2) / (4 * A * C)
    if ratio < 0.1:   return 'isotropic'
    elif ratio > 0.6: return 'skewed'
    else:             return 'mixed'

def laplace_equation(A, B, C, s1, s2):
    sym1, sym2 = symbols(s1), symbols(s2)
    uf = u(sym1, sym2)   # no time — steady state
    return Eq(
        A * Derivative(uf, sym1, 2)
        + B * Derivative(uf, sym1, sym2)
        + C * Derivative(uf, sym2, 2),
        0
    )

LAPLACE_NL = [
    "{A} times the second {s1}-derivative plus {B} times the mixed {s1}-{s2} derivative plus {C} times the second {s2}-derivative equals zero.",
    "Steady state in {dirs}: {A} times u_{s1}{s1} plus {B} times u_{s1}{s2} plus {C} times u_{s2}{s2} equals zero.",
    "u has zero weighted curvature in {dirs}: {A} times {s1}-curvature plus {B} times diagonal {s1}-{s2} curvature plus {C} times {s2}-curvature is zero.",
    "{A} times the second spatial derivative in {s1}, plus {B} times the cross derivative in {s1} and {s2}, plus {C} times the second derivative in {s2}, equals zero.",
    "No time evolution in {dirs}: the weighted sum {A}·u_{s1}{s1} plus {B}·u_{s1}{s2} plus {C}·u_{s2}{s2} vanishes.",
    "In the {s1}-{s2} plane: {A} times second {s1}-derivative plus {B} times mixed derivative plus {C} times second {s2}-derivative is zero.",
]

LAPLACE_OPERATORS = {
    'isotropic': ['sin', 'cos', 'polynomial'],
    'mixed':     ['sin', 'cos', 'exp', 'polynomial'],
    'skewed':    ['exp', 'polynomial'],
}

LAPLACE_REASONING = {
    'isotropic': [
        "B={B} is near zero in the {s1}-{s2} plane; equation separates cleanly — sin, cos, and polynomial apply.",
        "With B={B} close to zero, {s1} and {s2} decouple; trig and polynomial basis functions capture the solution.",
        "Nearly isotropic in {dirs} (B≈0): separable structure means sin, cos, and polynomial are sufficient.",
    ],
    'mixed': [
        "Moderate cross-coupling B={B} in {dirs}; both sinusoidal and exponential basis functions are active.",
        "B²/(4AC) in the intermediate range in {dirs}: anisotropy present but not dominant — full basis needed.",
        "Mixed elliptic regime in the {s1}-{s2} plane; diagonal coupling significant but well within elliptic territory.",
    ],
    'skewed': [
        "Large B={B} in the {s1}-{s2} plane; strong diagonal coupling dominates — exp and polynomial apply.",
        "B²/(4AC) exceeds 0.6 in {dirs}; strongly skewed, near parabolic boundary — exp and polynomial relevant.",
        "Heavy diagonal coupling (B={B}) in {dirs}: exponential basis functions are most relevant.",
    ],
}

def generate_laplace():
    coeffs = sample_laplace()
    A, B, C = coeffs['A'], coeffs['B'], coeffs['C']
    s1, s2  = coeffs['s1'], coeffs['s2']
    regime  = laplace_regime(A, B, C)
    eq      = laplace_equation(A, B, C, s1, s2)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = LAPLACE_OPERATORS[regime]
    rsn = random.choice(LAPLACE_REASONING[regime]).format(**coeffs)
    return {
        'family': 'Laplace', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': random.choice(LAPLACE_NL).format(**coeffs),
        },
        'target': f"family: Laplace | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# KLEIN-GORDON:  u_tt = c²·∑(u_ss) - m²·u
#
# Wave equation plus a mass term -m²·u.
# The mass term has NO derivatives — it is the ONLY term in the
# dataset with a zeroth-order (no-derivative) coefficient on u.
# That single extra term is what distinguishes it from Wave.
#
# c  = wave speed, c_sq = c²  (same as Wave)
# m  = mass,       m_sq = m²  (zeroth-order coefficient)
#
# Regime based on mass m:
#   low_mass  (m < 0.5):  wave-like   → sin, cos, polynomial
#   mixed     (0.5–1.5):  both effects → sin, cos, exp, polynomial
#   high_mass (m > 1.5):  exponential envelope → sin, cos, exp
# ============================================================

def sample_klein_gordon():
    svars = random.choice(ALL_SPATIAL_SUBSETS)
    c = round(random.uniform(0.1, 3.0), 2)
    m = round(random.uniform(0.1, 2.5), 2)
    return {
        'c': c, 'c_sq': round(c**2, 2),
        'm': m, 'm_sq': round(m**2, 2),
        'svars': svars, 'dirs': format_dirs(svars),
    }

def klein_gordon_regime(m):
    if m < 0.5:   return 'low_mass'
    elif m > 1.5: return 'high_mass'
    else:         return 'mixed'

def klein_gordon_equation(c_sq, m_sq, svars):
    syms = [symbols(s) for s in svars]
    uf   = u(t, *syms)
    return Eq(
        Derivative(uf, t, 2),
        c_sq * sum(Derivative(uf, s, 2) for s in syms) - m_sq * uf
    )

KLEIN_GORDON_NL = [
    "The second time derivative of u equals {c_sq} times the second spatial derivative in {dirs} minus {m_sq} times u itself.",
    "u oscillates with a restoring force in {dirs}: second temporal derivative equals {c_sq} times spatial curvature minus {m_sq} times u.",
    "The acceleration of u in time equals {c_sq} times spatial curvature in {dirs}, reduced by a mass term {m_sq} times u.",
    "u_tt equals {c_sq} times the second derivative in {dirs} minus {m_sq} times u — wave speed {c}, mass {m}.",
    "Second-order temporal change equals {c_sq} times second-order spatial change in {dirs} minus a zeroth-order term {m_sq} times u.",
    "Wave propagation at speed {c} in {dirs} with restoring force: u_tt = {c_sq} times spatial curvature minus {m_sq} times u.",
]

KLEIN_GORDON_OPERATORS = {
    'low_mass':  ['sin', 'cos', 'polynomial'],
    'mixed':     ['sin', 'cos', 'exp', 'polynomial'],
    'high_mass': ['sin', 'cos', 'exp'],
}

KLEIN_GORDON_REASONING = {
    'low_mass': [
        "Mass m={m} is below 0.5 in {dirs}; mass term is weak and equation behaves like Wave — trig and polynomial dominate.",
        "Small mass m={m}: the -m²·u term is a minor perturbation; sinusoidal and polynomial basis functions capture the structure.",
        "With m={m} near zero in {dirs}, Klein-Gordon reduces toward Wave behavior — weak restoring force.",
    ],
    'mixed': [
        "Mass m={m} in transitional range in {dirs}; wave propagation and mass term both contribute — full basis applies.",
        "Intermediate mass m={m} in {dirs}: neither pure wave nor strong exponential damping dominates.",
        "With m={m} between regimes in {dirs}, the restoring force modifies but doesn't dominate propagation.",
    ],
    'high_mass': [
        "Mass m={m} exceeds 1.5 in {dirs}; strong restoring force creates exponential envelope — sin, cos, exp needed.",
        "Large mass m={m}: the -m²·u term dominates in {dirs}, producing oscillations with exponential amplitude.",
        "With m={m} in the high-mass regime in {dirs}, solutions are sinusoidal oscillations on an exponential envelope.",
    ],
}

def generate_klein_gordon():
    coeffs = sample_klein_gordon()
    m, svars = coeffs['m'], coeffs['svars']
    regime   = klein_gordon_regime(m)
    eq       = klein_gordon_equation(coeffs['c_sq'], coeffs['m_sq'], svars)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = KLEIN_GORDON_OPERATORS[regime]
    rsn = random.choice(KLEIN_GORDON_REASONING[regime]).format(**coeffs)
    return {
        'family': 'KleinGordon', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': random.choice(KLEIN_GORDON_NL).format(**coeffs),
        },
        'target': f"family: KleinGordon | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# HEAT:  u_t = α·∑(u_ss)  for s in spatial_vars
#
# First-order in time, second-order in space.
# KEY STRUCTURAL IDENTITY vs other families:
#   Heat:  single d on time side  (u_t)   + double d on space side (u_ss)
#   Wave:  double d on time side  (u_tt)  + double d on space side
#   KG:    double d on time side  (u_tt)  + double d on space side + bare u
#   Burger: single d on time side + single d nonlinear + double d on space
#   Adv:   single d on time side  + single d on space (NO double d anywhere)
#
# Regime based on α (diffusivity):
#   slow   (α < 0.5):  many Fourier modes persist → sin, cos, exp, polynomial
#   medium (0.5–1.5):  moderate filtering          → sin, cos, exp
#   fast   (α > 1.5):  rapid Gaussian smoothing    → exp, polynomial
# ============================================================

def sample_heat():
    svars = random.choice(ALL_SPATIAL_SUBSETS)
    return {'alpha': round(random.uniform(0.05, 2.5), 2), 'svars': svars, 'dirs': format_dirs(svars)}

def heat_regime(alpha):
    if alpha < 0.5:   return 'slow'
    elif alpha > 1.5: return 'fast'
    else:             return 'medium'

def heat_equation(alpha, svars):
    syms = [symbols(s) for s in svars]
    uf   = u(t, *syms)
    return Eq(Derivative(uf, t), alpha * sum(Derivative(uf, s, 2) for s in syms))

HEAT_NL = [
    "The time derivative of u equals {alpha} times the second spatial derivative in {dirs}.",
    "u diffuses in {dirs}: its rate of change in time equals {alpha} times its spatial curvature in {dirs}.",
    "The first-order temporal change of u equals {alpha} times the second-order spatial change in {dirs}.",
    "u evolves in time at a rate proportional to {alpha} times its curvature in {dirs}.",
    "The partial derivative of u with respect to time equals {alpha} times the second partial derivative in {dirs}.",
    "u spreads over time in {dirs}: temporal derivative equals {alpha} times second spatial derivative.",
]

HEAT_OPERATORS = {
    'slow':   ['sin', 'cos', 'exp', 'polynomial'],
    'medium': ['sin', 'cos', 'exp'],
    'fast':   ['exp', 'polynomial'],
}

HEAT_REASONING = {
    'slow': [
        "α={alpha} is below 0.5 in {dirs}; slow diffusion — many Fourier modes persist and sin, cos, exp, polynomial all contribute.",
        "Low diffusivity α={alpha} in {dirs}: sharp spatial features survive long; full range of basis functions needed.",
        "Slow diffusion in {dirs} (α={alpha}): high-frequency modes decay slowly; sinusoidal, exponential, polynomial all relevant.",
    ],
    'medium': [
        "α={alpha} is in the moderate range in {dirs}; intermediate Fourier filtering — sin, cos, and exp capture the structure.",
        "Moderate diffusivity α={alpha} in {dirs}: intermediate modes survive; sinusoidal and exponential functions dominate.",
        "With α={alpha} in {dirs}, diffusion is neither slow nor rapid; sin, cos, exp basis functions apply.",
    ],
    'fast': [
        "α={alpha} exceeds 1.5 in {dirs}; rapid diffusion quickly smooths to a Gaussian — exp and polynomial apply.",
        "High diffusivity α={alpha} in {dirs}: high-frequency modes decay almost immediately; only exp and polynomial remain.",
        "Fast diffusion in {dirs} (α={alpha}): solution Gaussianifies quickly; exponential and polynomial capture the decay.",
    ],
}

def generate_heat():
    coeffs = sample_heat()
    alpha, svars = coeffs['alpha'], coeffs['svars']
    regime = heat_regime(alpha)
    eq     = heat_equation(alpha, svars)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = HEAT_OPERATORS[regime]
    rsn = random.choice(HEAT_REASONING[regime]).format(**coeffs)
    return {
        'family': 'Heat', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': random.choice(HEAT_NL).format(**coeffs),
        },
        'target': f"family: Heat | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# ADVECTION:  u_t + ∑(c_s · u_s) = 0  for s in spatial_vars
#
# Each spatial direction has its OWN independent speed coefficient c_s.
# This is the ONLY family where a multi-direction equation has
# DIFFERENT coefficients for each direction.
#
# KEY STRUCTURAL IDENTITY:
#   ZERO second-order (double-nested) derivatives anywhere.
#   Every other family has at least one u_ss term.
#   Advection has only first-order time + first-order spatial.
#   Also: RHS is always 0 in canonical form.
#
# NL is built dynamically (number of transport terms varies).
#
# Regime based on L2 speed magnitude:
#   slow (magnitude < 1.0):  smooth periodic waves → sin, cos, polynomial
#   fast (magnitude ≥ 1.0):  sharp fronts          → exp, polynomial
# ============================================================

def sample_advection():
    svars = random.choice(ALL_SPATIAL_SUBSETS)
    coeffs = {f'c_{s}': round(random.uniform(0.1, 3.0), 2) for s in svars}
    coeffs['svars'] = svars
    coeffs['dirs']  = format_dirs(svars)
    return coeffs

def advection_regime(coeffs, svars):
    speed = math.sqrt(sum(coeffs[f'c_{s}']**2 for s in svars))
    return 'slow' if speed < 1.0 else 'fast'

def advection_equation(coeffs, svars):
    syms = [symbols(s) for s in svars]
    uf   = u(t, *syms)
    transport = sum(coeffs[f'c_{s}'] * Derivative(uf, sym) for s, sym in zip(svars, syms))
    return Eq(Derivative(uf, t) + transport, 0)

def advection_nl(coeffs, svars):
    """Build NL dynamically since number of transport terms varies."""
    term_descs = [f"{coeffs[f'c_{s}']} times the {s}-derivative of u" for s in svars]
    if len(term_descs) == 1:
        transport_str = term_descs[0]
    elif len(term_descs) == 2:
        transport_str = f"{term_descs[0]} and {term_descs[1]}"
    else:
        transport_str = ', '.join(term_descs[:-1]) + f', and {term_descs[-1]}'
    templates = [
        f"The time derivative of u plus {transport_str} equals zero.",
        f"u is transported: u_t plus {transport_str} equals zero.",
        f"Pure transport: the sum of the time derivative and {transport_str} vanishes.",
        f"The first-order temporal change plus {transport_str} equals zero.",
        f"u moves without changing shape: u_t plus {transport_str} is zero.",
        f"No diffusion: the time derivative of u plus {transport_str} equals zero.",
    ]
    return random.choice(templates)

ADVECTION_OPERATORS = {
    'slow': ['sin', 'cos', 'polynomial'],
    'fast': ['exp', 'polynomial'],
}

ADVECTION_REASONING = {
    'slow': [
        "Transport speed below 1.0 in {dirs}; slow periodic waves — sin, cos, and polynomial apply.",
        "Low advection speed in {dirs}: smooth periodic transport; sinusoidal and polynomial basis functions dominate.",
        "Slow transport in {dirs}: long-wavelength waves move without deformation — trig and polynomial apply.",
    ],
    'fast': [
        "Transport speed exceeds 1.0 in {dirs}; rapid transport produces sharp wave fronts — exp and polynomial apply.",
        "High advection speed in {dirs}: fast-moving sharp fronts; exponential and polynomial basis functions apply.",
        "Fast transport in {dirs}: short-wavelength sharp features develop — exp and polynomial are relevant.",
    ],
}

def generate_advection():
    coeffs = sample_advection()
    svars  = coeffs['svars']
    regime = advection_regime(coeffs, svars)
    eq     = advection_equation(coeffs, svars)
    lhs_t, rhs_t = positional_shuffle(eq)
    ops = ADVECTION_OPERATORS[regime]
    rsn = random.choice(ADVECTION_REASONING[regime]).format(**coeffs)
    return {
        'family': 'Advection', 'coefficients': coeffs, 'regime': regime,
        'dialects': {
            'latex':   build_dialect(lhs_t, rhs_t, 'latex'),
            'prefix':  build_dialect(lhs_t, rhs_t, 'prefix'),
            'postfix': build_dialect(lhs_t, rhs_t, 'postfix'),
            'natural': advection_nl(coeffs, svars),
        },
        'target': f"family: Advection | operators: {', '.join(ops)} | reasoning: {rsn}",
    }


# ============================================================
# DATASET GENERATION
# ============================================================

GENERATORS = {
    'Burgers':     generate_burgers,
    'Wave':        generate_wave,
    'Laplace':     generate_laplace,
    'KleinGordon': generate_klein_gordon,
    'Heat':        generate_heat,
    'Advection':   generate_advection,
}

INSTANCES_PER_FAMILY = 2000

def generate_dataset(output_path='dataset.jsonl', seed=42):
    random.seed(seed)
    instances = []
    for family, gen_fn in GENERATORS.items():
        for _ in range(INSTANCES_PER_FAMILY):
            instances.append(gen_fn())
    random.shuffle(instances)
    with open(output_path, 'w') as f:
        for inst in instances:
            f.write(json.dumps(inst) + '\n')
    print(f"Saved {len(instances)} instances to {output_path}")
    print(f"\n{'Family':<14} {'latex':>8} {'prefix':>8} {'postfix':>8} {'natural':>8}")
    print('-' * 52)
    for family in GENERATORS:
        subset = [d for d in instances if d['family'] == family]
        print(
            f"{family:<14}"
            f"{len(set(d['dialects']['latex']   for d in subset)):>8}"
            f"{len(set(d['dialects']['prefix']  for d in subset)):>8}"
            f"{len(set(d['dialects']['postfix'] for d in subset)):>8}"
            f"{len(set(d['dialects']['natural'] for d in subset)):>8}"
        )


# ============================================================
# SANITY CHECKS
# ============================================================

def _shuffle_is_valid(eq):
    """Verify positional_shuffle preserves the equation (LHS-RHS unchanged)."""
    lhs_t, rhs_t = positional_shuffle(eq)
    new_lhs = sum(lhs_t) if lhs_t else S.Zero
    new_rhs = sum(rhs_t) if rhs_t else S.Zero
    return simplify((new_lhs - new_rhs) - (eq.lhs - eq.rhs)) == 0

def _shuffle_produces_variety(eq, n=60):
    """Verify shuffling generates more than one distinct prefix string."""
    results = set()
    for _ in range(n):
        lhs_t, rhs_t = positional_shuffle(eq)
        results.add(build_dialect(lhs_t, rhs_t, 'prefix'))
    return len(results) > 1

def verify():
    # --- Positional shuffle correctness ---
    test_eqs = [
        burgers_equation(0.3, 'x'),
        wave_equation(2.25, ['x']),
        wave_equation(2.25, ['x', 'y']),
        heat_equation(0.5, ['x', 'y', 'z']),
        klein_gordon_equation(2.25, 0.5, ['x', 'y']),
        advection_equation({'c_x': 0.5, 'c_y': 1.2, 'svars': ['x','y'], 'dirs': 'x and y'}, ['x', 'y']),
        laplace_equation(1.0, 0.5, 1.0, 'x', 'y'),
    ]
    for eq in test_eqs:
        for _ in range(10):
            assert _shuffle_is_valid(eq), f"Shuffle changed equation: {eq}"
    print("check 1 passed: positional_shuffle is mathematically valid for all families")

    # --- Shuffling produces variety ---
    for eq in test_eqs:
        assert _shuffle_produces_variety(eq), f"Shuffling produces no variety for: {eq}"
    print("check 2 passed: positional_shuffle produces varied outputs")

    # --- Burgers: single direction, double-nested u_ss in canonical prefix ---
    for svar in ['x', 'y', 'z']:
        p = to_prefix(burgers_equation(0.3, svar))
        assert f'd(d(u(t, {svar}), {svar}), {svar})' in p, f"Burgers u_{svar}{svar} not double-nested"
    print("check 3 passed: Burgers u_ss double-nested for all directions")

    # --- Wave: u_tt and all u_ss double-nested ---
    for svars in [['x'], ['y', 'z'], ['x', 'y', 'z']]:
        p = to_prefix(wave_equation(2.25, svars))
        assert 'd(d(u(t, ' in p and p.count('d(d(') == 1 + len(svars), \
            f"Wave prefix wrong nesting for {svars}"
    print("check 4 passed: Wave u_tt and u_ss all double-nested")

    # --- Heat: single d in time, double d in space ---
    for svars in [['y'], ['x', 'z']]:
        p = to_prefix(heat_equation(0.5, svars))
        # u_t must appear as single d
        for s in svars:
            assert f'd(d(u(t, ' in p, f"Heat missing double-nested u_{s}{s}"
        assert 'd(d(u(t, x), t)' not in p and 'd(d(u(t, y), t)' not in p, \
            "Heat has second-order time derivative — wrong"
    print("check 5 passed: Heat has single time derivative, double spatial derivatives")

    # --- Advection: ZERO double-nested derivatives in canonical prefix ---
    for svars in [['x'], ['y', 'z'], ['x', 'y', 'z']]:
        coeffs_a = {f'c_{s}': 1.0 for s in svars}
        coeffs_a.update({'svars': svars, 'dirs': format_dirs(svars)})
        p = to_prefix(advection_equation(coeffs_a, svars))
        assert p.count('d(d(') == 0, f"Advection has double-nested derivative — wrong: {p}"
    print("check 6 passed: Advection has zero double-nested derivatives for all directions")

    # --- Laplace: all 3 pairs produce valid elliptic equations ---
    for s1, s2 in [['x','y'], ['x','z'], ['y','z']]:
        eq = laplace_equation(1.0, 0.5, 1.0, s1, s2)
        p  = to_prefix(eq)
        assert f'd(d(u({s1}, {s2}), {s1}), {s2})' in p, \
            f"Laplace cross-term not in prefix for ({s1},{s2})"
    print("check 7 passed: Laplace cross-term correct for all pairs")

    # --- Laplace: B² < 4AC always holds ---
    for _ in range(200):
        c = sample_laplace()
        assert c['B']**2 < 4 * c['A'] * c['C'], f"Laplace not elliptic: {c}"
    print("check 8 passed: Laplace always elliptic (B²<4AC)")

    # --- All regime operator sets are distinct within each family ---
    assert len({str(v) for v in BURGERS_OPERATORS.values()}) == 3
    assert len({str(v) for v in WAVE_OPERATORS.values()}) == 2
    assert len({str(v) for v in LAPLACE_OPERATORS.values()}) == 3
    assert len({str(v) for v in KLEIN_GORDON_OPERATORS.values()}) == 3
    assert len({str(v) for v in HEAT_OPERATORS.values()}) == 3
    assert len({str(v) for v in ADVECTION_OPERATORS.values()}) == 2
    print("check 9 passed: all regime operator sets are distinct within each family")

    # --- Target format correct for all families ---
    random.seed(99)
    for gen_fn in GENERATORS.values():
        inst = gen_fn()
        assert inst['target'].startswith(f"family: {inst['family']} |")
        assert 'operators:' in inst['target']
        assert 'reasoning:' in inst['target']
    print("check 10 passed: target format correct for all families\n")


if __name__ == '__main__':
    verify()

    print("=== Sample instances ===\n")
    random.seed(0)
    for gen_fn in GENERATORS.values():
        inst = gen_fn()
        print(f"family:  {inst['family']}   regime={inst['regime']}   coeffs={inst['coefficients']}")
        print(f"latex:   {inst['dialects']['latex']}")
        print(f"prefix:  {inst['dialects']['prefix']}")
        print(f"postfix: {inst['dialects']['postfix']}")
        print(f"natural: {inst['dialects']['natural']}")
        print(f"target:  {inst['target']}")
        print()

    print("=== Generating dataset ===\n")
    generate_dataset()
