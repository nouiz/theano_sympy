import sympy
import theano
import theano.tensor as tt
mapping = {sympy.Add: theano.tensor.add,
           sympy.Mul: tt.mul,
#           sympy.Sub: tt.sub,
#           sympy.Mul(x, Pow(y, -1)): tt.true_div,  # int_div,
           #sympy.Mod: tt.mod, # master branch
           #clip,  # second,
           #identity,
           #cast,
           sympy.Abs: tt.abs_,
           sympy.sign: tt.sgn,
           sympy.ceiling: tt.ceil,
           sympy.floor: tt.floor,
#           round_half_to_even, round_half_away_from_zero,
           lambda x: sympy.Mul(x, -1): tt.neg,
           lambda x: sympy.Pow(x, -1): tt.inv,
           sympy.log: tt.log,
           lambda x: log(1 + p): tt.log1p,
           #log2, log10,
           sympy.exp: tt.exp,
           #exp2,
           lambda x: sympy.Mul(x, x): tt.sqr,
           sympy.sqrt: tt.sqrt,
           sympy.cos: tt.cos,
           sympy.acos: tt.arccos,
           sympy.sin: tt.sin,
           sympy.asin: tt.arcsin,
           sympy.tan: tt.tan,
           sympy.atan: tt.arctan,
           sympy.atan2: tt.arctan2,
           sympy.cosh: tt.cosh,
           sympy.acosh: tt.arccosh,
           sympy.sinh: tt.sinh,
           sympy.asinh: tt.arcsinh,
           sympy.tanh: tt.tanh,
           sympy.atanh: tt.arctanh,
#           real, imag,
           sympy.arg: tt.angle,
           lambda x, y: sympy.Add(x, sympy.Mul(y, sympy.I)): tt.complex,
#           sympy.conjugate: tt.conj,
#           complexfrompolar,
#           composite???
           sympy.erf: tt.erf,
           lambda x: 1.0 - sympy.erf(x): tt.erfc,
#           sympy.gamma: tt.gamma,
           lambda x: sympy.ln(sympy.Abs(sympy.gamma(x))): tt.gammaln,
#           psi,
           sympy.Pow: tt.pow,
           sympy.Eq: tt.eq,
           sympy.Gt: tt.gt,
           sympy.Lt: tt.lt,
           sympy.Le: tt.le,
           sympy.Ge: tt.ge,
#           tt.neq
#           isnan,
           lambda x: sympy.Eq(sympy.S.Infinity, sympy.Abs(x)): tt.isinf,
#           sympy.: tt.inrange,  #need to create an sympy graph.
#           switch,
#           or_, xor, and_, invert, Sympy don't have bit wise operation
           sympy.Max: tt.maximum,  # Sympy accept >2 inputs, Theano only 2
           sympy.Min: tt.minimum,  # Sympy accept >2 inputs, Theano only 2
}# Implement factorial, gamma in Theano

rev_mapping = dict([(v, k) for k, v in mapping.iteritems()])


def var_string(var):
    if var.name:
        return var.name
    else:
        return "theano_var_%d" % id(var)

def theano_to_sympy(g, inputs=None):
    if inputs is None:
        inputs = theano.gof.graph.inputs([g])
    return theano_to_sympy_impl(g, inputs)


def theano_to_sympy_impl(g, inputs):
    """ g is a theano graph"""
    assert isinstance(g, (tt.TensorVariable,
                          tt.TensorConstant)), type(g)
    if isinstance(g, tt.TensorConstant):
        assert g.ndim == 0
        # The sum transform the 0d ndarray to a numpy scalar
        # sympy don't accept ndarray
        return g.data.sum()
    elif g in inputs:
        return sympy.Symbol(var_string(g))
    elif g.owner.op in rev_mapping:
        return rev_mapping[g.owner.op](*[theano_to_sympy_impl(var, inputs)
                                         for var in g.owner.inputs])
    else:
        raise Exception("...")

def shape_and_dtype_map(g):
    return {var.name: (var.dtype, var.broadcastable)
               for var in theano.gof.graph.inputs([g])
               if isinstance(var, tt.TensorVariable)}


#In sympy 2 var with the same name are the same variable
def sympy_to_theano(g, var_map, inputs_map={}):
    assert isinstance(g, sympy.Expr)
    if isinstance(g, sympy.Symbol):
        if g.name in inputs_map:
            return inputs_map[g.name]
        else:
            dtype, broadcastable = var_map[g.name]
            return tt.TensorType(dtype, broadcastable)(g.name)
    elif isinstance(g, sympy.Number):
        return eval(str(g))
    else:
        return mapping[g.__class__](*[sympy_to_theano(arg, var_map, inputs_map)
                                            for arg in g.args])
