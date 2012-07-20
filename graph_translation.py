import sympy
import theano
import theano.tensor as tt
mapping = {sympy.Add: theano.tensor.add,
           sympy.Mul: theano.tensor.mul,
           sympy.Pow: theano.tensor.pow,
           sympy.sin: theano.tensor.sin,
           sympy.cos: theano.tensor.cos}

rev_mapping = dict([(v, k) for k, v in mapping.iteritems()])

def theano_to_sympy(g):
    """ g is a theano graph"""
    assert isinstance(g, (tt.TensorVariable,
                          tt.TensorConstant)), type(g)
    if isinstance(g, tt.TensorConstant):
        assert g.ndim == 0
        # The sum transform the 0d ndarray to a numpy scalar
        # sympy don't accept ndarray
        return g.data.sum()
    elif not g.owner:
        assert g.name
        assert g.ndim == 0
        return sympy.Symbol(g.name)
    #elif g.owner.op == theano.tensor.add:
    #    return sympy.Add(*map(transform, g.owner.inputs))
    elif g.owner.op in rev_mapping:
        return rev_mapping[g.owner.op](*map(theano_to_sympy, g.owner.inputs))

def shape_and_dtype_map(g):
    return {var.name : (var.dtype, var.broadcastable)
               for var in theano.gof.graph.inputs([g])
               if isinstance(var, tt.TensorVariable)}

#In sympy 2 var with the same name are the same variable
def sympy_to_theano(g, var_map):
    assert isinstance(g, sympy.Expr)
    if isinstance(g, sympy.Symbol):
        dtype, broadcastable = var_map[g.name]
        return tt.TensorType(dtype, broadcastable)(g.name)
    elif isinstance(g, sympy.Number):
        return eval(str(g))
    else:
        return mapping[g.__class__](*[sympy_to_theano(arg, var_map)
                                            for arg in g.args])
