"""This file define function to convert SymPy graph to and from Theano.

Currently the translation dictionnary is incomplete. So if you miss
something, don't hesitate to look for them. There is a high change
that you can find the translation easily. See the "theano_to_sympy" and
"sympy_to_theano" function.

This file also define a Theano optimization that try to conver the
complete Theano graph to SymPy and it to optimize the graph. You need
to import this file and use the Theano flag
"optimizer_including=sympy" to enable this optimization. It support
theano function with only 1 output.

The conversion currently support only theano.tensor.scalar with sympy.Symbol.

"""


import sympy
import theano
import theano.tensor as tt

class IncompatibleGraph(Exception):
    """
    This exception is raised when we can't make the conversion
    between Theano and SymPy graph.
    """

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
        raise IncompatibleGraph("...")


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


class SymPyOptimizer(theano.gof.opt.Optimizer):
    """Graph optimizer that use SymPy for its optimization"""
    def __init__(self):
        theano.gof.opt.Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(theano.gof.toolbox.ReplaceValidate())
        fgraph.attach_feature(theano.gof.DestroyHandler())

    def apply(self, fgraph):
        if len(fgraph.outputs) > 1:
            return
        out = fgraph.outputs[0]
        try:
            s_graph = theano_to_sympy(out)
            m = shape_and_dtype_map(out)
            s_graph2 = sympy.simplify(s_graph)
            t_graph = sympy_to_theano(s_graph2, m)
            t_graph = out.type.filter_variable(t_graph)
        except IncompatibleGraph:
            return
        try:
            fgraph.replace_all_validate(
                [(out, t_graph)],
                reason=self.__class__.__name__,
            )
        except Exception, e:
            raise
        return
        # Bellow is some test that would allow to use a function that
        # split the graph in subgraph to optimize each subgraph
        # individually. This could be useful as not all Theano graph
        # can be represented in SymPy graph.
        for subgraph in get_subgraph(fgraph):
            s_graph = theano_to_sympy(subgraph)
            m = shape_and_dtype_map(subgraph)
            s_graph2 = sympy.simplify(s_graph)
            t_graph = sympy_to_theano(s_graph2, m)

            try:
                fgraph.replace_all_validate_remove(
                    zip(subgraph.outputs, t_graph.outputs),
                    subgraph.list(),
                    reason='GemmOptimizer',
                    warn=True,
                )
                nb_replacement += 1
            except InconsistencyError, e:
                pass

#SymPy prefer int to float. If some constant are float instead of int
#some SymPy optimization don't get applied. So this optimization must run
#before upcast of elemwise's inputs in the canonicalizer.
theano.compile.mode.optdb.register('SymPyOptimizer', SymPyOptimizer(),
                                   0.2, 'sympy')
