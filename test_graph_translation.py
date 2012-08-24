from graph_translation import *

# SETUP
wt = theano.tensor.scalar('w')
xt = theano.tensor.scalar('x')
yt = theano.tensor.scalar('y')
zt = theano.tensor.add(xt, yt)

xs = sympy.Symbol('x')
ys = sympy.Symbol('y')
zs = sympy.Add(xs, ys)


def test_theano_to_sympy():

    assert theano_to_sympy(xt) == xs
    assert theano_to_sympy(zt) == zs
    assert theano_to_sympy(tt.cos(xt)**2 +    tt.sin(xt)**2) == \
                        sympy.cos(xs)**2 + sympy.sin(xs)**2

    var_map = {xs.name: ('float32', (False, False)),
               ys.name: ('float32', (False, False)),
           }


def test_shape_and_dtype_map():
    assert shape_and_dtype_map(xt) == {'x': ('float64', ())}
    assert shape_and_dtype_map(zt) == {'x': ('float64', ()),
                                       'y': ('float64', ())}
    assert shape_and_dtype_map(tt.matrix('name')) == \
            {'name': ('float64', (False, False))}


def test_sympy_to_theano():
    m = shape_and_dtype_map(zt)
    def theano_eq(a,b):
        return a.type == b.type and a.name == b.name

    assert theano_eq(sympy_to_theano(xs, m), xt)
    assert theano_eq(sympy_to_theano(zs, m), zt)


def test_sin_cos():
    zt = tt.cos(xt) ** 2 + tt.sin(xt) ** 2
    zs = theano_to_sympy(zt)
    m = shape_and_dtype_map(zt)
    assert sympy_to_theano(sympy.simplify(zs), m) == 1


def test_gammaln():
    zt = tt.gammaln(xt)
    zs = theano_to_sympy(zt)
    m = shape_and_dtype_map(zt)
    assert str(zs) == "log(Abs(gamma(x)))", zs
    assert str(sympy.simplify(zs)) == "log(Abs(x!/x))"
#    assert sympy_to_theano(zs, m) == 1
#    assert sympy_to_theano(sympy.simplify(zs), m) == 1


def test_theano_to_sympy_inputs():
    at = wt + xt
    bt = at + yt
    assert theano_to_sympy(bt, [at, yt]) == sympy.Symbol(var_string(at)) + ys


def test_theano_to_theano_inputs():
    at = wt + xt
    bt = at + yt
    zs = theano_to_sympy(bt, [at, yt])
#    m = shape_and_dtype_map(bt)
    inputs_map = {var_string(at): at, var_string(yt): yt}
    zt = sympy_to_theano(zs, {}, inputs_map)
    s1 = theano.printing.pprint(zt)
    s2 = theano.printing.pprint(bt)
    s3 = theano.printing.pprint(yt + at)
    assert s1 == s2 or s1 == s3


def test_theano_sympy_optimizer():
    zt = tt.cos(xt) ** 2 + tt.sin(xt) ** 2
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including('sympy')
    f = theano.function([xt], zt, mode=mode)
    # The graph is just a deep copy of the constant.
    assert len(f.maker.fgraph.toposort()) == 1
    assert isinstance(f.maker.fgraph.toposort()[0].op,
                      theano.compile.DeepCopyOp)

    # Test that we don't crash when we can't use SymPy
    # to optimize the graph.
    zt = tt.cos(xt.dimshuffle('x')) ** 2 + tt.sin(xt) ** 2
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including('sympy')
    f = theano.function([xt], zt, mode=mode)
    assert any(isinstance(node.op, theano.tensor.Elemwise)
               for node in f.maker.fgraph.toposort())
