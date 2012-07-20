from graph_translation import *

# SETUP
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
    zt = tt.cos(xt)**2 + tt.sin(xt)**2
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
