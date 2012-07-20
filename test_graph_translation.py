from graph_translation import *

# SETUP
xt = theano.tensor.scalar('x')
yt = theano.tensor.scalar('y')
zt = theano.tensor.add(xt, yt)

xs = sympy.Symbol('x')
ys = sympy.Symbol('y')
zs = sympy.Add(xs, ys)

print "transformed graph"
print theano_to_sympy(xt)
print theano_to_sympy(zt)
print sympy.simplify(theano_to_sympy(tt.cos(xt)**2 +
                                     tt.sin(xt)**2))

var_map = {xs.name: ('float32', (False, False)),
           ys.name: ('float32', (False, False)),
       }
print sympy_to_theano(xs, var_map)

print sympy_to_theano(xs + ys, var_map)
print theano.printing.pprint(
    sympy_to_theano(sympy.cos(xs)**2 + sympy.sin(xs)**2, var_map))

assert shape_and_dtype_map(xt) == {'x': ('float64', ())}
zt = tt.cos(xt)**2 + tt.sin(xt)**2
zs = theano_to_sympy(zt)
m = shape_and_dtype_map(zt)
print sympy_to_theano(sympy.simplify(zs), m)
