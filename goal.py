import theano
import theano.tensor as tt

import numpy

x = tt.matrix('x')
y = tt.matrix('y')
z = tt.matrix('z')

output = (z.T * (tt.cos(x)**2 + tt.sin(x)**2)).sum(1)

f = theano.function([x,y,z], output)

print theano.printing.debugprint(f)

xx = numpy.ones((5,5), dtype='float64')
print f(xx,xx,xx)
