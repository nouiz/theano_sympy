theano_sympy
============

Function to transform theano graph &lt;-> sympy graph.

Status
======

The SymPy -> Theano conversion is implemented and merged into SymPy proper under sympy.printing.theanocode

See the following pull request and blogposts for details on use

sympy/sympy#1905
http://matthewrocklin.com/blog/work/2013/03/19/SymPy-Theano-part-1/
http://matthewrocklin.com/blog/work/2013/03/28/SymPy-Theano-part-2/

There is nothing integrated in Theano to use SymPy simplification automatically.
