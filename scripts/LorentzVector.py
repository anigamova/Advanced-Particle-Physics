from sympy import *
from sympy.tensor.tensor import *
init_printing()

Lorentz = TensorIndexType("Lorentz", dummy_name="alpha")
pprint(Lorentz.metric)
pprint(Lorentz.delta)

mu = TensorIndex("mu", Lorentz, is_up=True)
nu = TensorIndex("nu", Lorentz, is_up=True)
pprint(nu)

repl = {Lorentz: diag(1, -1, -1, -1)}
#pprint(repl)

X = TensorHead('X', [Lorentz], TensorSymmetry.no_symmetry(1))
pprint(f"X: {X}")
pprint(f"index_types: {X.index_types}")
pprint(f"rank: {X.rank}")

pprint(X)
pprint(X(mu))

t, x, y, z = symbols('t x y z')
repl.update({X(mu): [t, x, y, z]})

expr = X(mu).replace_with_arrays(repl)
pprint(expr)
expr = X(-mu).replace_with_arrays(repl)
pprint(expr)
expr = X(mu)*X(-mu)
pprint(expr)
expr = expr.replace_with_arrays(repl)
pprint(expr)

print("==============")
A, B = tensor_heads("A B", [Lorentz])
repl[A(mu)] = [t, x, y, z]
repl[B(mu)] = [1, 2, 3, 4]
expr = B(mu).replace_with_arrays(repl)
pprint (expr)
expr = B(-mu).replace_with_arrays(repl)
pprint (expr)
expr = A(mu)*B(-mu)
pprint (expr.replace_with_arrays(repl))

pprint(A(nu).get_indices())

print("==============")
C = A(mu)+B(mu)
pprint(C)
expr = C.substitute_indices((mu,mu))
pprint(expr)
expr = C.substitute_indices((mu,nu))
pprint(expr)

D = C.substitute_indices((mu,mu))
pprint(D.replace_with_arrays(repl))
pprint(D.get_indices())

pprint(">>>>>>>")
g = Lorentz.metric
E = A(mu)*g(-mu,-nu)
pprint(E)
pprint(E.contract_metric(g))

pprint("----------")
T = TensorHead("T",[Lorentz,Lorentz])
pprint(T)
pprint(T(mu,nu))

expr = T(mu,nu)*A(-nu)
pprint(expr)

repl[T(mu,nu)] = [
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
]

pprint("\n")
pprint(T(mu,nu))
pprint("=")
pprint(T(mu,nu).replace_with_arrays(repl))

pprint("\n")
pprint(A(-mu))
pprint("=")
pprint(A(-mu).replace_with_arrays(repl))

pprint("\n")
expr = T(mu,nu)*A(-nu)
pprint(expr)
pprint("=")
pprint(expr.replace_with_arrays(repl))

pprint("\n")
expr = T(mu,nu)*A(-mu)*B(-nu)
pprint(expr)
pprint("=")
pprint(expr.replace_with_arrays(repl))
