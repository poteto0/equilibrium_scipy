# equilibrium_scipy

### Installing
```sh
pip install git+https://github.com/poteto0/equilibrium_scipy
```

```python
import equilibrium_scipy as eq
```

### Definition
The func means wrapped function. 

```python
EQ = eq.equilibrium(func = Function)
```

You can array but you need arg of t.

```python
def func(x[0,0], t=0):
  return x
```

### Calculate eq points
You can solve the function and calculate jacobian by scipy.

```python
root, jacobi = EQ.root(x0=x0, method='hybr')
```

You can judge stable or unstable.

```python
eigen  = EQ.calc_eigen(jacobi)
stable = EQ.is_stable(eigen)
```

### Phase Diagram
You can draw phase diagram which is phase diagram and nurkline.
If you want plot eq points, run `multi_equilibrium_analyze_and_plot()`.
If you want ode, run `ode()`.

y1_base means scale of y1 axis -> default 1.0

```python
y1 = np.linspace(y1_min, y1_max, num)
y2 = np.linspace(y2_min, y2_max, num)

# colors means nurkline colors
EQ.draw_phase_diagram(y1, y2, y1_base=1.0, colors=["green", "red"])

# If you want plot eq points
# this draw eq points with stable / unstable (● / ○)
x_list = [ # initial points list
            [r0, q0],
            [r1, q1],
            [r2, q2]
          ]
EQ.multi_equilibrium_analyze_and_plot(x_list)

# If you want ode
EQ.ode([y1_c, y2_c], y1_base=1.0, t_length=100000)

plt.show() # or Eq.show_plot()
```