Spline Calculus
===============

How to apply integration and differentiation to one-dimensional polynomial splines.

----

Area Under the Curve
--------------------

Let a piecewise polynomial one-dimensional periodic spline of nonnegative integer degree :math:`n` and positive integer period :math:`K` be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k),` where :math:`{\mathbf{c}}=\left(c[k]\right)_{k=0}^{K-1}\in{\mathbb{R}}^{K}` is its vector of spline coefficients and where :math:`\delta x` is its delay. The synthesis function of the spline :math:`f` is the B-spline :math:`\beta^{n}.` We want to compute the quantity

..  math::
    \int_{a}^{b}\,f(t)\,{\mathrm{d}}t,

with integration bounds :math:`a\in{\mathbb{R}}` and :math:`b\in{\mathbb{R}}.`

We provide now a few lines of code that create a random spline of specified period, degree, and delay. The integration bounds can be set freely and the returned integral is computed exactly, without approximation.

..  admonition:: Jupyter Lab notebook

    `Area under a spline curve <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/calculus/spline_integrate.ipynb&mode=single-document>`_

----

Anti-Gradient
-------------

Consider the spline :math:`f` above. While it is periodic, its generic integral :math:`\int_{0}^{x}\,f(t)\,{\mathrm{d}}t` is not necessarily periodic in terms of :math:`x.` Indeed, the integral :math:`\int_{0}^{K+x}\,f(t)\,{\mathrm{d}}t` can be segmented in two parts as :math:`\int_{0}^{K}\,f(t)\,{\mathrm{d}}t+\int_{K}^{K+x}\,f(t)\,{\mathrm{d}}t.` It is then equal to :math:`K\,E\{f\}+\int_{0}^{x}\,f(t)\,{\mathrm{d}}t,` which matches :math:`\int_{0}^{x}\,f(t)\,{\mathrm{d}}t` only when the mean of :math:`f` vanishes.

The anti-gradient of :math:`f` is defined as the function

..  math::
    S_{f}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto S_{f}(x)=\int_{0}^{x}\,\left(f(t)-E\{f\}\right)\,{\mathrm{d}}t.

It turns out to be a periodic spline of degree :math:`n+1.` As such, it can be handled by the ``splinekit`` library. Its most relevant property is that the gradient of the anti-gradient of a spline is the spline itself, up to an additive constant, so that :math:`\dot{S}_{f}=\left(f-E\{f\}\right).`

We provide now a few lines of code that first create a random spline of specified period, degree, and delay, and then show its anti-gradient, along with the gradient of this anti-gradient.

..  admonition:: Jupyter Lab notebook

    `Anti-gradient of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/calculus/spline_antigrad.ipynb&mode=single-document>`_

----

Gradient
--------

Consider the :math:`K`-periodic spline

..  math::
    f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k),

with :math:`n\in{\mathbb{N}}+1` a positive degree. The spline gradient is its first derivative

..  math::
    \dot{f}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto \dot{f}(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\dot{\beta}^{n}(x-\delta x-k).

Because B-splines of positive degree satisfy the relation :math:`\dot{\beta}^{n}(x)=\left(\beta^{n-1}(x+\frac{1}{2})-\beta^{n-1}(x-\frac{1}{2})\right)` for all :math:`x\in{\mathbb{R}},` the gradient of a spline is again a spline, albeit one whose degree is smaller and whose delay is offset by :math:`\frac{1}{2}.` As such, the gradient can be handled by the ``splinekit`` library.

The *centered finite difference* of a function :math:`f` is parameterized by a step :math:`h\in{\mathbb{R}}\setminus\{0\}` and is defined as the quantity :math:`\frac{1}{2\,h}\,\left(f(x+h)-f(x-h)\right).` At those arguments :math:`x\in{\mathbb{R}}` where the function is differentiable, one has that :math:`\dot{f}(x)=\lim_{h\rightarrow0}\frac{1}{2\,h}\,\left(f(x+h)-f(x-h)\right).`

We provide now a few lines of code that first create a random spline of specified period, degree, and delay, and then show its true gradient, along with the approximation of the gradient provided by centered finite differences of rational step :math:`h=\frac{1}{m},` with :math:`m\in{\mathbb{N}}+1` a positive integer.

..  admonition:: Jupyter Lab notebook

    `Gradient of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/calculus/spline_gradient.ipynb&mode=single-document>`_

----

Derivative of Positive Integer Order
------------------------------------

Given the spline :math:`f` as above, we want now to compute its derivative of order :math:`m,` with :math:`m\in{\mathbb{N}}+1.` This derivative will be a true function whenever the degree of the spline is large enough, with :math:`n\in{\mathbb{N}}+m;` more precisely, it is going to be a spline of degree :math:`\left(n-m\right).` To do so, we choose here to proceed with the linear-algebra equivalent representation of the spline as

..  math::
    f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k)={\mathbf{c}}_{r}^{{\mathsf{T}}}\,{\mathbf{W}}^{n}\,{\mathbf{v}}^{n}(\chi(x)).

There, the partial vector of coefficients :math:`{\mathbf{c}}_{r}\in{\mathbb{R}}^{n+1}` is defined as :math:`{\mathbf{c}}_{r}=\left(c[{\left(k-r\right)\bmod K}]\right)_{k=0}^{n},` where :math:`r=\left\lceil\xi(x)\right\rceil\in{\mathbb{Z}}` with :math:`\xi(x)=\left(\frac{n-1}{2}-x-\delta x\right)\in{\mathbb{R}}.` The spline evaluation matrix :math:`{\mathbf{W}}^{n}\in{\mathbb{R}}^{\left(n+1\right)\times\left(n+1\right)}` depends on :math:`n` only. Finally, the Vandermonde vector :math:`{\mathbf{v}}^{n}(\chi(x))\in{\mathbb{R}}^{n+1}` is computed as :math:`{\mathbf{v}}^{n}(\chi(x))=\left(1,\left(\chi^{k}(x)\right)_{k=1}^{n}\right),` with :math:`\chi(x)=\left(r-\xi(x)\right)\in[0,1).`

With these definitions, the derivative of order :math:`m` is

..  math::
    \frac{{\mathrm{d}}^{m}f(x)}{{\mathrm{d}}x^{m}}={\mathbf{c}}_{r}^{{\mathsf{T}}}\,{\mathbf{W}}^{n}\,{\mathbf{\Lambda}}_{m}^{n}\,\left(\left(0\right)_{k=0}^{m-1},\left([{\mathbf{v}}^{n}(\chi(x))]_{k-m+1}\right)_{k=m}^{n}\right),

where :math:`{\mathbf{\Lambda}}_{m}^{n}={\mathbf{diag}}(\left(\frac{k!}{\left(k-m\right)!}\right)_{k=0}^{n}).` This allows one to precompute :math:`{\mathbf{W}}^{n}\,{\mathbf{\Lambda}}_{m}^{n}` since it depends neither on the delay of the spline nor on the spline coefficients. Furthermore, the fact that there are leading zeros in the vector :math:`\left(\left(0\right)_{k=0}^{m-1},\left([{\mathbf{v}}^{n}(\chi(x))]_{k-m+1}\right)_{k=m}^{n}\right)\in{\mathbb{R}}^{n+1}` gives the opportunity of additional computational savings.

We provide now a few lines of code that create a random spline of specified period, degree, and delay, and show its derivative of arbitrary order.

..  admonition:: Jupyter Lab notebook

    `Derivatives of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/calculus/spline_derivative.ipynb&mode=single-document>`_
