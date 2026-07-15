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
