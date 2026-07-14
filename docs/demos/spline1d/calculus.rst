Spline Calculus
===============

How to apply integration and differentiation to one-dimensional polynomial splines.

----

Area Under the Curve
--------------------

Let a piecewise polynomial one-dimensional periodic spline of nonnegative integer degree :math:`n` and positive integer period :math:`K` be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k),` where :math:`{\mathbf{c}}=\left(c[k]\right)_{k=0}^{K-1}\in{\mathbb{R}}^{K}` is its vector of spline coefficients and where :math:`\delta x` is its delay. The synthesis function of the spline :math:`f` is the B-spline :math:`\beta^{n}.` We want to compute the quantity

..  math::
    \int_{a}^{b}\,f(x)\,{\mathrm{d}}x,

with integration bounds :math:`a\in{\mathbb{R}}` and :math:`b\in{\mathbb{R}}.`

We provide now a few lines of code that create a random spline of specified period, degree, and delay. The integration bounds can be set freely and the returned integral is computed exactly, without approximation.

..  admonition:: Jupyter Lab notebook

    `Area under a spline curve <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/calculus/spline_integrate.ipynb&mode=single-document>`_

