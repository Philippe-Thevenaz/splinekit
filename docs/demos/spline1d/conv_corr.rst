Convolutions and Correlations
=============================

How to convolve and correlate two piecewise-polynomial periodic one-dimensional splines.

----

Convolution
-----------

Let a first one-dimensional mapping be :math:`h:{\mathbb{R}}\rightarrow{\mathbb{C}}` and let a second one-dimensional mapping be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}.` Their *convolution* is notated :math:`h*f` and defined as the mapping

..  math::
    g:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto g(x)=\left(h*f\right)(x):=\int_{-\infty}^{\infty}\,h(y)\,f(x-y)\,{\mathrm{d}}y.

Caution must be taken that the integration process does converge. In particular, suppose that :math:`h` and :math:`f` are both periodic, with respective integer periods :math:`K_{1}` and :math:`K_{2}.` Now, if we assume that :math:`h*f` converges, then it is admissible to segment the corresponding improper integral as :math:`\left(h*f\right)(x)=\sum_{k\in{\mathbb{Z}}}\,\int_{0}^{K_{1}\,K_{2}}\,h(y+k\,K_{1}\,K_{2})\,f(x-y-k\,K_{1}\,K_{2})\,{\mathrm{d}}y.` The :math:`K_{1}` and :math:`K_{2}` periodicities imply that :math:`\left(h*f\right)(x)=\sum_{k\in{\mathbb{Z}}}\,\int_{0}^{K_{1}\,K_{2}}\,h(y)\,f(x-y)\,{\mathrm{d}}y.` The assumed convergence of :math:`h*f` implies that the definite integral takes some finite value :math:`F=\int_{0}^{K_{1}\,K_{2}}\,h(y)\,f(x-y)\,{\mathrm{d}}y\in{\mathbb{C}}.` But then, :math:`\sum_{k\in{\mathbb{Z}}}\,F` is not a finite quantity in general, which contradicts our assumption of convergence of the convolution of two periodic arbitrary functions.

Periodic Convolution
--------------------

Let a first one-dimensional :math:`K`-periodic mapping be :math:`h:{\mathbb{R}}\rightarrow{\mathbb{C}}` and let a second :math:`K`-periodic one-dimensional mapping be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}},` with :math:`h(x)=h(x+K)` and :math:`f(x)=f(x+K)` for all :math:`x\in{\mathbb{R}}.` Their *periodic convolution* is notated :math:`h*f` and defined as the mapping

..  math::
    g:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto g(x)=\left(h*f\right)(x):=\int_{0}^{K}\,h(y)\,f(x-y)\,{\mathrm{d}}y.

Caution must be taken that the integration process does converge; for simplicity, we shall henceforth restrict ourselves to nonsingular mappings :math:`h` and :math:`f.` Note that a convolution and a periodic convolution make use of the same operator :math:`*` but, more often than not, the context will allow one to distinguish between them.

Crosscorrelation
----------------

The *crosscorrelation* of :math:`f_{1}:{\mathbb{R}}\rightarrow{\mathbb{C}}` and :math:`f_{2}:{\mathbb{R}}\rightarrow{\mathbb{C}}` is notated :math:`f_{1}\star f_{2}` and defined as the mapping

..  math::
    \chi:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto \chi(x)=\left(f_{1}\star f_{2}\right)(x):=\int_{-\infty}^{\infty}\,f_{1}^{*}(y)\,f_{2}(x+y)\,{\mathrm{d}}y.

When the two mappings are periodic, the same caveats apply as for their convolution. Their *periodic crosscorrelation* is the mapping

..  math::
    \chi:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto \chi(x)=\left(f_{1}\star f_{2}\right)(x):=\int_{0}^{K}\,f_{1}^{*}(y)\,f_{2}(x+y)\,{\mathrm{d}}y.

Note that one must pay attention to distinguish the convolution operator :math:`*` from the crosscorrelation operator :math:`\star.`

Autocorrelation
---------------

The crosscorrelation :math:`R_{ff}=f\star f` is called the autocorrelation of :math:`f.`

Normalized Periodic Crosscorrelation
------------------------------------

..  One often takes advantage of crosscorrelation to describe how well :math:`f_{1}` matches :math:`f_{2}` in terms of their relative shift. 














Signum Function
^^^^^^^^^^^^^^^

The signum function is the complex-to-complex mapping

..  math::
    {\mathrm{sgn}}:{\mathbb{C}}\rightarrow{\mathbb{C}},z\mapsto{\mathrm{sgn}}(z)=\left\{\begin{array}{ll}0,&z=0\\{\mathrm{e}}^{{\mathrm{j}}\,\arg z},&z\neq0.\end{array}\right.

It maps the origin of the complex plane to itself, and it maps every other point to its closest point on the unit circle. Those points :math:`z\in{\mathbb{C}}` such that :math:`{\mathrm{sgn}}(z)=1` are said to be positive, and those points :math:`z\in{\mathbb{C}}` such that :math:`{\mathrm{sgn}}(z)=\left(-1\right)` are said to be negative; the set of the real numbers can then be partitioned as :math:`{\mathbb{R}}=\{z\in{\mathbb{C}}:{\mathrm{sgn}}(z)=-1\}\cup\{0\}\cup\{z\in{\mathbb{C}}:{\mathrm{sgn}}(z)=1\}.`

Zeros
^^^^^

Consider the piecewise polynomial spline :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` of nonnegative integer degree :math:`n.` If an interval :math:`{\mathbb{X}}` can be found such that it is of maximal diameter and such that :math:`\forall x\in{\mathbb{X}}:f(x)=0,` then this interval is said to be a *zero* of the spline.

Zero-Crossings
^^^^^^^^^^^^^^

Spline of Null Degree
"""""""""""""""""""""

Consider a spline of degree :math:`n=0.` If a proper interval :math:`{\mathbb{X}}` can be found such that :math:`\forall x\in{\mathbb{X}}:f(x)=0` and such that :math:`f(\inf{\mathbb{X}})\,f(\sup{\mathbb{X}})<0,` then the interval is said to be a *zero-crossing* of the spline.

If two proper intervals :math:`{\mathbb{A}}` and :math:`{\mathbb{B}}` can be found such that they are mutually disjoint, such that :math:`\sup{\mathbb{A}}=\inf{\mathbb{B}},` and such that :math:`\forall a\in{\mathbb{A}},\forall b\in{\mathbb{B}}:f(a)\,f(b)<0,` then, the degenerate interval :math:`{\mathbb{X}}=\{x\in{\mathbb{R}}:\sup{\mathbb{A}}=x=\inf{\mathbb{B}}\}` is said to be a *zero-crossing* of the spline.

Spline of Positive Degree
"""""""""""""""""""""""""

Consider a spline of degree :math:`n>0.` If three intervals :math:`{\mathbb{A}},` :math:`{\mathbb{X}},` and :math:`{\mathbb{B}}` can be found such that they are mutually disjoint, such that :math:`\left(\sup{\mathbb{A}}=\inf{\mathbb{X}}\right)\wedge\left(\sup{\mathbb{X}}=\inf{\mathbb{B}}\right),` such that :math:`\forall x\in{\mathbb{X}}:{\mathrm{sgn}}(f(x))=0,` and such that :math:`\forall a\in{\mathbb{A}},\forall b\in{\mathbb{B}}:f(a)\,f(b)<0,` then, :math:`{\mathbb{X}}` is said to be a *zero-crossing* of the spline.

Descending or Ascending Zero-Crossings
""""""""""""""""""""""""""""""""""""""

A *descending* zero-crossing is such that :math:`\lim_{x\uparrow\inf{\mathbb{X}}}{\mathrm{sgn}}(f(x))>\lim_{x\downarrow\sup{\mathbb{X}}}{\mathrm{sgn}}(f(x)).`

An *ascending* zero-crossing is a zero-crossing that is not descending.

----

Spline Signed Pieces
--------------------

Given a periodic spline of period :math:`K,` we want to establish a set of intervals that partitions the main period :math:`[0,K)` in such a way that each element of the set is an interval of largest diameter over which the spline takes a constant-valued sign. For ease of use, the set is returned as a list.

We now propose a few lines of code that create and display a random spline and extract its constant-sign pieces.

..  admonition:: Jupyter Lab notebook

    `Constant-sign pieces of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/sgn/spline_sgn.ipynb&mode=single-document>`_

----

Spline Zeros and Zero-Crossings
-------------------------------

Given a periodic spline of period :math:`K,` we want to establish a set of intervals of largest diameter that give either its zeros or its zero-crossings. We enforce that the diameter of the enclosure of all returned intervals is not larger than the period of this spline. For ease of use, the set is returned as a list.

We now propose a few lines of code that create and display a curated spline over which we illustrate and discuss several configurations, represented by a mixture of degenerate and proper intervals. In our control of the configurations, we retain the same spline coefficients but we let the degree of the spline vary.

..  admonition:: Jupyter Lab notebook

    `Zeros of a curated spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/sgn/spline_curated_zeros.ipynb&mode=single-document>`_

In the next notebook, we examine random splines of specified period, degree, and delay. Here, it is the values of the spline samples that we maintain across configurations.

..  admonition:: Jupyter Lab notebook

    `Zeros of a random spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/sgn/spline_zeros.ipynb&mode=single-document>`_
