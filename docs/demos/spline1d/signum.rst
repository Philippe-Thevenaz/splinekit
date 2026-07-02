Sign of Splines
===============

How to access the signed pieces of a piecewise-polynomial periodic one-dimensional spline and how to take advantage of them.

----

Signum Function
---------------

The signum function is the complex-to-complex mapping

..  math::
    {\mathrm{sgn}}:{\mathbb{C}}\rightarrow{\mathbb{C}},z\mapsto{\mathrm{sgn}}(z)=\left\{\begin{array}{ll}0,&z=0\\{\mathrm{e}}^{{\mathrm{j}}\,\arg z},&z\neq0.\end{array}\right.

It maps the origin of the complex plane to itself, and it maps every other point to its closest point on the unit circle. Those points :math:`z\in{\mathbb{C}}` such that :math:`{\mathrm{sgn}}(z)=1` are said to be positive, and those points :math:`z\in{\mathbb{C}}` such that :math:`{\mathrm{sgn}}(z)=\left(-1\right)` are said to be negative; the set of the real numbers can then be partitioned as :math:`{\mathbb{R}}=\{z\in{\mathbb{C}}:{\mathrm{sgn}}(z)=-1\}\cup\{0\}\cup\{z\in{\mathbb{C}}:{\mathrm{sgn}}(z)=1\}.`

----

Spline Signed Pieces
--------------------

Given a periodic spline of period :math:`K,` we want to establish a set of intervals that partitions the main period :math:`[0,K)` in such a way that each element of the set is an interval of largest diameter over which the spline takes a constant-valued sign. For ease of use, the set is returned as a list.

We now propose a few lines of code that create and display a random spline and extract its constant-sign pieces.

..  admonition:: Jupyter Lab notebook

    `Constant-sign pieces of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/sgn/spline_sgn.ipynb&mode=single-document>`_

----

Spline Zeros
------------

Given a periodic spline of period :math:`K,` we want to establish a set of intervals of largest diameter over which the spline vanishes. We enforce that the diameter of the enclosure of all returned intervals is not larger than the period of this spline. For ease of use, the set is returned as a list.

We now propose a few lines of code that create and display a curated spline over which we illustrate several configurations of zeros, represented by a mixture of degenerate and proper intervals.

..  admonition:: Jupyter Lab notebook

    `Zeros of a curated spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/sgn/spline_curated_zeros.ipynb&mode=single-document>`_

