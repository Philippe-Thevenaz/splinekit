Multiresolution
===============

How to manipulate the period of splines.

----

Upscaling
---------

We consider two approaches to upscale a spline by an arbitrary positive integer factor :math:`M\in{\mathbb{N}}+1.` The first one builds an upscaled spline that is an exact replica of the spline at the nominal scale, except for being larger. The second one offers freedom in the choice of the degree and delay of the upscaled spline.

Exact
^^^^^

A polynomial B-spline :math:`\beta^{n}` of any nonnegative integer degree :math:`n\in{\mathbb{N}}` satisfies the M-scale equality, according to which its version at nominal scale can be expressed as a finite weighted sum of B-splines that are appropriately shifted and shrunk by the factor :math:`M.` (Equivalently, a B-spline enlarged by :math:`M` can be expressed through B-splines at their nominal scale.) The collection of weights depend on :math:`n` and :math:`M` and is called the M-scale filter :math:`h_{M}^{n}.`

Because a spline is itself a weighted sum of shifted B-splines, it benefits from the M-scale equality. Indeed, let a spline of positive integer period :math:`K\in{\mathbb{N}}+1` and delay :math:`\delta x\in{\mathbb{R}}` be defined at nominal scale as

..  math::
    f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k),

and let its :math:`M`-enlarged version (with the same, non-enlarged delay) be

..  math::
    f_{M}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{M}(x)=f(\frac{x}{M})=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(\frac{x}{M}-\delta x-k).

There, the B-splines that take part in the sum are not at their nominal scale. However, for all :math:`x\in{\mathbb{R}},` it also holds that

..  math::
    f_{M}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{M}^{n}[{k\bmod \left(M\,K\right)}]\,\beta^{n}(x-\delta x_{M}^{n}-k),

where it is more immediately apparent that :math:`f_{M}` is not only :math:`\left(M\,K\right)`-periodic, but also a weighted sum of integer-shifted B-splines at nominal scale, with the vector of weights being :math:`\left(c_{M}^{n}[k]\right)_{k=0}^{M\,K-1}` and the delay being :math:`\delta x_{M}^{n}.` More precisely, the equality between :math:`f_{M}` and the enlarged :math:`f` is achieved for

..  math::
    c_{M}^{n}[k]=\frac{1}{M^{n}}\,\sum_{q=\left\lceil\frac{{k\bmod\left(M\,K\right)}-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{{k\bmod\left(M\,K\right)}}{M}\right\rfloor}\,c[{q\bmod K}]\,h_{M}^{n}[k-M\,q]

and

..  math::
    \delta x_{M}^{n}[k]=M\,\delta x-\frac{\left(M-1\right)\,\left(n+1\right)}{2}.

We now propose a few lines of code that first create a random spline of specified period, degree, and delay, and then enlarge it by a factor :math:`M.` We display a stack where the top figure contains the spline at its nominal size and the bottom figure contains the enlarged spline. A pair of synchronized sliders allows one to explore the values taken by the two functions and to conclude that, up to change of scale, the two versions are identical.

..  admonition:: Jupyter Lab notebook

    `Upscaling of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_upscaling.ipynb&mode=single-document>`_

----

Projected
^^^^^^^^^

..  math::
    \left(c_{M}^{n}\right)'[k]=\frac{1}{M^{n}}\,\sum_{q=\left\lceil\frac{{k\bmod\left(M\,K\right)}-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{{k\bmod\left(M\,K\right)}}{M}\right\rfloor}\,c[{q\bmod K}]\,h_{M}^{n}[k-M\,q]

..  math::
    \left(c_{M}^{n}\right)''[k]=\left(\left(b^{2\,n+1}\right)^{-1}*\left(c_{M}^{n}\right)'\right)[{k\bmod M\,K}]

..  math::
    c_{M}^{n}[k]=\sum_{q=\left\lceil-x_{0}-\frac{n_{0}+n+2}{2}\right\rceil}^{\left\lfloor-x_{0}+\frac{n_{0}+n+2}{2}\right\rfloor}\,\beta^{n_{0}+n+1}(q+x_{0})\,\left(c_{M}^{n}\right)''[{\left(k-q\right)\bmod M\,K}]

:math:`x_{0}=\left(\delta x-M\,\delta x_{0}+\frac{\left(M-1\right)\,\left(n_{0}+1\right)}{2}\right)`

``f.upscaled(M).projected(degree = degree, delay = delay)``


----

Downscaling
-----------

----

Rescaling
---------

