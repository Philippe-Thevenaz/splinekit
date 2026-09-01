Multiresolution
===============

How to manipulate the period of splines.

----

Upscaling
---------

We consider two approaches to upscale a spline by an arbitrary positive integer magnification :math:`M\in{\mathbb{N}}+1.` The first one builds an upscaled spline that is an exact replica of the spline at the nominal scale, except for being larger. The second one offers freedom in the choice of the degree and delay of the upscaled spline.

Exact
^^^^^

A polynomial B-spline :math:`\beta^{n_{0}}` of any nonnegative integer degree :math:`n_{0}\in{\mathbb{N}}` satisfies the M-scale equality, according to which its version at nominal scale can be expressed as a finite weighted sum of B-splines that are appropriately shifted and shrunk by the factor :math:`M.` (Equivalently, a B-spline enlarged by :math:`M` can be expressed through B-splines at their nominal scale.) The collection of weights depend on :math:`n_{0}` and :math:`M` and is called the M-scale filter :math:`h_{M}^{n_{0}}.`

Because a spline is itself a weighted sum of shifted B-splines, it benefits from the M-scale equality. Indeed, let a spline of positive integer period :math:`K_{0}\in{\mathbb{N}}+1` and delay :math:`\delta x_{0}\in{\mathbb{R}}` be defined at nominal scale as

..  math::
    f_{0}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{0}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(x-\delta x_{0}-k),

and let its :math:`M`-enlarged version (with the same, non-enlarged delay) be

..  math::
    f_{{\color{blue}{M}}}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{{\color{blue}{M}}}(x)=f_{0}(\frac{x}{{\color{blue}{M}}})=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(\frac{x}{{\color{blue}{M}}}-\delta x_{0}-k).

There, the B-splines that take part in the sum are not at their nominal scale. However, for all :math:`x\in{\mathbb{R}},` the M-scale equality implies that

..  math::
    f_{M}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{M}^{n_{0}}[{k\bmod K}]\,\beta^{n_{0}}(x-\delta x_{M}^{n_{0}}-k)

also holds true, where it is more immediately apparent that :math:`f_{M}` is not only :math:`K`-periodic with :math:`K=M\,K_{0},` but also a weighted sum of integer-shifted B-splines at nominal scale, with the vector of weights being :math:`{\mathbf{c}}_{M}^{n_{0}}=\left(c_{M}^{n_{0}}[k]\right)_{k=0}^{K-1}` and the delay being :math:`\delta x_{M}^{n_{0}}.` More precisely, the equality between :math:`f_{M}` and the enlarged :math:`f_{0}` is achieved for

..  math::
    {\mathbf{c}}_{M}^{n_{0}}=\left(\frac{1}{M^{n_{0}}}\,\sum_{q=\left\lceil\frac{k-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{k}{M}\right\rfloor}\,c_{0}[{q\bmod K_{0}}]\,h_{M}^{n_{0}}[k-M\,q]\right)_{k=0}^{K-1}

and

..  math::
    \delta x_{M}^{n_{0}}=M\,\delta x_{0}-\frac{\left(M-1\right)\,\left(n+1\right)}{2}.

We now propose a few lines of code that first create a random spline of specified period, degree, and delay, and then enlarge it by a factor :math:`M.` We display a stack where the top figure contains the spline at its nominal size and the bottom figure contains the enlarged spline. A pair of synchronized sliders allows one to explore the values taken by the two functions and to conclude that, up to change of scale, the two versions are identical.

..  admonition:: Jupyter Lab notebook

    `Upscaling of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_upscaling.ipynb&mode=single-document>`_

----

Projected
^^^^^^^^^

We want now to determine which spline :math:`g_{M}` of arbitrary degree :math:`n\in{\mathbb{N}}` and arbitrary delay :math:`\delta x\in{\mathbb{R}}` best represents the magnified spline :math:`f_{M}` with, as before, a positive integer magnification :math:`M\in{\mathbb{N}}+1` and

..  math::
    f_{M}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{M}(x)=f_{0}(\frac{x}{M})=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(\frac{x}{M}-\delta x_{0}-k),

where the spline of of positive integer period :math:`K_{0}\in{\mathbb{N}}+1` at nominal scale is

..  math::
    f_{0}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{0}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(x-\delta x_{0}-k).


More precisely, for :math:`K=M\,K_{0},` we want to establish the value of the spline coefficients :math:`c_{M}^{n}` that parameterize the :math:`K`-periodic spline

..  math::
    g_{M}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto g_{M}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{M}^{n}[{k\bmod K}]\,\beta^{n}(x-\delta x-k)

and are such that the least-squares criterion

..  math::
    J=\frac{1}{2}\,\int_{0}^{K}\,\left(g_{M}(x)-f_{M}(x)\right)^{2}\,{\mathrm{d}}x

is minimized. While the desired spline :math:`g_{M}` could be directly obtained as ``gm = f0.upscaled(m).projected(degree = n, delay = dx)``, we propose here a combined approach. Letting :math:`{\mathbf{[\![}}\cdot\,{\mathbf{]\!]}}` be the notation for the Iverson bracket, we observe that, :math:`\forall q\in[0\ldots K],`

..  math::
    \frac{\partial J}{\partial c_{M}^{n}[q]}=\int_{0}^{K}\,\sum_{k\in{\mathbb{Z}}}\,{\mathbf{[\![}}q={k\bmod K}\,{\mathbf{]\!]}}\,\beta^{n}(x-\delta x-k)\,\left(g_{M}(x)-f_{M}(x)\right)\,{\mathrm{d}}x.

Wen the coefficients are optimal, :math:`\frac{\partial J}{\partial c_{M}^{n}[q]}` vanishes. Now, the multiplication of this zero value by the quantity :math:`c_{M}^{n}[q]` is still zero, and so is the sum over all indices :math:`q.` This leads to

..  math::
    \begin{array}{rcl}
    0&=&\sum_{q=0}^{K-1}\,c_{M}^{n}[q]\,\frac{\partial J}{\partial c_{M}^{n}[q]}\\
    &=&\int_{0}^{K}\,g_{M}(x)\,\left(g_{M}(x)-f_{M}(x)\right)\,{\mathrm{d}}x\\
    &=&\left(g_{M}^{\vee}*g_{M}\right)(0)-\left(g_{M}^{\vee}*f_{M}\right)(0),
    \end{array}

where the last equality involves periodic convolutions and mirrored versions :math:`g_{M}^{\vee}` of :math:`g_{M}.` The solution of this equation in terms of the vector :math:`{\mathbf{c}}_{M}^{n}=\left(c_{M}^{n}[q]\right)_{q=0}^{K-1}` is obtained in the three successive steps

..  math::
    \left({\mathbf{c}}_{M}^{n}\right)'=\left(\frac{1}{M^{n}}\,\sum_{q=\left\lceil\frac{k-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{k}{M}\right\rfloor}\,c_{0}[{q\bmod K_{0}}]\,h_{M}^{n}[k-M\,q]\right)_{k=0}^{K-1}

..  math::
    \left({\mathbf{c}}_{M}^{n}\right)''=\left(\left(\left(b^{2\,n+1}\right)^{-1}*\left(c_{M}^{n}\right)'\right)[k]\right)_{k=0}^{K-1}

..  math::
    {\mathbf{c}}_{M}^{n}=\left(\sum_{q=\left\lceil-x_{0}-\frac{n_{0}+n+2}{2}\right\rceil}^{\left\lfloor-x_{0}+\frac{n_{0}+n+2}{2}\right\rfloor}\,\beta^{n_{0}+n+1}(q+x_{0})\,\left(c_{M}^{n}\right)''[{\left(k-q\right)\bmod K}]\right)_{k=0}^{K-1},

where :math:`x_{0}=\left(\delta x-M\,\delta x_{0}+\frac{\left(M-1\right)\,\left(n_{0}+1\right)}{2}\right)` and where :math:`\left(b^{2n+1}\right)^{-1}` represents a B-spline inverse sequence.

We now propose a few lines of code that first create a random spline :math:`f_{0}` of specified period :math:`K_{0},` degree :math:`n_{0},` and delay :math:`\delta x_{0},` and then display its :math:`M`-magnified version :math:`f_{M}.` The spline :math:`g_{M}` of arbitrary degree :math:`n` and arbitrary delay :math:`\delta x` that best represents :math:`f_{M}` is then determined and displayed. We validate optimality by verifying that a quantity that vanishes in theory does so numerically, too, first through the explicit numerical estimate of an integral, then through the convolution form.

..  admonition:: Jupyter Lab notebook

    `Upscaling and projection of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_up_proj.ipynb&mode=single-document>`_

----

Downscaling
-----------

TODO

----

Rescaling
---------

TODO