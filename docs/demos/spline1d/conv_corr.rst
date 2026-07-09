Convolutions and Correlations
=============================

How to convolve and correlate two piecewise-polynomial periodic one-dimensional splines.

----

Definitions
-----------

Convolution
^^^^^^^^^^^

Let a first one-dimensional mapping be :math:`h:{\mathbb{R}}\rightarrow{\mathbb{C}}` and let a second one-dimensional mapping be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}.` Their *convolution* is notated :math:`h*f` and defined as the mapping

..  math::
    g:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto g(x)=\left(h*f\right)(x):=\int_{-\infty}^{\infty}\,h(y)\,f(x-y)\,{\mathrm{d}}y.

Caution must be taken that the integration process does converge. In particular, suppose that :math:`h` and :math:`f` are both periodic, with respective integer periods :math:`K_{1}` and :math:`K_{2}.` Now, if we assume that :math:`h*f` converges, then it is admissible to segment the corresponding improper integral as :math:`\left(h*f\right)(x)=\sum_{k\in{\mathbb{Z}}}\,\int_{0}^{K_{1}\,K_{2}}\,h(y+k\,K_{1}\,K_{2})\,f(x-y-k\,K_{1}\,K_{2})\,{\mathrm{d}}y.` The :math:`K_{1}` and :math:`K_{2}` periodicities imply that :math:`\left(h*f\right)(x)=\sum_{k\in{\mathbb{Z}}}\,\int_{0}^{K_{1}\,K_{2}}\,h(y)\,f(x-y)\,{\mathrm{d}}y.` The assumed convergence of :math:`h*f` implies that the definite integral takes some finite value :math:`F=\int_{0}^{K_{1}\,K_{2}}\,h(y)\,f(x-y)\,{\mathrm{d}}y\in{\mathbb{C}}.` But then, :math:`\sum_{k\in{\mathbb{Z}}}\,F` is not a finite quantity in general, which contradicts our assumption of convergence of the convolution of two periodic arbitrary functions. Thus, convolution must be defined defined differently in this case.

Periodic Convolution
^^^^^^^^^^^^^^^^^^^^

Let a one-dimensional :math:`K`-periodic mapping be :math:`h:{\mathbb{R}}\rightarrow{\mathbb{C}}` and let another one-dimensional :math:`K`-periodic mapping be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}},` with :math:`h(x)=h(x+K)` and :math:`f(x)=f(x+K)` for all :math:`x\in{\mathbb{R}}.` Their *periodic convolution* is notated :math:`h*f` and defined as the mapping

..  math::
    g:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto g(x)=\left(h*f\right)(x):=\int_{0}^{K}\,h(y)\,f(x-y)\,{\mathrm{d}}y.

Caution must be taken that the integration process does converge; for simplicity, we shall henceforth restrict ourselves to nonsingular mappings :math:`h` and :math:`f.` Note that the notation for a periodic convolution does not differ from that for a periodic convolution; both make use of the same operator :math:`*.` More often than not, however, the context allows one to easily distinguish between them.

Crosscorrelation
^^^^^^^^^^^^^^^^

The *crosscorrelation* of :math:`f_{1}:{\mathbb{R}}\rightarrow{\mathbb{C}}` and :math:`f_{2}:{\mathbb{R}}\rightarrow{\mathbb{C}}` is notated :math:`f_{1}\star f_{2}` and defined as the mapping

..  math::
    \chi:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto \chi(x)=\left(f_{1}\star f_{2}\right)(x):=\int_{-\infty}^{\infty}\,f_{1}^{*}(y)\,f_{2}(x+y)\,{\mathrm{d}}y.

When the two mappings are periodic of the same integer period :math:`K,` the same caveats apply as for their convolution. Their *periodic crosscorrelation* is the mapping

..  math::
    \chi:{\mathbb{R}}\rightarrow{\mathbb{C}},x\mapsto \chi(x)=\left(f_{1}\star f_{2}\right)(x):=\int_{0}^{K}\,f_{1}^{*}(y)\,f_{2}(x+y)\,{\mathrm{d}}y.

Note that one must pay attention to distinguish the convolution operator :math:`*` from the crosscorrelation operator :math:`\star.`

Autocorrelation
^^^^^^^^^^^^^^^

The crosscorrelation :math:`R_{ff}=f\star f` is called the autocorrelation of :math:`f.`

Periodic Normalized Crosscorrelation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Crosscorrelation measures how well :math:`f_{1}` matches :math:`f_{2}` in terms of their relative shift. Unfortunately, the measure is not easy to interpret because it depends on the units in which the functions are expressed. By contrast, the *normalized crosscorrelation* proposes a closely related measure that returns values in the interval :math:`[-1,1]` when the mappings :math:`f_{1}:{\mathbb{R}}\rightarrow{\mathbb{R}}` and :math:`f_{2}:{\mathbb{R}}\rightarrow{\mathbb{R}}` are real. For periodic functions, the *periodic normalized crosscorrelation* is defined as the mapping

..  math::
    \rho_{f_{1}f_{2}}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto \rho_{f_{1}f_{2}}(x)=\frac{1}{K}\,\int_{0}^{K}\,\frac{f_{1}(y)-E\{f_{1}\}}{\sqrt{{\mathrm{Var}}\{f_{1}\}}}\,\frac{f_{2}(x+y)}{\sqrt{{\mathrm{Var}}\{f_{2}\}}}\,{\mathrm{d}}y,

where the mean value is defined as :math:`{\mathrm{E}}\{f\}=\frac{1}{K}\,\int_{0}^{K}\,f(x)\,{\mathrm{d}}x` and where the variance is defined as :math:`{\mathrm{Var}}\{f\}=\frac{1}{K}\,\int_{0}^{K}\,\left(f(x)-{\mathrm{E}}\{f\}\right)^{2}\,{\mathrm{d}}x.`

It is easy to verify that the normalized crosscorrelation computed as above is indeed identical to two competing definitions, a first one in which no mean value is subtracted from :math:`f_{1}(y)` but the mean value :math:`E\{f_{2}\}` is subtracted from :math:`f_{2}(x+y),` and a second one in which appropriate mean values are subtracted at the same time from :math:`f_{1}(y)` and from :math:`f_{2}(x+y).`

When compared to crosscorrelation, one downside of the normalized crosscorrelation is that it is not well-defined when :math:`{\mathrm{Var}}\{f_{1}\}\,{\mathrm{Var}}\{f_{2}\}=0.`

Pearson Correlation Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Pearson correlation coefficient is the normalized crosscorrelation evaluated at the origin, as :math:`\rho_{f_{1}f_{2}}(0).`

----

Convolution and Filtering of Splines
------------------------------------

The periodic convolution between a periodic :math:`h` and a periodic :math:`f` is best understood as the (non-periodic) convolution between the windowed version :math:`h_{[0,K)}` of :math:`h` and the periodic :math:`f.` Indeed, if we let :math:`{\mathbf{[\![}}\cdot\,{\mathbf{]\!]}}` denote the Iverson bracket, then it holds that

..  math::
    \begin{array}{rcl}
    \left(h*f\right)(x)&=&\int_{0}^{K}\,h(y)\,f(x-y)\,{\mathrm{d}}y\\
    &=&\int_{-\infty}^{\infty}\,\underbrace{{\mathbf{[\![}}0\leq y<K\,{\mathbf{]\!]}}\,h(y)}_{h_{[0,K)}(y)}\,f(x-y)\,{\mathrm{d}}y.
    \end{array}

In other words, the periodic convolution :math:`h*f` is identical to the non-periodic, classic filtering of the (still periodic) signal :math:`f` by a filter of finite, non-periodic impulse response :math:`h_{[0,K)}.`

Let :math:`h` be a periodic polynomial spline of period :math:`K,` degree :math:`n_{1},` and delay :math:`\delta x_{1}.` In addition, let :math:`f` be a periodic polynomial spline of period :math:`K,` degree :math:`n_{2},` and delay :math:`\delta x_{2}.` Then, :math:`h*f` turns out to be a spline of same period :math:`K,` degree :math:`n_{1}+n_{2}+1,` and delay :math:`\delta x_{1}+\delta x_{2}.`

We now propose a few lines of code that create and display a pair of random splines and convolves them.

*   The first spline plays the role of the filter with continuously defined impulse response :math:`h.` Its coefficients are built as the absolute value of independent realizations of a random variable that follows a Cauchy distribution; moreover, the spline is normalized in such a way that it has a unit integral over one period. This promotes lowpass filters; if chance has it, one of the Cauchy coefficients will dwarf the other ones and the impulse response will have a pronounced bulge.
*   The second spline plays the role of the signal :math:`f` being filtered. Its coefficients are built as independent realizations of a random variable that follows a Gaussian distribution.

..  admonition:: Jupyter Lab notebook

    `Convolution of periodic splines <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/conv_corr/spline_convolve.ipynb&mode=single-document>`_

----

Crosscorrelation of Splines
---------------------------


