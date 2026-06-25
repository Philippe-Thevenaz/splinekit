Fourier Transform of Periodic Splines
=====================================

How to access the Fourier coefficients of a one-dimensional spline.

----

Preamble
--------

Functions
^^^^^^^^^

Let :math:`f:{\mathbb{X}}\rightarrow{\mathbb{Y}},x\mapsto f(x)` be some mapping from the *domain* :math:`{\mathbb{X}}` to the *codomain* :math:`{\mathbb{Y}}.` The domain is the set of all admissible elements being mapped, and the codomain is a superset of all the elements being mapped to. Typical relevant examples are the real functions

..  math::
    f:{\mathbb{R}}\rightarrow{\mathbb{R}}

and the vector-to-vector mappings

..  math::
    {\mathbf{f}}:{\mathbb{R}}^{K}\rightarrow{\mathbb{R}}^{K}.

It is sometimes convenient to describe the mapping by a computational recipe. For instance, a quadratic monomial would be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=x^{2}.` Observe that the codomain is announced here to be the whole of :math:`{\mathbb{R}},` while indeed only nonnegative values are mapped to. The exact set of elements being mapped to is called the *image* of the mapping; it is a subset of the codomain. For a quadratic monomial, this image would be :math:`{\mathbb{R}}_{\geq0}.`

Transforms
^^^^^^^^^^

With real functions, the domain and the codomain are made of real numbers. But nothing prevents us to let the elements of the domain and the codomain be themselves mappings. This would lead to

..  math::
    {\mathcal{T}}:\left({\mathbb{X}}_{1}\rightarrow{\mathbb{Y}}_{1}\right)\rightarrow\left({\mathbb{X}}_{2}\rightarrow{\mathbb{Y}}_{2}\right),

with domain :math:`{\mathbb{X}}=\left({\mathbb{X}}_{1}\rightarrow{\mathbb{Y}}_{1}\right)` and codomain :math:`{\mathbb{Y}}=\left({\mathbb{X}}_{2}\rightarrow{\mathbb{Y}}_{2}\right).` In such cases, :math:`{\mathcal{T}}` is called a *transform*. The cases of interest to us are those transforms that map complex functions to complex functions.

Various Brands of Fourier Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A transform that got a lot of traction is the *continuous-time Fourier transform* described by the recipe

..  math::
    {\mathcal{F}}:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),f\mapsto{\mathcal{F}}\{f\}=\int_{{\mathbb{R}}}\,f(x)\,{\mathrm{e}}^{-{\mathrm{j}}\,\left(\cdot\right)\,x}\,{\mathrm{d}}x.

It proves to be an invaluable tool to make theoretical forays in the analysis of signals and systems, in statistics, and in many other disciplines.

By nature, however, :math:`{\mathcal{F}}` is handling objects defined in the continuum, whereas experimental data are always discrete and finitely supported. Consequently, another tool has been designed to retain the flavor of the continuous-time Fourier transform, but in the finite-dimensional discrete world; it is called the *discrete-time Fourier transform*, even if this discrete version is not really a transform but truly a vector-to-vector function. It is expressed as

..  math::
    {\mathbf{F}}:{\mathbb{C}}^{K}\rightarrow{\mathbb{C}}^{K},{\mathbf{x}}\mapsto{\mathbf{F}}({\mathbf{x}})=\left(\sum_{k=0}^{K-1}\,x[k]\,{\mathrm{e}}^{-{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\right)_{\nu=0}^{K-1}

which, for suitable :math:`{\mathbf{W}},` is nothing but the matrix multiplication :math:`{\mathbf{F}}({\mathbf{x}})={\mathbf{W}}\,{\mathbf{x}}` in disguise. A great part of the success of the discrete-time Fourier transform is due to the existence of a clever organization of the matrix multiplication that leads to its efficient computation.

There exists a third brand of the Fourier transform, called a *Fourier series*. It maps a complex periodic function of positive integer period :math:`K\in{\mathbb{N}}+1` to a series of complex coefficients, according to

..  math::
    F:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{Z}}\rightarrow{\mathbb{C}}\right),f\mapsto F\{f\}=\left(\frac{1}{K}\,\int_{0}^{K}\,f(x)\,{\mathrm{e}}^{-{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x}\,{\mathrm{d}}x\right)_{\nu\in{\mathbb{Z}}}.

Inverses
^^^^^^^^

All three versions of the Fourier transform (continuous-time, discrete-time, and series) are invertible, so that there exist inverse transforms

..  math::
    {\mathcal{F}}^{-1}:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),\hat{f}\mapsto{\mathcal{F}}^{-1}\{\hat{f}\}=\frac{1}{2\,\pi}\,\int_{{\mathbb{R}}}\,\hat{f}(\omega)\,{\mathrm{e}}^{{\mathrm{j}}\,\omega\,\left(\cdot\right)}\,{\mathrm{d}}\omega

..  math::
    {\mathbf{F}}^{-1}:{\mathbb{C}}^{K}\rightarrow{\mathbb{C}}^{K},\hat{{\mathbf{x}}}\mapsto{\mathbf{F}}^{-1}(\hat{{\mathbf{x}}})=\frac{1}{K}\,\left(\sum_{\nu=0}^{K-1}\,\hat{x}[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\right)_{k=0}^{K-1}

..  math::
    F^{-1}:\left({\mathbb{Z}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),\left(c[\nu]\right)_{\nu\in{\mathbb{Z}}}\mapsto\left(x\mapsto\sum_{\nu\in{\mathbb{Z}}}\,c[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x}\right).

However, some of the expressions of the direct or inverse transforms involve integration processes or infinite sums, which call for mathematical care about the proper treatment of converge. Even in the absence of convergence in the classic sense, there are advanced mathematical theories that typically allow one to handle cases where :math:`\int_{{\mathbb{R}}}\,\left({\mathcal{F}}^{-1}\{{\mathcal{F}}\{f\}\}(x)-f(x)\right)^{2}\,{\mathrm{d}}x=0` without that :math:`{\mathcal{F}}^{-1}\{{\mathcal{F}}\{f\}\}(x)=f(x)` holds true for all :math:`x\in{\mathbb{R}}.`

----

Truncated Fourier Series
------------------------







..  admonition:: Jupyter Lab notebook

    `Truncated Fourier series <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/fourier/spline_fourier.ipynb&mode=single-document>`_
