Change of Basis
===============

Conversion from one basis to another.

----

Parametric Function
-------------------

Suppose we are given a sequence :math:`\left(c[k]\right)_{k\in{\mathbb{Z}}}` of coefficients, along with some real function :math:`\varphi:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto\varphi(x)` in the role of a basis. Then, we may attempt to sum the weighted integer shifts of :math:`\varphi` to build the function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\varphi(x-k).` Our attempt is successful if the conditions of convergence and unicity of the parameters are met jointly.

*   *Convergence* the sum found in the expression of the parametric function :math:`f` converges for any sequence :math:`c` of parameters.

*   *Unicity* the function :math:`f_{1},` built out of a first sequence :math:`c_{1}` of coefficients, and the function :math:`f_{2},` built out of the second sequence :math:`c_{2},` are such that there exists at least one :math:`x\in{\mathbb{R}}` with :math:`f_{1}(x)\ne f_{2}(x)` whenever there exists at least one :math:`k\in{\mathbb{Z}}` with :math:`c_{1}[k]\ne c_{2}[k].`

Convergence and Unicity
-----------------------

The convergence of a one-sided infinite sum :math:`\sum_{k=1}^{\infty}\,s_{k}` of terms :math:`s_{k}` is notoriously fiddly. Many notions have been called for to capture what is meant by the word "convergence." For instance, a qualified convergence can be conditional, unconditional, pointwise, almost uniform, almost everywhere, uniform, absolute, uniform-absolute (which, confusingly enough, is not the same as being at the same time uniform and absolute), among many other qualifiers. And then, there are quasi-convergent sums.

In our case, the form of the function :math:`f` features the double-sided infinite sum :math:`\sum_{k=-\infty}^{\infty}\,s[k]` of the terms :math:`s[k]=c[k]\,\varphi(x-k).` It has been found sensible to part such doubled-sided sums in pairs of one-sided sums to study their one-sided convergence; but even then, the parting can be made in several ways.

*   The principal-value approach considers either the limit :math:`s[0]+\lim_{K\rightarrow\infty}\sum_{k=1}^{K}\,\left(s[-k]+s[k]\right)` or the limit :math:`\lim_{K\rightarrow\infty}\sum_{k=-K}^{K}\,s[k],` which may not converge unconditionally, meaning that convergence may or may not happen, depending on the ordering of the terms.
*   The form :math:`\lim_{\left(K_{1},K_{2}\right)\rightarrow\left(\infty,\infty\right)}\sum_{k=-K_{1}}^{K_{2}}\,s[k],` with independence between :math:`K_{1}` and :math:`K_{2},` has links with uniform convergence.
*   The index-based parting considers the limits :math:`\lim_{K_{1}\rightarrow\infty}\sum_{k_{1}=-K_{1}}^{-1}\,s[k_{1}]+s[0]+\lim_{K_{2}\rightarrow\infty}\sum_{k_{2}=1}^{K_{2}}\,s[k_{2}].` It has links with uniform convergence, too.
*   The value-based parting :math:`\left(-\sup\{\left(-\sum_{k\in{\mathbb{A}}}\,s[k]\right)\in{\mathbb{R}}:{\mathbb{A}}\subseteq\{k\in{\mathbb{Z}}:s[k]<0\}\}\right.`:math:`\left.\mbox{}+\sup\{\left(\sum_{k\in{\mathbb{B}}}\,s[k]\right)\in{\mathbb{R}}:{\mathbb{B}}\subseteq\{k\in{\mathbb{Z}}:s[k]\geq0\}\}\right)` involves the notion of supremum over a set and has links with absolute convergence.

The various ways to part a double-sided infinite sum into a pair of one-sided infinite sums, combined with the various notions in which such one-sided sums may—or may not—converge, make for a complicated state of affairs. However, when :math:`\varphi` is a polynomial B-spline :math:`\beta^{n}` of nonnegative integer degree :math:`n\in{\mathbb{Z}},` the situation is crystal-clear.

1.  Because of the finite support of :math:`\beta^{n},` the sum is finite and all worries about the convergence of double-sided infinite sums evaporate.
2.  Because :math:`\beta^{n}` is nonnegative and upper-bounded, the quantity :math:`f(x)` is well-defined for all :math:`x\in{\mathbb{R}}.`
3.  Because :math:`\beta^{n}` turns out to be a so-called Riesz basis, unicity is guaranteed, at least for all square-summable sequences :math:`c.`

Multiplicity of the Representations
-----------------------------------

The representation of :math:`f` is not unique. For instance, one can create a new sequence of coefficients by multiplying every given coefficient by some non-vanishing number; at the same time, one divides the function :math:`\varphi` by the same number to create a new basis. Together, the new coefficients and the new basis will do the same job as the given ones. Another example in which we thwart unicity is to create yet another new sequence of coefficients by the addition of a constant integer offset to every index of the given sequence, and to compensate it by the creation of yet another basis as an offset-shifted version of the given one.

The two proposed examples are trivial. A less trivial approach arises for those bases :math:`\varphi_{1}` that admit being themselves expressed as a sum of weighted and integer-shifted bases :math:`\varphi_{2},` as in

..  math::
        \varphi_{1}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto \varphi_{1}(x)=\sum_{k'\in{\mathbb{Z}}}\,h[k']\,\varphi_{2}(x-k'),

where the sequence :math:`h` of weights relates the two bases. If, in addition, the technical conditions for the interchange of the order of summations allows it, then we can write that

..  math::
    \begin{array}{rcl}
    f(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{1}[k]\,\varphi_{1}(x-k)\\
    &=&\sum_{k\in{\mathbb{Z}}}\,c_{1}[k]\,\sum_{k'\in{\mathbb{Z}}}\,h[k']\,\varphi_{2}(x-k-k')\\
    &=&\sum_{k'\in{\mathbb{Z}}}\,\left(\sum_{k\in{\mathbb{Z}}}\,c_{1}[k]\,h[k'-k]\right)\,\varphi_{2}(x-k').
    \end{array}

Assuming that the discrete convolution :math:`c_{2}[k']=\sum_{k\in{\mathbb{Z}}}\,c_{1}[k]\,h[k'-k]` is well-defined for all :math:`k'\in{\mathbb{Z}},` we have thus established a new representation of :math:`f` in the basis :math:`\varphi_{2},` with coefficients :math:`c_{2}.` We shall refer to this mechanism as a change of basis.







The ``splinekit`` library comes equipped with paddings that guarantee the well-defined interpolation of finite-length data.

..  admonition:: Jupyter Lab notebook

    `Padded splines <https://splinekit.github.io/splinekit-jupyterlite/notebooks/index.html?path=padding_data.ipynb>`_

