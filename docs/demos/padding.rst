:orphan:

..  role:: raw-html(raw)
    :format: html

Padding
=======

How to extend a finite-length vector of data to a virtually infinite-length sequence.

----

Purpose
-------

We wish to build the real function

..  math::
        f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\varphi(x-k).

There, the sequence :math:`c` of coefficients is used to parameterize the function :math:`f` and gives us a good many degrees of freedom to shape it to our taste. Moreover, the basis :math:`\varphi` is assumed to have the desirable technical property of being a Riesz basis. Among other things, this ensures that the infinite sum found in the construction of :math:`f` is always well-behaved.

The adaptability of :math:`c` makes it a tool of choice to represent sampled data as the continuously defined function :math:`f.` Given a sequence :math:`y=\left(y[q]\right)_{q\in{\mathbb{Z}}}` of regularly indexed samples :math:`y[q]` for :math:`q\in{\mathbb{Z}},` the so-called interpolation condition leads to a procedure that yields a sequence :math:`c` such that :math:`f(q)=y[q].` Fortunately, the assumption of a Riesz basis guarantees the existence of :math:`c` for any :math:`y.` Furthermore, in case :math:`\varphi` is a B-spline, there exist very efficient algorithms to get :math:`c` out of :math:`y.`

Unfortunately, the theoretical derivations of the algorithms rely on :math:`y` being a *sequence*, which means that infinitely many samples are required. Now, one never has access to a sequence of samples in practice, only to a finite-dimensional *vector* :math:`{\mathbf{y}}=\left(y_{q}\right)_{q=1}^{K}\in{\mathbb{R}}^{K}` of samples. To take advantage of the theoretical derivations of the efficient algorithms, it is thus an unavoidable necessity that a procedure be engineered that converts the vector :math:`{\mathbf{y}}` into the sequence :math:`y.` We call padding the operation that consists in the engineering of the subsequences :math:`\left(y[k]\right)_{k\in{\mathbb{Z}}_{<0}}` to the left and :math:`\left(y[k]\right)_{k\in{\mathbb{Z}}_{\geq K}}` to the right of the provided :math:`\left(y[k]\right)_{k=0}^{K-1}.`

The engineering of :math:`{\mathbf{y}}\Rightarrow y` is application-dependent.

*   For instance, :math:`{\mathbf{y}}` could represent angular data, in which case one would have to cope with angular wrapping.
*   Or, :math:`{\mathbf{y}}` could represent intensity data, in which case one would have to discourage negative intensities in the continuously defined function :math:`f` being constructed through the steps :math:`{\mathbf{y}}\Rightarrow y\Rightarrow c\Rightarrow f.`
*   Or, one could pretend that all unobserved samples do vanish and take the special value :math:`0.`
*   Or, one could assume that the first observed sample :math:`y[0]` has indeed the same value as all unobserved samples that came before, while the last observed sample :math:`y[K-1]` has the same value as all unobserved samples that folllow.

Every method is valid. Which one is the best in the context of your application is only a matter of convenience.

Unicity
-------

In the sequel, we let :math:`\varphi` be a polynomial B-spline of nonnegative integer degree :math:`n,` which is a real function :math:`\beta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto\beta^{n}(x).` For any degree :math:`n\geq2` and for :math:`m\in[1\ldots\left\lfloor n/2\right\rfloor],` it is known that there exist real, negative numbers :math:`z_{n,m}` in the open interval :math:`(-1,0)` that satisfy the relation

..  math::
        \frac{1}{\sum_{k\in{\mathbb{Z}}}\,\beta^{n}(k)\,z_{n,m}^{-k}}\not\in{\mathbb{C}}.

These numbers are called the poles of the spline and are such that :math:`\sum_{k\in{\mathbb{Z}}}\,\beta^{n}(k)\,z_{n,m}^{-k}=0.` Because B-splines are even-symmetric, the pole reciprocals :math:`z_{n,m}^{-1}\in{\mathbb{R}}_{<-1}` satisfy the same relations.

Suppose now we have identified a sequence :math:`c` that verifies the interpolation condition :math:`y[q]=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(q-k).` Then, it turns out that the sequence :math:`c'[k]=c[k]+\sum_{m=1}^{\left\lfloor n/2\right\rfloor}\,\left(\lambda_{n,m}^{-}\,z_{n,m}^{-k}+\lambda_{n,m}^{+}\,z_{n,m}^{k}\right)` is also such that :math:`y[q]=\sum_{k\in{\mathbb{Z}}}\,c'[k]\,\beta^{n}(q-k),` for any choice of the constants :math:`\lambda_{n,m}^{-},\lambda_{n,m}^{+}\in{\mathbb{R}}.` One concludes that additional requirements are needed to make the interpolation task well-defined. The ``splinekit`` library comes equipped with paddings that guarantee the well-defined interpolation of finite-length data.

Available Paddings
------------------

The unobserved samples are, well, unobserved. Consequently, every strategy that assigns specific values to them is valid, but some are less practical than others. The ``splinekit`` library deals with paddings of low complexity; in particular, we focus on some for which the overall organization of :math:`y,` :math:`c,` and :math:`f` can be made consistent and solves the unicity issue. The seven paddings being considered are

*   Periodic
*   Narrow Mirror
*   Wide Mirror
*   Anti-Mirror
*   Nega-Periodic
*   Nega-Narrow Mirror
*   Nega-Wide Mirror

Except for the anti-mirror padding, all retained forms honor some sort of periodicity over :math:`y,` albeit the length of a period may differ from the number :math:`K` of observed samples. We arbitrarily impose that :math:`y` or :math:`c` follow the same sort of periodicity, which implies that :math:`f` is likewise periodic, too. Taking :math:`y` to be periodic, the trivial choice :math:`\lambda_{n,m}^{-}=0` and :math:`\lambda_{n,m}^{+}=0` is the only one that makes :math:`c'` periodic too, with :math:`c'=c.` Unicity is thus achieved. 

We give now a piece of code that illustrates visually the effect of the various paddings on random splines of a specified degree.

..  admonition:: Jupyter Lab notebook

    `Padded splines <https://splinekit.github.io/splinekit-jupyterlite/notebooks/index.html?path=padding_data.ipynb>`_

Periodic Padding
----------------

An easy, general-purpose padding approach is to engineer the sequence :math:`c` of coefficients to be :math:`K`-periodic. This implies that the sequence :math:`y` of samples has to be the straighforward :math:`K`-periodized version of the vector :math:`{\mathbf{y}}\in{\mathbb{R}}^{K}.` Ultimately, this also implies that the function :math:`f` is itself :math:`K`-periodic. In summary, under a periodic padding, the relations being satisfied for any :math:`k\in{\mathbb{Z}}` and any :math:`x\in{\mathbb{R}}` are

..  math::
        c[k]=c[k+K]\;\Rightarrow\;\left\{\begin{array}{rcl}y[k]&=&y[k+K]\\f(x)&=&f(x+K).\end{array}\right.

Algorithmic Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the context of a straightforward periodic padding, there are three major algorithmic approaches to the solution of the interpolation constraint :math:`f(q)=y[q]` for :math:`q\in[0\ldots K-1].`

1) The **linear-algebra approach** first establishes an explicit system of :math:`K` linear equations. The :math:`q`-th equation of the system would be :math:`y[q]=\sum_{k=0}^{K-1}\,\left(\sum_{p\in{\mathbb{Z}}}\,\varphi(q-p\,K-k)\right)\,c[k].` Tools of linear algebra would then be deployed to solve the system in terms of the unknown variables :math:`c[k].` The overall computational cost is :math:`{\mathcal{O}}(K^{3})` when general solvers are used, and the cost would be :math:`{\mathcal{O}}(K^{2})` for Toeplitz systems. For periodic paddings, the system is circulant and the overall computational cost reduces to :math:`{\mathcal{O}}(K\,\log K)` with Fourier-based techniques to solve linear-algebra inversion problems.

2) The **discrete-Fourier approach** is best described concisely in matrix notations. Let :math:`{\mathbf{F}}\in{\mathbb{C}}^{K\times K}` be the discrete Fourier transform, with the :math:`\nu`-th row and :math:`q`-th column entry given by :math:`{\mathrm{e}}^{-{\mathrm{j}}\,\left(\nu-1\right)\,\frac{2\,\pi}{K}\,\left(q-1\right)}.` Let the vector :math:`{\mathbf{c}}` represent one period of the periodic sequence :math:`c.` Moreover, let :math:`{\mathbf{\phi}}=(\sum_{p\in{\mathbb{Z}}}\,\varphi(p\,K+q))_{q=0}^{K-1}` be the data-independent vector that contains the samples (at the integers) of the periodized version of the basis :math:`\varphi.` Then, one has that :math:`{\mathbf{c}}={\mathbf{F}}^{-1}\,\left(\left({\mathbf{F}}\,{\mathbf{y}}\right)\oslash\left({\mathbf{F}}\,{\mathbf{\phi}}\right)\right),` where :math:`\oslash` is an element-wise division. In practice, the Fourier transformation and its inverse are implemented via the fast Fourier algorithm, in which case the overall computational cost is :math:`{\mathcal{O}}(K\,\log K).`

3) The **recursive-filtering approach** is the one followed in the ``splinekit`` library. It requires that the basis :math:`\varphi` has a finite support, is even-symmetric, and that the poles of the reciprocal of the :math:`z`-transform of its samples at the integers are real numbers. These properties are all satisfied by the polynomial B-splines of nonnegative integer degree :math:`n\in{\mathbb{N}}+2.`

Recursive Filtering
"""""""""""""""""""

Start the algorithm by letting :math:`{\mathbf{c}}={\mathbf{y}}.` Then, iteratively for every one of the poles :math:`z_{n,m}\in(-1,0)` indexed by :math:`m\in[1\ldots\left\lfloor n/2\right\rfloor]` and associated to the degree :math:`n,` apply the in-place recursive updates

..  math::
        \left\{\begin{array}{rcll}c[0]&\leftarrow&\frac{1}{1-z_{n,m}^{K}}\,\left(c[0]+\sum_{k=1}^{K-1}\,z_{n,m}^{k}\,c[K-k]\right)\\c[k]&\leftarrow&c[k]+z_{n,m}\,c[k-1],&k\in[1\ldots K-1]\\c[K-1]&\leftarrow&\frac{\left(1-z_{n,m}\right)^{2}}{1-z_{n,m}^{K}}\,\left(c[K-1]+\sum_{k=0}^{K-2}\,z_{n,m}^{k+1}\,c[k]\right)\\c[K-1-k]&\leftarrow&z_{n,m}\,c[K-k]+\left(1-z_{n,m}\right)^{2}\,c[K-1-k],&k\in[1\ldots K-1].\end{array}\right.

The overall computational cost is now :math:`{\mathcal{O}}(K\,\left\lfloor n/2\right\rfloor).` In practice, further acceleration can be achieved if the sums that appear in the recursive-update equations are truncated at that index :math:`k` where the term :math:`z_{n,m}^{k}` becomes negligible.

Experimental Performance
^^^^^^^^^^^^^^^^^^^^^^^^

The following experiment establishes some simple statistics over the gain in speed achieved by the recursive approach over the fast-Fourier-based one.

..  admonition:: Jupyter Lab notebook

    `Recursive vs Fourier <https://splinekit.github.io/splinekit-jupyterlite/notebooks/index.html?path=padding_speed.ipynb>`_

    *   The link above allows you to inspect the notebook. Unfortunately, running it from the browser is meaningless: the timings of the *installation-free* version are not representative because the Python kernel is WebAssembly-based and does not run natively.

    *   If you want to test for realistic timings, then you will have to first install in full the ``splinekit`` library. Only after that will you be able to launch the notebook either as a regular, full-fledged Jupyter Lab or as a module executed by the native Python kernel.

    *   The timings reported below correspond to those of the native execution.

..  hint::
    The notebook is available for download in compressed form from
    :download:`here <padding_speed.ipynb.gz>`. Decompression is achieved from the terminal with ``gunzip padding_speed.ipynb.gz``.

Results
"""""""

*   In the first table, the number :math:`K` of observed data are powers of two, a situation that is very much to the advantage of the discrete Fourier methods. For each such length :math:`K` and for each spline degree, we synthesize :math:`50` random vectors :math:`{\mathbf{y}}\in{\mathbb{Z}}^{K}` and let both the Fourier approach and the recursive approach determine the spline coefficients on the same data. We time those computations and report by how many times the recursive approach is faster relatively to the Fourier approach. For instance, the number :math:`2` would mean that the recursive approach would be :math:`200\%` times faster or, equivalently, that it runs twice as fast.
*   In the second table, we repeat the experiment, with the difference that the :math:`50` lengths are now chosen randomly in some range.

On a desktop computer of 2021, a typical resulting table is as follows.







:raw-html:`<TABLE frame="hsides" rules="groups">
<CAPTION>Accuracy and time in the critical regime</CAPTION>
<COLGROUP>
<COLGROUP span="2">
<COLGROUP span="2">
<COLGROUP span="2">
<COLGROUP span="2">
<TR><TH><TH colspan="2" align="center">&#160;Ground-Truth&#160;<TH colspan="2" align="center">&#160;Classic&#160;<TH colspan="2" align="center">&#160;De Boor&#160;<TH colspan="2" align="center">&#160;splinekit&#160;
<TR><TH>&#160;Degree&#160;<TH>&#160;SNR[dB]&#160;&#160;<TH>&#160;Time[s]&#160;<TH>&#160;SNR[dB]&#160;&#160;<TH>&#160;Time[s]&#160;<TH>&#160;SNR[dB]&#160;&#160;<TH>&#160;Time[s]&#160;<TH>&#160;SNR[dB]&#160;&#160;<TH>&#160;Time[s]&#160;
<TBODY>
<TR><TD align="right">0&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">3.1e-03&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">2.6e-04&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.6e-04&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.2e-04&#160;
<TR><TD align="right">1&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">7.2e-03&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">5.5e-04&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">3.2e-04&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.4e-03&#160;
<TR><TD align="right">2&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.0e-02&#160;<TD align="right">309.4&#160;&#160;<TD align="right">6.7e-04&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">6.3e-04&#160;<TD align="right">327.2&#160;&#160;<TD align="right">1.5e-03&#160;
<TR><TD align="right">3&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.4e-02&#160;<TD align="right">297.7&#160;&#160;<TD align="right">7.5e-04&#160;<TD align="right">320.9&#160;&#160;<TD align="right">1.3e-03&#160;<TD align="right">318.6&#160;&#160;<TD align="right">1.5e-03&#160;
<TR><TD align="right">4&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.8e-02&#160;<TD align="right">282.3&#160;&#160;<TD align="right">8.7e-04&#160;<TD align="right">319.3&#160;&#160;<TD align="right">2.5e-03&#160;<TD align="right">313.0&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">5&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">2.2e-02&#160;<TD align="right">262.1&#160;&#160;<TD align="right">9.9e-04&#160;<TD align="right">317.5&#160;&#160;<TD align="right">5.0e-03&#160;<TD align="right">310.6&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">6&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">2.7e-02&#160;<TD align="right">243.6&#160;&#160;<TD align="right">1.2e-03&#160;<TD align="right">317.2&#160;&#160;<TD align="right">9.9e-03&#160;<TD align="right">303.3&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">7&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">3.1e-02&#160;<TD align="right">224.9&#160;&#160;<TD align="right">1.4e-03&#160;<TD align="right">314.9&#160;&#160;<TD align="right">2.0e-02&#160;<TD align="right">302.2&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">8&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">3.7e-02&#160;<TD align="right">199.3&#160;&#160;<TD align="right">1.4e-03&#160;<TD align="right">315.2&#160;&#160;<TD align="right">4.0e-02&#160;<TD align="right">295.9&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">9&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">4.3e-02&#160;<TD align="right">171.9&#160;&#160;<TD align="right">1.6e-03&#160;<TD align="right">315.3&#160;&#160;<TD align="right">7.9e-02&#160;<TD align="right">294.5&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">10&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">5.0e-02&#160;<TD align="right">143.8&#160;&#160;<TD align="right">1.7e-03&#160;<TD align="right">313.5&#160;&#160;<TD align="right">1.6e-01&#160;<TD align="right">290.8&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">11&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">5.7e-02&#160;<TD align="right">120.2&#160;&#160;<TD align="right">1.8e-03&#160;<TD align="right">314.2&#160;&#160;<TD align="right">3.2e-01&#160;<TD align="right">287.6&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">12&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">6.5e-02&#160;<TD align="right">94.4&#160;&#160;<TD align="right">2.0e-03&#160;<TD align="right">312.2&#160;&#160;<TD align="right">6.4e-01&#160;<TD align="right">281.7&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">13&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">7.4e-02&#160;<TD align="right">72.6&#160;&#160;<TD align="right">2.3e-03&#160;<TD align="right">310.9&#160;&#160;<TD align="right">1.3e+00&#160;<TD align="right">284.1&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">14&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">8.3e-02&#160;<TD align="right">45.4&#160;&#160;<TD align="right">2.4e-03&#160;<TD align="right">310.3&#160;&#160;<TD align="right">2.5e+00&#160;<TD align="right">277.3&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">15&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">9.2e-02&#160;<TD align="right">15.9&#160;&#160;<TD align="right">2.6e-03&#160;<TD align="right">310.0&#160;&#160;<TD align="right">5.1e+00&#160;<TD align="right">273.7&#160;&#160;<TD align="right">1.3e-03&#160;
<TR><TD align="right">16&#160;<TD align="right">&#8734;&#160;&#160;<TD align="right">1.0e-01&#160;<TD align="right">-13.4&#160;&#160;<TD align="right">2.7e-03&#160;<TD align="right">312.6&#160;&#160;<TD align="right">1.0e+01&#160;<TD align="right">270.1&#160;&#160;<TD align="right">1.3e-03&#160;
</TABLE>`









Discussion
""""""""""

The conclusion of the experiments is unequivocal: The recursive approach is substantially faster than the fast-Fourier approach, at all lengths (except :math:`K=1`), ranges of lengths, and all degrees being investigated.

