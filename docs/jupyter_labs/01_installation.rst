Installation
============

Setup
-----

The setup instructions of this *Setup* section need be performed once only and will serve for as many *Session* and *Cleanup* sessions as you will.

We recommend that you work in a virtual environment. Every command is given from a terminal; from the terminal, navigate to a folder/directory of your choice (preferentially a fresh one). Create and activate the virtual environment with

(on Unix)::

    python -m venv splinekit-env
    source splinekit-env/bin/activate

(on macOS)::

    python3 -m venv splinekit-env
    source splinekit-env/bin/activate

(on Windows)::

    python -m venv splinekit-env
    .splinekit-env/Scripts/Activate

The additional notebooks are distributed in compressed format. Their execution relies on the availability of a workable ``Jupyter Lab`` environment and of an installed version of the ``splinekit`` library, along with its dependencies. To install every dependency in one shot, launch a terminal and issue the commands::

    pip install pip --upgrade
    pip install numpy scipy sympy matplotlib jupyterlab ipywidgets splinekit

Session
-------

An activated virtual environment is characterized by a terminal prompt that starts with the string *(splinekit-env)*. If this is not the case, do activate the virtual environment with

(on Unix and MacOS)::

    source splinekit-env/bin/activate

(on Windows)::

    .splinekit-env/Scripts/Activate


Download one of the additional notebooks and store its uncompressed version within the folder/directory where your virtual environment resides. On many systems, decompression is achieved by a double-click on the file. If you prefer to use the terminal, the command is::

    gunzip jupyter_lab_notebook.ipynb.gz

The Jupyter Lab sessions are hosted in a (compatible) browser of your choice. To open a session, from the terminal issue the command::

    jupyter lab

The notebook you downloaded and decompressed ought to be accessible from Jupyter's file-browser panel. One executes one of its cells by selecting it with a click (anywhere within the cell) and by hitting::

    <Shift-CR>

Cleanup
-------

To kill the Jupyter Lab process, with the terminal being the active window, hit::

    <CTRL-C>

To deactivate the virtual environment, type::

    deactivate
