#!/bin/bash

devDir="/Users/pthevena/Documents/Programmation/Python/splinekit/dev/Jupyter"
docDir="/Users/pthevena/Documents/Programmation/Python/splinekit/docs/jupyter_labs"
rm $docDir/*.ipynb
cp $devDir/*.ipynb $docDir
