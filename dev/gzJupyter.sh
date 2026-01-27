#!/bin/bash

devDir="/Users/pthevena/Documents/Programmation/Python/splinekit/dev/Jupyter"
docDir="/Users/pthevena/Documents/Programmation/Python/splinekit/docs/jupyter_labs"
mkdir $devDir/gz/
cp $devDir/*.ipynb $devDir/gz/
gzip $devDir/gz/*
rm $docDir/*.gz
mv $devDir/gz/* $docDir/
rmdir $devDir/gz/
