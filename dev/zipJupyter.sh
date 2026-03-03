#!/bin/bash

devDir="/Users/pthevena/Documents/Programmation/Python/splinekit/dev/Jupyter"
docDir="/Users/pthevena/Documents/Programmation/Python/splinekit/docs/jupyter_labs"
mkdir $devDir/zip/
cp $devDir/*.ipynb $devDir/zip/
find $devDir/zip/*.ipynb -exec zip {}.zip {} \;
rm $docDir/*.zip
mv $devDir/zip/*.zip $docDir/
rm $devDir/zip/*.ipynb
rmdir $devDir/zip/
