#!/bin/bash

devDir="/Users/pthevena/Documents/Programmation/Python/splinekit/dev/PurePython"
splinekitDir="/Users/pthevena/Documents/Programmation/Python/splinekit"
cp $devDir/release.yml $splinekitDir/.github/workflows/release.yml
cp $devDir/pyproject.toml $splinekitDir/pyproject.toml
cp $devDir/spline_padding.py $splinekitDir/src/splinekit/spline_padding.py

# Edit splinekit-jupyterlite/environment.yml