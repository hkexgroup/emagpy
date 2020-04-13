#!/bin/bash
. ../../pyenv/bin/activate
pyinstaller -y ui-dir.spec
cd dist
zip -r ui.zip ui
cd ..
mv dist/ui.zip ui.zip
pyinstaller -y splashScreen-exe.spec
mv ui.zip dist/EMagPy-linux.zip
mv dist/EMagPy-launch dist/EMagPy-linux


