#!/bin/bash
#convert logo.png -alpha off -resize 256x256 -define icon:auto-resize="256,128,96,64,48,32,16" logo.ico
. ../../pyenv/bin/activate
pyinstaller -y ui-dir.spec
cd dist/ui
zip -r ../../ui.zip *
cd ../..
pyinstaller -y splashScreen-exe.spec
mv ui.zip dist/EMagPy-linux.zip
mv dist/EMagPy-launch dist/EMagPy-linux


