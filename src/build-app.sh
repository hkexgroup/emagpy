source ../../pyenv/bin/activate
pyinstaller -y ui-osx.spec
mv ./dist/EMagPy.app ./macdmg/EMagPy.app
hdiutil create ./dist/EMagPy-macos.dmg -srcfolder macdmg -ov
mv ./macdmg/EMagPy.app ./dist/EMagPy-macos.app
