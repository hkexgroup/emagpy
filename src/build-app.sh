source ../../pyenv/bin/activate
pyinstaller -y ui-osx.spec
mv ./dist/EMagPy.app ./macdmg/EMagPy.app
hdiutil create ./dist/EMagPy.dmg -srcfolder macdmg -ov
mv ./macdmg/EMagPy.app ./dist/EMagPy.app
