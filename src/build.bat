..\..\pyenv\Scripts\activate.bat && ^
pyinstaller -y ui-dir.spec && ^
cd dist\ui && ^
7z a -aoa -r ..\..\ui.zip * && ^
cd ..\.. && ^
move /y dist\ui.zip ui.zip && ^
pyinstaller -y splashScreen-exe.spec && ^
move /y dist\EMagPy-launch.exe dist\EMagPy-windows.exe && ^
move /y ui.zip dist\EMagPy-windows.zip
