# -*- mode: python -*-

block_cipher = None


a = Analysis(['ui.py'],
             pathex=[],
             binaries=[],
             datas=[('./emagpy/test/*','./emagpy/test'),
                    ('./logo.png', '.'),
                    ('./emagpy/j0_120.txt', './emagpy'),
                    ('./emagpy/j1_140.txt', './emagpy'),
                    ('./emagpy/hankelpts.txt', './emagpy'),
                    ('./emagpy/hankelwts0.txt', './emagpy'),
                    ('./emagpy/hankelwts1.txt', './emagpy'),
                    ('./loadingLogo.png', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='ui')
app = BUNDLE(coll,
             name='EMagPy.app',
             icon='logo.icns',
             bundle_identifier=None,
             info_plist={'NSHighResolutionCapable': 'True'})
