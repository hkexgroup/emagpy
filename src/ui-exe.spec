## -*- mode: python -*-
import sys
sys.setrecursionlimit(1500)
block_cipher = None


a = Analysis(['ui.py'],
             pathex=[],
             binaries=[],
             datas=[('./emagpy/test/*','./emagpy/test'),
                    ('./logo.png', '.'),
                    ('./loadingLogo.png', '.'),
                    ('./logo.ico', '.'),
                    ('./emagpy/j0_120.txt', './emagpy'),
                    ('./emagpy/j1_140.txt', './emagpy'),
                    ('./emagpy/hankelpts.txt', './emagpy'),
                    ('./emagpy/hankelwts0.txt', './emagpy'),
                    ('./emagpy/hankelwts1.txt', './emagpy'),
                    ('./emagpy/pcs.csv', './emagpy')
                    ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='EMagPy',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False,
		  icon='logo.ico')

