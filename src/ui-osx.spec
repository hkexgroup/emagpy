# -*- mode: python -*-

block_cipher = None

datas=[
       ('./logo.png', '.'),
       ('./loadingLogo.png', '.'),
       ('./emagpy/j0_120.txt', './emagpy'),
       ('./emagpy/j1_140.txt', './emagpy'),
       ('./emagpy/hankelpts.txt', './emagpy'),
       ('./emagpy/hankelwts0.txt', './emagpy'),
       ('./emagpy/hankelwts1.txt', './emagpy'),
       ('./emagpy/pcs.csv', './emagpy')
      ]

def extra_datas(mydir):
    def rec_glob(p, files):
        import os
        import glob
        for d in glob.glob(p):
            if os.path.isfile(d):
                files.append(d)
            rec_glob("%s/*" % d, files)
    files = []
    rec_glob("%s/*" % mydir, files)
    extra_datas = []
    for f in files:
        extra_datas.append((f, os.path.dirname(os.path.join('emagpy',f))))

    return extra_datas

datas += extra_datas('examples')
datas += extra_datas('image')


a = Analysis(['ui.py'],
             pathex=[],
             binaries=[],
             datas=datas,
             hiddenimports=['pkg_resources.py2_warn'],
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
