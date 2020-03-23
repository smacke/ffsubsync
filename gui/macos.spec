# -*- mode: python -*-

import os
import platform
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix = 'gooey/images')
a = Analysis(['./subsync-gui.py'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             )
pyz = PYZ(a.pure)

options = [('u', None, 'OPTION'), ('u', None, 'OPTION'), ('u', None, 'OPTION')]

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          options,
          gooey_languages, # Add them in to collected files
          gooey_images, # Same here.
          name='Subsync',
          debug=False,
          strip=None,
          upx=True,
          console=False,
          windowed=True,
          icon=os.path.join(gooey_root, 'images', 'program_icon.ico'))


if platform.system() == 'Darwin':
    info_plist = {'addition_prop': 'additional_value'}
    app = BUNDLE(exe,
                 name='Subsync.app',
                 bundle_identifier=None,
                 info_plist=info_plist
                )
