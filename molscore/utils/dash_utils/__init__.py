from __future__ import print_function as _
from __future__ import absolute_import
import os as _os
import sys as _sys
import json

import dash as _dash

from molscore.utils.dash_utils.utils import pdb_to_dict, sdf_to_dict, write_style_dict
from molscore.utils.dash_utils.dash_Molecule3DViewer import Molecule3dViewer

dash_monitor_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dash_monitor.py')
dash_monitor3D_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'dash_monitor_3D.py')

if not hasattr(_dash, 'development'):
    print('Dash was not successfully imported. '
          'Make sure you don\'t have a file '
          'named \n"utils.py" in your current directory.', file=_sys.stderr)
    _sys.exit(1)

_basepath = _os.path.dirname(__file__)
_filepath = _os.path.abspath(_os.path.join(_basepath, 'package-info.json'))
with open(_filepath) as f:
    package = json.load(f)

package_name = package['name'].replace(' ', '_').replace('-', '_')
__version__ = package['version']

_current_path = _os.path.dirname(_os.path.abspath(__file__))
_components = _dash.development.component_loader.load_components(
    _os.path.join(_current_path, 'metadata.json'),
    package_name
)

_this_module = _sys.modules[__name__]

async_resources = [
    'alignment',
    'circos',
    'ideogram',
    'moleculeviewer2',
    'moleculeviewer3',
    'needle',
    'nglmoleculeviewer',
    'onco',
    'sequence',
    'speck'
]

_js_dist = []

_js_dist.extend([{
        'relative_package_path': 'async-{}.js'.format(async_resource),
        'external_url': (
            'https://unpkg.com/dash-bio@{}'
            '/' + package_name + '/async-{}.js'
        ).format(__version__, async_resource),
        'namespace': 'dash_bio',
        'async': True
    } for async_resource in async_resources])

_js_dist.extend([{
        'relative_package_path': 'async-{}.js.map'.format(async_resource),
        'external_url': (
            'https://unpkg.com/dash-bio@{}'
            '/' + package_name + '/async-{}.js.map'
        ).format(__version__, async_resource),
        'namespace': 'dash_bio',
        'dynamic': True
    } for async_resource in async_resources])

_js_dist.extend([
    {
        'relative_package_path': 'bundle.js',
        'external_url': (
            'https://unpkg.com/dash-bio@{}'
            '/' + package_name + '/bundle.js'
        ).format(__version__),
        'namespace': package_name
    }
])

_css_dist = []


for _component in _components:
    setattr(_this_module, _component.__name__, _component)
    setattr(_component, '_js_dist', _js_dist)
    setattr(_component, '_css_dist', _css_dist)