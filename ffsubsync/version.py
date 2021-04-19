# -*- coding: future_annotations -*-
import os
from .constants import SUBSYNC_RESOURCES_ENV_MAGIC
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def get_version():
    if 'unknown' in __version__.lower():
        with open(os.path.join(os.environ[SUBSYNC_RESOURCES_ENV_MAGIC], '__version__')) as f:
            return f.read().strip()
    else:
        return __version__


def make_version_tuple(vstr=None):
    if vstr is None:
        vstr = __version__
    if vstr[0] == 'v':
        vstr = vstr[1:]
    components = []
    for component in vstr.split('+')[0].split('.'):
        try:
            components.append(int(component))
        except ValueError:
            break
    return tuple(components)


def update_available():
    import requests
    from requests.exceptions import Timeout
    from .constants import API_RELEASE_URL
    try:
        resp = requests.get(API_RELEASE_URL, timeout=1)
        latest_vstr = resp.json()['tag_name']
    except Timeout:
        return False
    except KeyError:
        return False
    if not resp.ok:
        return False
    return make_version_tuple(get_version()) < make_version_tuple(latest_vstr)
