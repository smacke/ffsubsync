# -*- coding: utf-8 -*- 
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


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
    return make_version_tuple(__version__) < make_version_tuple(latest_vstr)
