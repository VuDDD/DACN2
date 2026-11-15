"""
Windows Application Control (WDAC/AppLocker) Pure-Python Cython Compatibility Hook.
Intercepts imports of SQLAlchemy Cython modules (*_cy) and redirects to their
official pure-Python (.py) siblings when unsigned .pyd binaries are restricted by OS policy.
"""
import sys
import os
import importlib.util

class PurePythonCythonFinder:
    def find_spec(self, fullname, path=None, target=None):
        if 'sqlalchemy' in fullname and fullname.endswith('_cy') and path:
            mod_name = fullname.split('.')[-1]
            for p in path:
                py_path = os.path.join(p, f"{mod_name}.py")
                if os.path.exists(py_path):
                    return importlib.util.spec_from_file_location(fullname, py_path)
        return None

# Register at highest priority in sys.meta_path if not already registered
if not any(isinstance(finder, PurePythonCythonFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, PurePythonCythonFinder())

