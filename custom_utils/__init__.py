# This command finds the module names that should be imported
# when the command `from [package] import *` is called.

##### THIS WAS COMMENTED TO FACILITATE DOCUMENTATION WITH SPHINX
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]