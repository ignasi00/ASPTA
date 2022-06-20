# try to import framework <- if pythonpath is set, the package is installed, symbolic link on build or whatever makes the package available, then it is ok
# except add previous folder to pythonpath (wishing it will work)

try:
    import frameworks
except:
    import os
    import sys

    _CURRENT_PATH = os.path.dirname(__file__)
    _PARENT_PATH = os.path.join(_CURRENT_PATH, os.pardir)
    _PARENT_PATH = os.path.abspath(_PARENT_PATH)
    
    sys.path.append(PROJECT_ROOT)

    import frameworks
