# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# mast3r submodule import
# --------------------------------------------------------

import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
MASt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../mast3r'))
MASt3R_LIB_PATH = path.join(MASt3R_REPO_PATH, 'mast3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(MASt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MASt3R_REPO_PATH)
else:
    raise ImportError(f"mast3r is not initialized, could not find: {MASt3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")