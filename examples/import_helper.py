import sys
import os

path_work = "/Users/jesu/Desktop/ESI/spatialize/src/python"
path_home = "/Users/jesu/Desktop/WORK/spatialize/src/python"

if os.path.exists(path_work):
    sys.path.insert(0, path_work)

elif os.path.exists(path_home):
    sys.path.insert(0, path_home)

else:
    spatialize_path = input("Ingresar path a repositorio spatialize: ")
    full_path = spatialize_path + "/src/python"
    sys.path.insert(0, path_work)