from pathlib import Path
import platform
import numpy as np
import joblib
import os
import plotly.io as pio
pio.renderers.default = "browser"

##############################################################################
## Settings ###################################################################
###############################################################################

# get the absolute path to this script location
script_location = Path(__file__).resolve().parent

# automatically set input directories and other variables depending on
# the machine that is currently used.
name_of_this_machine = platform.node()

if name_of_this_machine.startswith('zislrds'):
    print("This machine is a virtual machine")
    data_dir = '/zi/flstorage/group_csp/in_house_datasets/sosense/' # use /zi/flstorage
    cache_dir = '/data/cache_directories/project_sosense' # use /data as cache directory
    n_jobs = 12

elif name_of_this_machine.startswith('zislhcn'):
    print("This machine is a slurm node")
    data_dir = '/zi/flstorage/group_csp/in_house_datasets/sosense/' # use /zi/flstorage
    cache_dir = os.path.join('/data/',os.environ['SLURM_JOB_ID']) # use the slurm nodes /data as cache directory
    n_jobs = -5

else:
    print("This machine is a local machine")
    data_dir = os.path.join(script_location,'../../data/') # use data folder within the project folder
    cache_dir = os.path.join(script_location,'../../cache/') # use cache folder within the project folder
    n_jobs = 8

# always convert to absolute paths
data_dir = os.path.abspath(data_dir)
cache_dir = os.path.abspath(cache_dir)

print('Data directory:',data_dir)
print('Cache directory:',cache_dir)
print('\n\n')

# set a random number generator that you can use for all your analyses
rng = np.random.default_rng(42)

# set cache object
memory = joblib.Memory(cache_dir,verbose=0)