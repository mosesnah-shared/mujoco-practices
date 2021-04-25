"""

# ============================================================================= #
| Project:        [MuJoCo Practivces]
| Title:          Python + mujoco-py
| Author:         Moses C. Nah
| Email:          [Moses ] mosesnah@mit.edu
| Creation Date:  Monday, September 7th, 2020
# ============================================================================= #

# ============================================================================= #
| (0A) [DESCRIPTION]
|
|  - Python Script for running multiple models with its corresponding controllers.
|  - This will be useful for educational purpose.
|
# ============================================================================= #

# ============================================================================= #
| (0B) [KEYWORDS DEFINITION]
|       : type the following "keywords" for cases as...
|         - [BACKUP] [NAME]: Back-up code in case it's needed for the near future
|         - [TIP]: The reason why the following code was written.
|         - [TODO]: The part where modification is needed in the near future
# ============================================================================= #

# ============================================================================= #
| (0C) [PYTHON NAMING CONVENTION]
|       Our project will follow the python naming convention, [REF]: https://stackoverflow.com/a/8423697/13437196
|       ---------------------------------------------------------------------------------------------------------
|       module_name, package_name, ClassName, method_name, ExceptionName, function_name,
|       GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.
# ============================================================================= #

# ============================================================================= #
| (0D) [DOCOPT PARSE]
|      From now on, the written comments are specifically for "docopt" function.
|      [REF] http://docopt.org/
# ============================================================================= #

Usage:
    run.py [options]
    run.py -h | --help

Arguments:

Options:
    -h --help                  Showing the usage and options
    --version                  Show version
    -s --saveData              Saving the neccessary data from MuJoCo simulation as a txt file in the current directory
                               [default: False]
    -r --recordVideo           Record simulation video as a .mp4 file in the current directory
                               [default: False]
    --runTime=TIME             The total time of the simulation
                               [default: 5.0]
    --startTime=TIME           The start time of the movement, or controller
                               [default: 1.0]
    --modelName=NAME           Setting the xml model file name which will be used for the simulation.
                               The starting number of the xml model file indicates the type of simulation, hence the --modelName
                               [default: 1_mass_PD.xml]
    --camPos=STRING            Setting the Camera Position of the simulation.
                               default is None
    --quiet                    Print less text
                               [default: False]
    --verbose                  Print more text
                               [default: False]

Examples, try:
    python3 run.py --help
    python3 run.py --version
    python3 run.py --modelName="mass_PD.xml" --findCamPos
"""

# ============================================================================= #
# (0A) [IMPORT MODULES]
# Importing necessary modules + declaring basic configurations for running the whole mujoco simulator.

# [Built-in modules]
import sys
import os
import re
import argparse
import datetime
import shutil

# [3rd party modules]
import numpy       as np
import cv2

# [3rd party modules] - mujoco-py
try:
    import mujoco_py as mjPy
except ImportError as e:
    raise error.DependencyNotInstalled( "{}. (HINT: you need to install mujoco_py, \
                                             and also perform the setup instructions here: \
                                             https://github.com/openai/mujoco-py/.)".format( e ) )

from docopt  import docopt

# [3rd party modules] - pyPlot for graphs
import matplotlib.pyplot as plt
# import nevergrad   as ng  # [BACKUP] Needed for Optimization


# [Local modules]
from modules.constants    import Constants
from modules.controllers  import ( PID_Controller )
from modules.utils        import ( args_cleanup, my_print, my_mkdir, my_rmdir )
from modules.simulation   import Simulation
# from modules.output_funcs import (dist_from_tip2target, tip_velocity )
# from modules.input_ctrls  import ( ImpedanceController, Excitator, ForwardKinematics, PositionController )
# from modules.utils        import ( add_whip_model, my_print, my_mkdir, args_cleanup,
#                                    my_rmdir, str2float, camel2snake, snake2camel )


# ============================================================================= #

# ============================================================================= #
# (0B) [SYSTEM SETTINGS]
                                                                                # [Printing Format]
np.set_printoptions(  linewidth = Constants.PRINT_LW ,
                      suppress  = True               ,
                      precision = Constants.PREC   )                            # Setting the numpy print options, useful for printing out data with consistent pattern.

args = docopt( __doc__, version = Constants.VERSION )                           # Parsing the Argument
args = args_cleanup( args, '--' )                                               # Cleaning up the dictionary, discard prefix string '--' for the variables

if sys.version_info[ : 3 ] < ( 3, 0, 0 ):                                       # Simple version check of the python version. python3+ is recommended for this file.
    my_print( NOTIFICATION = " PYTHON3+ is recommended for this script " )

# If video needs to be recorded or data should be saved, then append 'saveDir' element to args dictionary
args[ 'saveDir' ] = my_mkdir( ) if args[ 'recordVideo' ] or args[ 'saveData' ] else None

assert not ( args[ 'quiet' ] and args[ 'verbose' ] )                            # If quiet and verbose are true at the same time, assert!
my_print( saveDir = args[ 'saveDir' ] )


# ============================================================================= #


# ============================================================================= #
def main( ):
    # ============================================================================= #

    model_name = args[ 'modelName' ]                                            # Calling Model
    my_print( modelName = model_name )

    mySim = Simulation(       model_name = model_name,
                               arg_parse = args )

    sim_type = model_name[ 0 ]                                                  # The first charater of model name is the index of simulation type.

    if "1" == sim_type:                                                         # 1: Simple Mass Simulation
        controller_object = PID_Controller( mySim.mjModel, mySim.mjData,
                                            Kp = 0, Kd = 0, Ki = 0, ref_type = 0)

    mySim.attach_controller( controller_object )
    mySim.run( )

    if args[ 'saveDir' ] is not None:
        mySim.save_simulation_data( args[ 'saveDir' ]  )
        shutil.copyfile( Constants.MODEL_DIR + model_name,
                         args[ 'saveDir' ]   + model_name )

    mySim.reset( )

    # ============================================================================= #

if __name__ == "__main__":

    try:
        main( )

    except KeyboardInterrupt:
        print( "Ctrl-C was inputted. Halting the program. ", end = ' ' )

        if args[ 'saveDir' ] is not None:
            my_rmdir( args[ 'saveDir' ] )

    except ( FileNotFoundError, IndexError, ValueError ) as e:
        print( e, end = ' ' )

        if args[ 'saveDir' ] is not None:
            my_rmdir( args[ 'saveDir' ] )
