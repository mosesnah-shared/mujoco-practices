"""

[PYTHON NAMING CONVENTION]
    module_name, package_name, ClassName, method_name, ExceptionName, function_name,
    GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name,
    local_var_name.


"""

import sys, os
import cv2
import re
import pprint
import numpy as np
import time, datetime


from modules.utils        import ( my_print, quaternion2euler, MyVideo, str2float)
from modules.constants    import Constants

try:
    import mujoco_py as mjPy
except ImportError as e:
    raise error.DependencyNotInstalled( "{}. (HINT: you need to install mujoco_py, \
                                             and also perform the setup instructions here: \
                                             https://github.com/openai/mujoco-py/.)".format( e ) )


class Simulation( ):
    """
        Class for the mujoco-simulation

        [INHERITANCE]

        [DESCRIPTION]


        [NOTE]
            All of the model files are saved in "models" directory, and we are using "relative directory"
            to generate and find the .xml model file. Hence do not change of "model directory" variable within this

    """

    MODEL_DIR     = Constants.MODEL_DIR
    SAVE_DIR      = Constants.SAVE_DIR

    current_time      = 0
    controller        = None                                                    # Control input function
    controller_inputs = None

    t_step            = None                                                    # Time step of the simulation
    n_steps           = 0                                                       # Number of total steps of the simulation

    def __init__( self, model_name = None, arg_parse = None ):
        """
            Default constructor of THIS class

            [ARGUMENTS]
                [NAME]             [TYPE]        [DESCRIPTION]
                (1) model_name     string        The xml model file name for running the MuJoCo simulation.
                (2) is_visualized  boolean       Turn ON/OFF the mjViewer (visualizer) of the simulation. This flag is useful when optimizing a simulation.
                (3) arg_parse      dictionary    Dictionary which contains all the arguments given to the main `run.py` script.
        """

        if model_name is None:
            self.mjModel  = None
            self.mjSim    = None
            self.mjData   = None
            self.mjViewer = None
            self.args     = arg_parse

            my_print( WARNING = "MODEL FILE NOT GIVEN, PLEASE INPUT XML MODEL FILE WITH `attach_model` MEMBER FUNCTION" )

        else:
            # If model_name is given, then check if there exist ".xml" at the end, if not, append
            model_name = model_name + ".xml" if model_name[ -4: ] != ".xml" else model_name
            self.model_name = model_name

            # Based on the model_name, construct the simulation.
            self.mjModel  = mjPy.load_model_from_path( self.MODEL_DIR + model_name )      #  Loading xml model as and save it as "model"
            self.mjSim    = mjPy.MjSim( self.mjModel )                                    # Construct the simulation environment and save it as "sim"
            self.mjData   = self.mjSim.data                                               # Construct the basic MuJoCo data and save it as "mjData"
            self.mjViewer = mjPy.MjViewerBasic( self.mjSim )                              # Construct the basic MuJoCo viewer and save it as "myViewer"
            self.args     = arg_parse

            # Saving the default simulation variables
            self.fps         = 60                                               # Frames per second for the mujoco render
            self.dt          = self.mjModel.opt.timestep                        # Time step of the simulation [sec]
            self.sim_step    = 0                                                # Number of steps of the simulation, in integer [-]
            self.update_rate = round( 1 / self.dt / self.fps )                  # 1/dt = number of steps N for 1 second simulaiton, dividing this with frames-per-second (fps) gives us the frame step to be updated.
            self.g           = self.mjModel.opt.gravity                         # Calling the gravity vector of the simulation environment

            # Saving additional model parameters for multiple purposes
            self.act_names      = self.mjModel.actuator_names
            self.geom_names     = self.mjModel.geom_names
            self.idx_geom_names = [ self.mjModel._geom_name2id[ name ] for name in self.geom_names  ]

            self.run_time   = float( self.args[ 'runTime'   ] )                 # Run time of the total simulation
            self.start_time = float( self.args[ 'startTime' ] )                 # Start time of the movements

    def attach_model( self, model_name ):

        if self.mjModel is not None:
            my_print( WARNING = "MODEL FILE EXIST! OVERWRITTING THE WHOLE MUJOCO FILE" )

        self.__init__( model_name )

    def attach_controller( self, controller ):
        """
            Attaching the controller object for running the simulation.

            For detailed controller description, please check "input_ctrls.py"

        """

        ctrl_name = self.controller.__class__.__name__                          # Getting the name of the controller. The controller names are indicated in "input_ctrls.py"
        my_print( Controller = ctrl_name )
        self.controller = controller


    def run( self ):
        """
            Running a single simulation.

            [INPUT]
                [VAR NAME]             [TYPE]     [DESCRIPTION]
                (1) run_time           float      The whole run time of the simulation.
                (2) ctrl_start_time    float
        """

        # Check if mjModel or mjSim is empty and raise error
        if self.mjModel is None or self.mjSim is None:
            raise ValueError( "mjModel and mjSim is Empty! Add it before running simulation"  )

        # Warn the user if input and output function is empty
        if self.controller is None:
            raise ValueError( "CONTROLLER NOT ATTACHED TO SIMULATION. \
                               PLEASE REFER TO METHOD 'attach_output_function' and 'attach_controller' " )


        if self.args[ 'recordVideo' ]:
            vid = MyVideo( fps = self.fps,
                       vid_dir = self.args[ 'saveDir' ] )                       # If args doesn't have saveDir attribute, save vid_dir as None

        if self.args[ 'saveData' ]:
            file = open( self.args[ 'saveDir' ] + "data_log.txt", "w+" )

        # Setting the camera position for the simulation
        # [camParameters]: [  0.17051,  0.21554, -0.82914,  2.78528,-30.68421,162.42105 ]
        # [camParameters]: [ -0.10325,  0.     , -2.51498,  7.278  ,-45.     , 90.     ]
        if self.args[ 'camPos' ] is not None:

            tmp = str2float( self.args[ 'camPos' ] )

            self.mjViewer.cam.lookat[ 0:3 ] = tmp[ 0 : 3 ]
            self.mjViewer.cam.distance      = tmp[ 3 ]
            self.mjViewer.cam.elevation     = tmp[ 4 ]
            self.mjViewer.cam.azimuth       = tmp[ 5 ]


        while self.current_time <= self.run_time:

            # [BACKUP] If we want to save all the details of the simulation
            # if self.args[ 'saveData' ]:
            #     my_print( currentTime       = self.current_time,
            #               geomXYZVelocities = self.mjData.geom_xvelp[ self.idx_geom_names ],
            #               file              = file )

            if self.sim_step % self.update_rate == 0:

                self.mjViewer.render( )                                         # Render the simulation

                if self.args[ 'verbose' ]:
                    my_print( camParameters = [ self.mjViewer.cam.lookat[ 0 ], self.mjViewer.cam.lookat[ 1 ], self.mjViewer.cam.lookat[ 2 ],
                                                self.mjViewer.cam.distance,    self.mjViewer.cam.elevation,   self.mjViewer.cam.azimuth ] )

                if self.args[ 'recordVideo' ]:
                    vid.write( self.mjViewer )

                # if self.args[ 'saveData' ]:
                    # my_print( currentTime       = self.current_time,
                    #           jointAngleActual  = self.mjData.qpos[ : ],
                    #           jointVelActual    = self.mjData.qvel[ : ],
                    #           geomXYZPositions  = self.mjData.geom_xpos[ self.idx_geom_names ],
                    #           geomXYZVelocities = self.mjData.geom_xvelp[ self.idx_geom_names ],
                    #           jacobian          = self.mjData.get_geom_jacp(  "geom_EE"    ).reshape( 3, -1 ),
                    #           file              = file )

            # xvel stands for positional(cartesian) velocity in world frame[REF] https://github.com/openai/mujoco-py/issues/255

            # [input controller]
            # input_ref: The data array that are aimed to be inputted (e.g., qpos, qvel, qctrl etc.)
            # input_idx: The specific index of input_ref data array that should be inputted
            # input:     The actual input value which is inputted to input_ref
            input_ref, input_idx, input = self.controller.input_calc( self.start_time, self.current_time )
            input_ref[ input_idx ] = input

            # self.ctrl_array = np.vstack( ( self.ctrl_array, input  ) ) if self.ctrl_array is not None else input                                                           # Saving the whole control input the ctrl_array
            # self.q_array    = np.vstack( ( self.q_array, self.mjData.qpos[ : self.ctrl_input.n_act ] ) ) if self.q_array is not None else self.mjData.qpos[ : self.ctrl_input.n_act ]    # The q position of the model
            # self.t_vec      = np.append( self.t_vec,  self.current_time ) if self.t_vec   is not None else self.current_time


            self.mjSim.step( )                                                  # Single step update
            if( self.is_sim_unstable() ):                                       # Check if simulation is stable

                print( "[WARNING] UNSTABLE SIMULATION, HALTED AT {0:f} for at {1:f}".format( self.current_time, self.run_time )  )
                self.output_array = [ np.nan ]

                break

            self.current_time = self.mjData.time                                # Update the current_time variable of the simulation

            if self.sim_step % self.update_rate == 0:

                if self.args[ 'saveData' ]:
                    # Saving all the necessary datas for the simulation
                    my_print(  inputVal = input,
                              outputVal = output_val if self.output_func is not None else 0,
                                   file = file )

            self.sim_step += 1

            # [BACKUP] If we want to save all the details of the simulation
            # if self.args[ 'saveData' ]:
            #     # Saving all the necessary datas for the simulation
            #     my_print(  inputVal = input,
            #               outputVal = output_val if self.output_func is not None else 0,
            #                    file = file )

        if self.args[ 'recordVideo' ]:
            vid.release( )                                                      # If simulation is finished, wrap-up the video file.

        if self.args[ 'saveData' ]:
            file.close()

        return self.ctrl_array, self.output_array

    def save_simulation_data( self, dir ):
        """
            Save all the details of the controller parameters, inputs and output of the simulation
        """

        if dir is not None and dir[ -1 ] != "/":                                # Quick Check of whether result_dir has backslash "/" at the end
            dir += "/"                                                          # Append the backslash

        f = open( dir + "simulation_details.txt", "w+" )
        pprint.pprint( self.ctrl_input.__dict__, f )                            # Using pretty-print (pprint) to flush out the data in a much readable format
        print( self.args                , file = f )                            # Flushing out all the arguments detail.
        f.close()

    def is_sim_unstable( self ):

        thres = 5 * 10 ** 6

        if ( max( np.absolute( self.mjData.qpos ) ) > thres ) or \
           ( max( np.absolute( self.mjData.qvel ) ) > thres ) or \
           ( max( np.absolute( self.mjData.qacc ) ) > thres ):
           return True

        else:
           return False


    def reset( self ):
        """
            Reseting the mujoco simulation
        """
        self.current_time      = 0
        self.sim_step          = 0
        self.controller_inputs = None
        self.mjSim.reset( )
