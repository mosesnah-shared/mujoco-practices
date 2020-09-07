# [Built-in modules]

# [3rd party modules]
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

try:
    import mujoco_py as mjPy
except ImportError as e:
    raise error.DependencyNotInstalled( "{}. (HINT: you need to install mujoco_py, \
                                             and also perform the setup instructions here: \
                                             https://github.com/openai/mujoco-py/.)".format( e ) )

# [Local modules]

class Controller( ):
    """

    """


    def __init__( self, mjModel, mjData ):
        """

        """
        self.mjModel        = mjModel
        self.mjData         = mjData
        self.ctrl_par_names = None


    def set_ctrl_par( self, **kwargs ):
        """
            Setting the control parameters
        """
        if kwargs is not None:
            for args in kwargs:
                if args in self.ctrl_par_names:
                    setattr( self, args, kwargs[ args ] )
                else:
                    pass

    def input_calc( self, start_time, current_time ):
        raise NotImplementedError                                               # Adding this NotImplementedError will force the child class to override parent's methods.

class PID_Controller( Controller ):
    """
        Description:
        ----------
            Class for an Impedance Controller

    """

    def __init__( self, mjModel, mjData, Kp = 1, Kd = 0.3, Ki = 0, ref_type = 0, input_type = "step" ):
        """
            Description:
            ----------
                Class for an Impedance Controller

            Arguments:
            ----------

            Returns:
            ----------
        """

        super().__init__( mjModel, mjData )

        self.act_names      = mjModel.actuator_names
        self.n_act          = len( mjModel.actuator_names )
        self.idx_act        = np.arange( 0, self.n_act )

        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.ref_type   = ref_type
        self.input_type = input_type

    def input_calc( self, start_time, current_time ):
        """

        """
        p  = self.mjData.qpos[ 0 ]
        dp = self.mjData.qvel[ 0 ]

        if   current_time >= start_time:                                        # If time greater than startTime

            if self.input_type == "step":   # If input is unit step function.
                f_input = self.Kp * ( 0.1 - p ) + self.Kd * ( - dp )

        else:
            f_input = 0

        return self.mjData.ctrl, self.idx_act, f_input
