class Constants:
    PROJECT_NAME            = '[MOSES NAH] Mujoco Practices'
    VERSION                 = '1.0.0'
    AUTHOR_GITHUB           = 'mosesnah-shared'
    AUTHOR_FULL_NAME        = 'Moses C. Nah'
    COLLABORATORS_GITHUB    = "TO BE ADDED"
    COLLABORATORS_FULL_NAME = "TO BE ADDED"
    DESCRIPTION             = "Python + mujoco-py code for running practice simulations"
    URL                     = 'https://github.com/mosesnah-shared/mujoco-practices',
    AUTHOR_EMAIL            = 'mosesnah@mit.edu', 'mosesnah@naver.com'
    COLLABORATORS_EMAIL     = 'TO BE ADDED'

    # =============================================================== #
    # Constant variables for running the simulation

    MODEL_DIR      = "models/"                                                  # The model directory which contains all the xml model files.
    CONTROLLER_DIR = "controllers/"                                             # The directory which saves all the controller classes (e.g., PD, PID etc.)
    SAVE_DIR       = "results/"                                                 # The directory which saves all the simulation results

    PREC     = 4                                                                # Precision for the floating point print
    PRINT_LW = 8000                                                             # Line Width for the nlopt print.
