# [Built-in modules]
import os
import re
import sys
import shutil
import time, datetime
import math  as myMath

# [3rd party modules]
import cv2
import argparse
import numpy                 as np
import xml.etree.ElementTree as ET

from scipy.special    import lambertw
from scipy.integrate  import quad

# [Local modules]
from modules.constants import Constants


class MyVideo:
    """
        Description:
        ----------

        Arguments:
        ----------

        Returns:
        ----------
    """
    def __init__( self, vid_dir = None, height = 2880, width = 1700, fps = 60 ):
        self.height    = height
        self.width     = width
        self.vid_dir   = vid_dir if not None else "."
        self.fps       = fps

        fourcc         = cv2.VideoWriter_fourcc( *'MP4V' )
        self.outVideo  = cv2.VideoWriter( self.vid_dir + "/video.mp4", fourcc, fps, ( height, width ) )

    def write( self, myViewer ):
        data = myViewer.read_pixels( self.height, self.width, depth = False )   # Get the pixel from the render screen
        self.outVideo.write( np.flip( data, axis = 0 ) )

    def release( self ):
        self.outVideo.release()

def snake2camel( s ):
    """
        Switch string s from snake_form_naming to CamelCase
    """

    return ''.join( word.title() for word in s.split( '_' ) )

def camel2snake( s ):
    """
        Switch string s from CamelCase to snake_form_naming
        [REF] https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    re.sub( r'(?<!^)(?=[A-Z])', '_', s ).lower()

def args_cleanup( args, s ):
    """
        Clean-up the substring s for keys in args

        Arguments
        ---------

            args: The dictionary to be parsed
            s   : Substring to be discarded. e.g. s = '--', then "--record" --> "record"

    """
    if not isinstance( args, dict ) or not isinstance( s, str ):
        raise ValueError( "Wrong input type. args should be type dict and s should be type str. {0:} and {1:} are rather given".format(
                                                                                            type( args ), type( str ) ) )

    for old_key in list( args ) :
        new_key = old_key.replace( s, '' )
        args[ new_key ] = args.pop( old_key )

    return args

def quaternion2euler( quatVec ):                                                # Inputting quaternion matrix and outputing the yaw, pitch, roll of the euler angle.
    """
        Converting a R4 quaternion vector (w, x, y, z) to Euler Angle (Roll, Pitch, Yaw)

        [ARGUMENTS]
            [NAME]          [TYPE]        [DESCRIPTION]
            (1) quatVec     List          The quaternion vector, ordered in w, x, y and z

        [OUTPUT]
            [NAME]                   [TYPE]        [DESCRIPTION]
            (1) yaw, pitch, roll                   The euler angles of the given quaternion vector.

        [DESCRIPTION]
        This code is directly from the following reference
        [REF] https://computergraphics.stackexchange.com/questions/8195/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr

    """

    if len( quatVec ) != 4:
        raise ValueError( "Wrong size of input argument. Given size is [{0:d}] while it should be 4".format(
                                                                    len( quatVec ) ) )

    w, x, y ,z  = quatVec[:]

    t0     =       + 2.0 * ( w * x + y * z )
    t1     = + 1.0 - 2.0 * ( x * x + y * y )
    roll   = myMath.atan2( t0, t1 )

    t2     = + 2.0 * ( w * y - z * x )
    t2     = + 1.0 if t2 > +1.0 else t2
    t2     = - 1.0 if t2 < -1.0 else t2
    pitch  = myMath.asin( t2 )

    t3     =       + 2.0 * ( w * z + x * y )
    t4     = + 1.0 - 2.0 * ( y * y + z * z )
    yaw    = myMath.atan2( t3, t4 )

    return yaw, pitch, roll

def str2bool( s ):
    """

        Description:
        ----------
        Converting an input string to a boolean

        Arguments:
        ----------
            [NAME]          [TYPE]        [DESCRIPTION]
            (1) s           dict, str     The string which

        Returns:
        ----------
            True/False depending on the given input strin gv

    """
    if isinstance( s, dict ):
        for key, _ in s.items():
            s[ key ] = str2bool( s[ key ] )
    else:
        return v.lower() in ( "yes", "true", "t", "1" )

def str2float( s ):
    """

        Description:
        ----------
        Converting an input string to a float arraay

        Arguments:
        ----------
            [NAME]          [TYPE]        [DESCRIPTION]
            (1) s           str           The string which will be parsed to float array

        Returns:
        ----------
            The parsed float array

    """
    if not isinstance( s, str ):
        raise ValueError( "Input argument should be string, but {} is given".format( type( s ) ) )

    return [ float( i ) for i in re.findall( r"[-+]?\d*\.\d+|[-+]?\d+", s ) ]

def my_mkdir(  ):

    dir  = "results/"
    dir  += datetime.datetime.now().strftime( "%Y%m%d_%H%M%S" )                 # Appending the date when this directory is called.
    if not os.path.exists( dir ):                                               # If directory not exist
        os.makedirs( dir )                                                      # Make the directory

    return dir + "/"

def my_rmdir( dir ):

    if not isinstance( dir, str ):
        raise ValueError( "Input directory should be a str, {} is given".format( type ( dir ) ) )

    try:
        shutil.rmtree( dir  )
    except:
        print( "{0:s} Doesn't exist, hence cannot remove the directory".format( dir ) )

    print( "Erasing Directory [{0:s}]".format( dir ) )

def my_print( **kwargs ):
    """
        Description:
        ----------
            ** double asterisk means giving the argument as dictionary
            By using double asterisk "kwargs" as input argument,

        Arguments:
        ----------

        Returns:
        ----------
    """

    prec = kwargs[ "prec" ] if "prec" in kwargs else 5
    f    = kwargs[ "file" ] if "file" in kwargs else sys.stdout                 # If there is a keyword called "file" then use that as our standard output

    tmpMaxLen = len( max( kwargs.keys( ), key = len ) )                         # Getting the maximum length of a string list

    for args in kwargs:

        if 'file' == args.lower( ):
            # Ignore the file's value, since it should not be added to the "output.txt" log file.
            continue


        print( "[{1:{0}s}]:".format( tmpMaxLen, args ), end = ' ', file = f )   # Printing out the name of the array
                                                                                # {1:{0}s} Enables to set a variable as format length.
        tmpData = kwargs[ args ]

        if   isinstance( tmpData, ( float, int ) ):
            tmpPrint = "{2:{1}.{0}f}".format( prec, prec + 2, tmpData )

        elif isinstance( tmpData, list  ):
            tmpPrint = np.array2string( np.array( tmpData ).flatten(), precision = prec, separator = ',' )

        elif isinstance( tmpData, np.ndarray  ):
            tmpPrint = np.array2string( tmpData.flatten()            , precision = prec, separator = ',' )

        elif isinstance( tmpData, str   ):
            tmpPrint = tmpData

        elif tmpData is None:
            tmpPrint = "None"

        else:
            raise ValueError( "CHECK INPUT")

        print( tmpPrint, file = f )

def add_whip_model( N = 10, L = 1, M = 1, rho = 1150, k = 0.05, b = 0.005, taper_type = None, model_name2add = "2D_model.xml", target_type = 1 ):
    """
        Description:
        ----------
        Generating the XML Whip model for the MuJoCo Simulation

        From the given whip parameters, this function imports the template xml model file under "../model" directory, and auto-generate an XML Model File.

        2DOF open-chain linkage model was used for the upper-limb model. Everything distal from the wrist joint was excluded from the model.
        [CBC] [MOSES NAH] A 3DOF linkage model, including a wrist, could be implemented in the near future.

        The geometrical and inertial parameters for each limb segment were obtained from a computational model by Hatze.

        [REF]: Hatze, H. (1980). A mathematical model for the computational determination of parameter values of anthropomorphic segments. Journal of Biomechanics, 13(10):833–843.


    Arguments:
    ----------
        N: total number of nodes for the N-Node whip model                                              [DEFAULT] 10    [-]
        M: the mass of the entire model                                                                 [DEFAULT] 0.1   [kg]
        L: the length of the entire model                                                               [DEFAULT] 0.1   [m]
        rho: the rho of the material used for the whip                                                  [DEFAULT] 1150  [kg/m^3]
        k: coefficient of linear torsional spring  assigned at the node joint of the whip model.        [DEFAULT] 0.05  [N-m/rad]
        b: coefficient of linear torsional damping assigned at the node joint of the whip model.        [DEFAULT] 0.005 [N-m-s/rad]
        taper_type: type of (mass) tapering for the whip model (e.g. linear or exponential).            [DEFAULT] None

    Returns:
    ----------
        model_name: name of the xml model

    """

    model_dir = Constants.MODEL_DIR

    N     = int( N ) if not isinstance( N, int ) else N
    PI    = np.pi
    l     = L / N                                                               # The length of a sub-model
    m     = M / N                                                               # (Ideal) point mass suspended at the end of a sub-model.
    nNode = N                                                                   # Saving the number of nodes separately for "generate_sub_model" function


    # If model file to add whip model is given, then check if there exist ".xml" at the end.
    model_name2add   += ".xml" if model_name2add[ -4: ] != ".xml" else ""
    my_model_tree   = ET.parse( model_dir + "/" + model_name2add )              # Importing/parsing the mujoco mjcf xml template file to customize and generate the N-Node whip model.
    root            = my_model_tree.getroot()


    model_name_added = model_name2add[ :-4 ]                                    # Name of the .xml model file which the whip is added
    model_name_added += "_w_N" + str( N )                                       # Name modification: Add the node number of the whip model with the _w_ character
    model_name_added += "_" + taper_type if taper_type is not None else ""      # Name modification: Add the taper type of the whip model.

    root.attrib[ 'model' ] = model_name_added                                   # Modify the name of the xml model file

    def clean_up( elem, level = 0 ):                                             # Basic Clean up function for the xml model file, adding newline and indentation.
                                                                                # [REF] https://norwied.wordpress.com/2013/08/27/307/
        i = "\n" + level * "   "

        if len( elem ):
            if not elem.text or not elem.text.strip( ):
                elem.text = i + "  "

            if not elem.tail or not elem.tail.strip( ):
                elem.tail = i

            for elem in elem:
                clean_up( elem, level + 1 )

            if not elem.tail or not elem.tail.strip( ):
                elem.tail = i
        else:
            if level and ( not elem.tail or not elem.tail.strip( ) ):
                elem.tail = i

        # [BACKUP] [MOSES NAH] For readability, you can also add comments to the xml file via ET.Comment("comment")
        # elem.insert( 0, ET.Comment("comment") )

    V = M / rho

    def create_linear_regression( L, M, rho ):                              # Creates a linear function for tapering given the initial parameters
        r = np.sqrt( 3 * V / ( PI * L ) )                                       # 1) Visualize a cone
        slope = - r / L

        def linear( x ):                                                        # 2) Create a linear function based off the cone
            return slope * x + r                                                # y = mx + b

        def linear_squared( x ):                                                # 3) Create the square of a linear function based off the cone
            return ( slope * x + r ) ** 2                                       # y = (mx + b)^2

        return ( linear, linear_squared )                                       # 4) Returns a tuple, including (linear function, square of the linear function)

    def create_exponential_regression( L, M, rho ):                                        # Creates an exponential decay function for tapering given the initial parameters
        handle_size = 3                                                         # Assume V of exponential function = V of cylinder / handleInverse
        a = np.sqrt( handle_size * V / ( PI * L ) )                             # Estimate r handle

        W = lambertw( a ** 2 * np.exp( -a ** 2 * L * PI / V ) * L * PI )
        b = 1 / 2 * ( W / L + PI * a ** 2 / V )                                 # Derived from Vs of solids of revolution

        def exponential( x ):
            return a * np.exp( -b * x )                                         # y = ae^(-bx)

        def exponential_squared( x ):
            return ( a * np.exp( -b * x ) ) ** 2                                # y = (ae^(-bx))^2

        return ( exponential, exponential_squared )

    def create_real_linear( ):                                                  # Creates a linear function based on experimental data ( gathered by The Action Lab at Northeastern )
        def real_linear( x ):
            return ( -6.7574 * x + 22.224 ) / 2 / 1000

        def real_linear_squared( x ):
            return ( ( -6.7574 * x + 22.224 ) / 2 / 1000 ) ** 2

        return ( real_linear, real_linear_squared )

    def create_real_quad( ):                                                    # Creates an quadratic function based on experimental data ( gathered by The Action Lab at Northeastern )
        def real_quad( x ):
            return ( 2.8289 * x ** 2 - 11.284 * x + 23.355 ) / 2 / 1000

        def real_quad_squared( x ):
            return ( ( 2.8289 * x ** 2 - 11.284 * x + 23.355 ) / 2 / 1000) ** 2

        return ( real_quad, real_quad_squared )

    if   taper_type == 'linear_regression':
        equations = create_linear_regression( L, M, rho )

    elif taper_type == 'exponential_regression':
        equations = create_exponential_regression( L, M, rho )

    elif taper_type == 'real_whip_linear':
        equations = create_real_linear( )

        I = quad( equations[ 1 ], 0, L )
        V = PI * I[ 0 ]
        rho = M / V

    elif taper_type == 'real_whip_quad':
        equations = create_real_quad( )

        I = quad( equations[ 1 ], 0, L )
        V = PI * I[ 0 ]
        rho = M / V
    else:
        pass

    def find_size( node, l, equation ):                                         # Finds the size of a tapered sub-model using Riemann sums
        n = node - 0.5 # midpoint
        x = n * l
        return equation( x )

    def find_mass( node, l, equation, rho ):                                    # Finds the (ideal) point mass of a tapered sub-model using volumes of solids of revolution
        end_b   = node * l
        start_b = end_b - l
        I       = quad( equation, start_b, end_b )                              # Integration
        V       = PI * I[ 0 ]
        return rho * V

    def find_inertia( node, l, equation, mass ):
        # Specifically for the tapered whip model.
        r1 = equation( ( node-1 ) * l )
        r2 = equation( (  node  ) * l )
        m  = mass

        inert = 1/10 * m * l ** 2 * ( r1 * r1 + 3 * r1 * r2 + 6 * r2 * r2 )/( r1*r1 + r1*r2 + r2*r2 ) \
              + 3/20 * m * ( r1 ** 4 + r1 ** 3 * r2 + r1 ** 2 * r2 ** 2 + r1 * r2 ** 3 + r2 ** 4 ) / ( r1*r1 + r1*r2 + r2*r2 )

        return inert

    def generate_sub_model( N, parent_node ):                                   # parent_node, this is for recursion.
        if taper_type is None:
            size = m / 2
            mass = m
            k_calc = k
            b_calc = b

        else:
            # If taper exist
            # size = find_size( N, l, equations[ 0 ] ) / 4
            size = find_size( N, l, equations[ 0 ] ) / 32
            size = np.real( size )

            mass = find_mass( N, l, equations[ 1 ], rho )
            mass_ref = find_mass( int( nNode/2 ), l, equations[ 1 ], rho )

            inert     = find_inertia( N, l, equations[ 0 ], mass )
            inert_ref = find_inertia( int( nNode/2 ), l, equations[ 0 ], mass_ref )

            k_calc = 0.242 * inert / inert_ref
            b_calc = 0.092 * inert / inert_ref


        joint_elem = ET.Element( "joint", attrib = {
                                        "name" : "node" + str( N ) + "JointY"  ,
                                        "type" : "hinge"                       ,
                                        "axis" : "0 -1 0"                      ,
                                        "pos"  : "0 0 0"                       ,
                                   "springref" : "0"                           ,
                                   "stiffness" : "0" if N == 1 else str( k_calc )   ,# If N == 1, then the joint corresponds to the connection of the upper-limb and whip model. Setting the stiffness and damping value as zero for the connection.
                                     "damping" : "0" if N == 1 else str( b_calc ) } )

        joint_elem2 = ET.Element( "joint", attrib = {
                                        "name" : "node" + str( N ) + "JointX"  ,
                                        "type" : "hinge"                       ,
                                        "axis" : "1 0 0"                       ,
                                        "pos"  : "0 0 0"                       ,
                                   "springref" : "0"                           ,
                                   "stiffness" : "0" if N == 1 else str( k_calc )   ,# If N == 1, then the joint corresponds to the connection of the upper-limb and whip model. Setting the stiffness and damping value as zero for the connection.
                                     "damping" : "0" if N == 1 else str( b_calc ) } )

        geom_elem1 = ET.Element( "geom",  attrib = {                            # Massless cylinder of the sub-model.
                                        "type" : "cylinder"                        ,
                                      "fromto" : "0 0 0 0 0 {0:7.4f}".format( -l ) ,
                                        "size" : "{0:8.7f}".format( size )         ,
                                        "mass" : "0"                             } )

        geom_elem2 = ET.Element( "geom",  attrib = {                            # (Ideal) point mass of the sub-model.
                                        "name" : "geom" + str( N ) if N != nNode else "geom" + str( N ) + "Tip"  ,
                                        "type" : "sphere"                      ,
                                   "material"  : "pointMassColor"              ,
                                        "pos"  : "0 0 {0:7.4f}".format( -l )   ,
                                        "size" : "{0:8.7f}".format( 2 * size ) ,
                                        "mass" : "{0:8.7f}".format( mass     ) } )

        if "3D" in model_name2add:
            parent_node.append( joint_elem2 )                                   # Attaching the joint element       to the parent node

        parent_node.append( joint_elem )                                        # Attaching the joint element       to the parent node
        parent_node.append( geom_elem1 )                                        # Attaching the cylinder main model to the parent node
        parent_node.append( geom_elem2 )                                        # Attaching the point-mass model    to the parent node

        if N is not nNode:                                                      # Using recursive method to concatenate the sub-models
            node_elem = ET.Element( "body",  attrib = {
                                            "name" : "node" + str( N + 1 )         ,
                                           "euler" : "0 0 0"                       ,
                                            "pos"  : "0 0 {0:7.4f}".format( -l ) } )
            parent_node.append( node_elem )
            generate_sub_model( N + 1, node_elem )
        else:                                                                   # If the submodel concatenation reach the end, simply return the function.
            return

    for sub_elem in root.iter( 'geom' ):                                         # Setting the position of the target
        try:
            if sub_elem.attrib[ 'name' ] == "target":                            # If the geometry is "target"
                tmp = N * l + 0.294 + 0.291 + 0.01

                if   target_type == 1:
                    targetPosition = np.array( [ tmp,0,0 ] )

                elif target_type == 2:
                    targetPosition = np.array( [ tmp/np.sqrt( 2 ), tmp/np.sqrt( 2 ),0 ] )

                elif target_type == 3:
                    targetPosition = 0.5 * np.array( [ tmp/2, tmp/2, tmp/np.sqrt( 2 ) ] )

                sub_elem.attrib[ 'pos' ] = ' '.join( "{0:7.4f}".format( n ) for n in targetPosition )

        except KeyError as e:                                                   # If sub_elem does not have a "name", simply ignore the exception
            continue

        except:                                                                 # Other exceptions
            print( sys.exc_info( )[ 0 ] )
    parent_map =  dict( (c, p) for p in root.iter() for c in p )                # Manually constructing the child-parent node relationship, this will become handy for appending the whip model.
                                                                                # [REF] https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node

    # Append the whip model to the template file
    for elem in root.iter( 'geom' ):
        if 'geom_EE' in elem.attrib.values( ):
            # Within the geom elements, if there exist a geom name with 'geom_EE'
            # Get the parent_element (p_elem)
            p_elem = parent_map[ elem ]
            c_elem = ET.Element( "body",  attrib = {
                                            "name" : "node1"  ,
                                           "euler" : "0 0 0"  ,
                                            "pos"  : "0 0 0" if "2" == model_name2add[ 0 ] or ("3" and "type4" in model_name2add ) else "0 0 -0.291" } )
            p_elem.append( c_elem )
            p_elem.append( ET.Comment(' === [ N, L, M, k, b, rho ] = [{0:d},{1:8.6f},{2:8.6f},{3:8.6f},{4:8.6f},{5:8.6f}] '.format(
                                                                          N, L, M, k, b, rho ) ) )
            generate_sub_model( 1, c_elem )

    clean_up( root )                                                            # Cleaning up the XML file

    model_name_added += ".xml"

    my_model_tree.write( model_dir + model_name_added,
                         encoding = "utf-8" ,
                  xml_declaration = False   )                                   # Writing the actual xml model file.

    return model_name_added

if __name__ == '__main__':
    pass
