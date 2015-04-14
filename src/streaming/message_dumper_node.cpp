//ds std
#include <iostream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

//ds ROS
#include <ros/ros.h>

//ds fps_mapper includes
#include "txt_io/message_dumper_trigger.h"
#include "ros_wrappers/imu_interpolator.h"
#include "ros_wrappers/image_message_listener.h"
#include "ros_wrappers/imu_message_listener.h"

//ds command line parameters parser
boost::program_options::variables_map parseCommandLineParameters( int argc, char** argv );

int main( int argc, char **argv )
{
    //ds parse input
    boost::program_options::variables_map cVariablesMap = parseCommandLineParameters( argc, argv );

    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strTopicCamera_0 = "/thin_visensor_node/camera_0/image_raw";
    std::string strTopicCamera_1 = "/thin_visensor_node/camera_1/image_raw";
    std::string strTopicIMU      = "/thin_visensor_node/imu_adis16448";
    std::string strOutfileName   = "/home/dominik/ros_dumps/dump_vi.txt";

    //ds check for provided parameters
    if( 1 == cVariablesMap.count( "camera_0" ) )
    {
        strTopicCamera_0 = cVariablesMap["camera_0"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "camera_1" ) )
    {
        strTopicCamera_1 = cVariablesMap["camera_1"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "imu" ) )
    {
        strTopicIMU = cVariablesMap["imu"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "outfile" ) )
    {
        strOutfileName = cVariablesMap["outfile"].as< std::string >( );
    }

    //ds setup node
    ros::init( argc, argv, "message_dumper_node" );
    ros::NodeHandle hNode( ros::NodeHandle( "~" ) );

    //ds escape here on failure
    if( !hNode.ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) ---------------------------------------------- CONFIGURATION ----------------------------------------------\n" );
    std::printf( "(main) ROS Node namespace := '%s'\n", hNode.getNamespace( ).c_str( ) );
    std::printf( "(main) strTopicCamera_0   := '%s'\n", strTopicCamera_0.c_str( ) );
    std::printf( "(main) strTopicCamera_1   := '%s'\n", strTopicCamera_1.c_str( ) );
    std::printf( "(main) strTopicIMU        := '%s'\n", strTopicIMU.c_str( ) );
    std::printf( "(main) strOutfileName     := '%s'\n", strOutfileName.c_str( ) );
    std::printf( "(main) -----------------------------------------------------------------------------------------------------------\n" );
    std::fflush( stdout );

    //ds image transporter
    image_transport::ImageTransport cImageTransporter( hNode );

    //ds message handling
    txt_io::SensorMessageSorter cMessageSorter;
    txt_io::MessageWriter cMessageWriter;

    //ds open the outfile
    cMessageWriter.open( strOutfileName );

    //ds image dumping
    txt_io::MessageDumperTrigger cDumperTrigger = txt_io::MessageDumperTrigger( &cMessageSorter, 0, &cMessageWriter );

    //ds imu interpolation
    fps_mapper::ImuInterpolator cInterpolator( &hNode );
    cInterpolator.subscribe( strTopicIMU );

    //ds register the two cameras
    fps_mapper::ImageMessageListener cListenerCamera_0 = fps_mapper::ImageMessageListener( &hNode, &cImageTransporter, &cMessageSorter, 0, "", "" );
    cListenerCamera_0.setImuInterpolator( &cInterpolator );
    cListenerCamera_0.subscribe( strTopicCamera_0 );
    cListenerCamera_0.setVerbose( true );
    fps_mapper::ImageMessageListener cListenerCamera_1 = fps_mapper::ImageMessageListener( &hNode, &cImageTransporter, &cMessageSorter, 0, "", "" );
    cListenerCamera_1.setImuInterpolator( &cInterpolator );
    cListenerCamera_1.subscribe( strTopicCamera_1 );
    cListenerCamera_1.setVerbose( true );

    //ds register the IMU listener
    fps_mapper::CIMUMessageListener cListenerIMU = fps_mapper::CIMUMessageListener( &hNode, &cMessageSorter );
    cListenerIMU.subscribe( strTopicIMU );

    std::printf( "(main) start dumping..\n" );

    //ds main loop - callback pump
    ros::spin( );

    std::printf( "\n(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

boost::program_options::variables_map parseCommandLineParameters( int argc, char** argv )
{
    try
    {
        //ds description of all valid options
        boost::program_options::options_description cOptionDescription( "available options" );
        cOptionDescription.add_options( )( "help,h", "show usage information (this page) and exit" )
                ( "camera_0,l",boost::program_options::value<std::string>( ), "[string] ros topic for camera 0, default: /thin_visensor_node/camera_0/image_raw" )
                ( "camera_1,r", boost::program_options::value< std::string >( ), "[string] ros topic for camera 1, default: /thin_visensor_node/camera_1/image_raw" )
                ( "imu,i", boost::program_options::value< std::string >( ), "[string] ros topic for the IMU, default: /thin_visensor_node/imu_adis16448" )
                ( "outfile,o", boost::program_options::value< std::string >( ), "[string] filepath to write the local maps to, default: ./dump.txt" );

        //ds parse the options
        boost::program_options::variables_map cVariablesMap;
        boost::program_options::store( boost::program_options::parse_command_line( argc, argv, cOptionDescription), cVariablesMap );
        boost::program_options::notify( cVariablesMap );

        //ds handle help here
        if( cVariablesMap.count( "help" ) )
        {
            //ds print help and exit
            cOptionDescription.print( std::cout );
            exit( 0 );
        }

        //ds return the map
        return cVariablesMap;
    }
    catch( const boost::program_options::unknown_option& ex )
    {
        std::printf( "(parseCommandLineParameters) ERROR: you passed an unknown command line option, pass -h or [ --help ] for usage information\n" );
        exit( 1 );
    }
    catch( const boost::program_options::invalid_option_value& ex )
    {
        std::printf( "(parseCommandLineParameters) ERROR: you passed an invalid command line option value, pass -h or [ --help ] for usage information\n" );
        exit( 1 );
    }
    catch( const boost::program_options::invalid_command_line_syntax& ex )
    {
        std::printf( "(parseCommandLineParameters) ERROR: syntax error in command line options, pass -h [ --help ] for usage information\n" );
        exit( 1 );
    }
}
