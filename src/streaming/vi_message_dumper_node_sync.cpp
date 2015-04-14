//ds std
#include <iostream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>

//ds ROS
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>

//ds homebrewed includes
#include "globals/system_utils.h"
#include "txt_io/message_dumper_trigger.h"
#include "txt_io/pinhole_image_message.h"
#include "ros_wrappers/imu_interpolator.h"
#include "ros_wrappers/image_message_listener.h"

//ds UGLY globals TODO purge
const static uint32_t g_uMaxQueueSize = 1024;
boost::lockfree::spsc_queue< sensor_msgs::ImageConstPtr, boost::lockfree::capacity< g_uMaxQueueSize > > g_queMessagesCamera0;
boost::lockfree::spsc_queue< sensor_msgs::ImageConstPtr, boost::lockfree::capacity< g_uMaxQueueSize > > g_queMessagesCamera1;
boost::lockfree::spsc_queue< sensor_msgs::ImuConstPtr, boost::lockfree::capacity< g_uMaxQueueSize > > g_queMessagesIMU;

//ds sensor topic callbacks
void callbackCamera0( const sensor_msgs::ImageConstPtr msg ){ g_queMessagesCamera0.push( msg ); }
void callbackCamera1( const sensor_msgs::ImageConstPtr msg ){ g_queMessagesCamera1.push( msg ); }
void callbackIMU0( const  sensor_msgs::ImuConstPtr msg ){ g_queMessagesIMU.push( msg ); }

//ds threads
void handleMessageTriplets( const std::string& p_strTopicCamera0,
                          const std::string& p_strTopicCamera1,
                          const std::string& p_strOutfileName,
                          const boost::shared_ptr< ros::NodeHandle >& p_hNode );

//ds command line parameters parser
boost::program_options::variables_map parseCommandLineParameters( int argc, char** argv );

int main( int argc, char **argv )
{
    //ds parse input
    boost::program_options::variables_map cVariablesMap = parseCommandLineParameters( argc, argv );

    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strTopicCamera0    = "/cam0/image_raw";
    std::string strTopicCamera1    = "/cam1/image_raw";
    std::string strTopicIMU        = "/imu0";
    std::string strBaseLinkFrameID = "";
    std::string strOutfileName     = "dump.txt";
    std::string strTopicOdometry   = "/odom";

    //ds check for provided parameters
    if( 1 == cVariablesMap.count( "camera_0" ) )
    {
        strTopicCamera0 = cVariablesMap["camera_0"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "camera_1" ) )
    {
        strTopicCamera1 = cVariablesMap["camera_1"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "imu" ) )
    {
        strTopicIMU = cVariablesMap["imu"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "base_link_frame_id" ) )
    {
        strBaseLinkFrameID = cVariablesMap["base_link_frame_id"].as< std::string >( );
    }
    if( 1 == cVariablesMap.count( "outfile" ) )
    {
        strOutfileName = cVariablesMap["outfile"].as< std::string >( );
    }

    //ds setup node
    ros::init( argc, argv, "vi_message_dumper_node" );
    boost::shared_ptr< ros::NodeHandle > hNode( new ros::NodeHandle( "~" ) );

    //ds escape here on failure
    if( !hNode->ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) ---------------------------------------------- CONFIGURATION ----------------------------------------------\n" );
    std::printf( "(main) ROS Node namespace := '%s'\n", hNode->getNamespace( ).c_str( ) );
    std::printf( "(main) strTopicCamera0    := '%s'\n", strTopicCamera0.c_str( ) );
    std::printf( "(main) strTopicCamera1    := '%s'\n", strTopicCamera1.c_str( ) );
    std::printf( "(main) strTopicIMU        := '%s'\n", strTopicIMU.c_str( ) );
    std::printf( "(main) strBaseLinkFrameID := '%s'\n", strBaseLinkFrameID.c_str( ) );
    std::printf( "(main) strOutfileName     := '%s'\n", strOutfileName.c_str( ) );
    std::printf( "(main) strTopicOdometry   := '%s'\n", strTopicOdometry.c_str( ) );
    std::printf( "(main) g_uMaxQueueSize    := '%u'\n", g_uMaxQueueSize );
    std::printf( "(main) -----------------------------------------------------------------------------------------------------------\n" );
    std::fflush( stdout );

    //ds subscribe to visensor topics
    ros::Subscriber cSubscriberCamera0 = hNode->subscribe( strTopicCamera0, g_uMaxQueueSize, callbackCamera0 );
    ros::Subscriber cSubscriberCamera1 = hNode->subscribe( strTopicCamera1, g_uMaxQueueSize, callbackCamera1 );
    ros::Subscriber cSubscribeIMU0     = hNode->subscribe( strTopicIMU, g_uMaxQueueSize, callbackIMU0 );

    //ds start threads
    std::printf( "(main) start dumping..\n" );
    boost::thread threadDumper( handleMessageTriplets, strTopicCamera0, strTopicCamera1, strOutfileName, hNode );

    //ds main loop - pump callbacks
    ros::spin( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

void handleMessageTriplets(  const std::string& p_strTopicCamera0,
                           const std::string& p_strTopicCamera1,
                           const std::string& p_strOutfileName,
                           const boost::shared_ptr< ros::NodeHandle >& p_hNode )
{
    std::printf( "(dumpTripletMessages) thread launched\n" );

    //ds verify lock freeness
    if( !g_queMessagesCamera0.is_lock_free( ) || !g_queMessagesCamera1.is_lock_free( ) || !g_queMessagesIMU.is_lock_free( ) )
    {
        std::printf( "(dumpTripletMessages) stacks are not lockfree - thread terminated\n" );
        return;
    }

    //ds message handling
    txt_io::MessageWriter cMessageWriter;

    //ds open the outfile
    cMessageWriter.open( p_strOutfileName );

    //ds hold steady
    while( ros::ok( ) )
    {
        //ds as long as we have data - process
        while( !g_queMessagesCamera0.empty( ) && !g_queMessagesCamera1.empty( ) && !g_queMessagesIMU.empty( ) )
        {
            //ds pop the camera images
            sensor_msgs::ImageConstPtr pImageCamera0; g_queMessagesCamera0.pop( pImageCamera0 );
            sensor_msgs::ImageConstPtr pImageCamera1; g_queMessagesCamera1.pop( pImageCamera1 );

            //ds current triplet timestamp
            const ros::Time tmTimestamp( pImageCamera0->header.stamp );

            //ds timestamps have to match
            if( tmTimestamp == pImageCamera1->header.stamp )
            {
                //ds get the most recent imu measurement
                sensor_msgs::ImuConstPtr pMessageIMU; g_queMessagesIMU.pop( pMessageIMU );

                //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                while( tmTimestamp < pMessageIMU->header.stamp && !g_queMessagesIMU.empty( ) )
                {
                    g_queMessagesIMU.pop( pMessageIMU );
                }

                //ds in case we have not received the matching timestamp yet
                while( tmTimestamp > pMessageIMU->header.stamp )
                {
                    //ds pop the most recent imu measurement and check
                    if( !g_queMessagesIMU.empty( ) )
                    {
                        g_queMessagesIMU.pop( pMessageIMU );
                    }
                }

                //ds verify timestamps (TODO remove for performance)
                if( tmTimestamp == pMessageIMU->header.stamp )
                {
                    //ds dump triplet
                    cMessageWriter.writeMessage( txt_io::PinholeImageMessage( p_strTopicCamera0, pImageCamera0->header.frame_id, pImageCamera0->header.seq, tmTimestamp.toSec( ) ) );
                    cMessageWriter.writeMessage( txt_io::PinholeImageMessage( p_strTopicCamera1, pImageCamera1->header.frame_id, pImageCamera1->header.seq, tmTimestamp.toSec( ) ) );

                    //ds progress
                    std::printf( "x" );
                }
                else
                {
                    //ds unable to find matching IMU measurement
                    std::printf( "o" );
                }
            }
            else
            {
                std::printf( "ERROR: frame timestamp mismatch\n" );
            }
        }

        //ds flush buffer
        std::fflush( stdout );
    }

    std::printf( "\n(dumpTripletMessages) thread terminated\n" );
}

boost::program_options::variables_map parseCommandLineParameters( int argc, char** argv )
{
    try
    {
        //ds description of all valid options
        boost::program_options::options_description cOptionDescription( "available options" );
        cOptionDescription.add_options( )( "help,h", "show usage information (this page) and exit" )
                ( "camera_0,l",boost::program_options::value<std::string>( ), "[string] ros topic for camera 0, default: /cam0/image_raw" )
                ( "camera_1,r", boost::program_options::value< std::string >( ), "[string] ros topic for camera 1, default: /cam1/image_raw" )
                ( "imu,i", boost::program_options::value< std::string >( ), "[string] ros topic for the IMU, default: /imu0" )
                ( "base_link_frame_id,b", boost::program_options::value< std::string >( ), "[string] if provided it will consider the odometry, default: empty" )
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
