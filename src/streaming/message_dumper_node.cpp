//ds std
#include <iostream>

//ds ROS
#include <ros/ros.h>

//ds custom
#include "txt_io/message_dumper_trigger.h"
#include "ros_wrappers/imu_interpolator.h"
#include "ros_wrappers/image_message_listener.h"
#include "ros_wrappers/imu_message_listener.h"
#include "utility/CLogger.h"


int main( int argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strTopicCameraLEFT  = "/thin_visensor_node/camera_left/image_raw";
    std::string strTopicCameraRIGHT = "/thin_visensor_node/camera_right/image_raw";
    std::string strTopicIMU         = "/thin_visensor_node/imu_adis16448";
    std::string strOutfileName      = "/home/n551jw/ros_dumps/dump_"+CLogger::getTimestamp( )+".txt";

    //ds if specific filename set TODO real parsing
    if( 2 == argc )
    {
        strOutfileName = argv[1];
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
    std::printf( "(main) ROS Node namespace  := '%s'\n", hNode.getNamespace( ).c_str( ) );
    std::printf( "(main) strTopicCameraLEFT  := '%s'\n", strTopicCameraLEFT.c_str( ) );
    std::printf( "(main) strTopicCameraRIGHT := '%s'\n", strTopicCameraRIGHT.c_str( ) );
    std::printf( "(main) strTopicIMU         := '%s'\n", strTopicIMU.c_str( ) );
    std::printf( "(main) strOutfileName      := '%s'\n", strOutfileName.c_str( ) );
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
    txt_io::MessageDumperTrigger cDumperTrigger( txt_io::MessageDumperTrigger( &cMessageSorter, 0, &cMessageWriter ) );

    //ds imu interpolation
    fps_mapper::ImuInterpolator cInterpolator( &hNode );
    cInterpolator.subscribe( strTopicIMU );

    //ds register the two cameras
    fps_mapper::ImageMessageListener cListenerCameraLEFT( fps_mapper::ImageMessageListener( &hNode, &cImageTransporter, &cMessageSorter, 0, "", "" ) );
    cListenerCameraLEFT.setImuInterpolator( &cInterpolator );
    cListenerCameraLEFT.subscribe( strTopicCameraLEFT );
    cListenerCameraLEFT.setVerbose( true );
    fps_mapper::ImageMessageListener cListenerCameraRIGHT( fps_mapper::ImageMessageListener( &hNode, &cImageTransporter, &cMessageSorter, 0, "", "" ) );
    cListenerCameraRIGHT.setImuInterpolator( &cInterpolator );
    cListenerCameraRIGHT.subscribe( strTopicCameraRIGHT );
    cListenerCameraRIGHT.setVerbose( true );

    //ds register the IMU listener
    fps_mapper::CIMUMessageListener cListenerIMU( fps_mapper::CIMUMessageListener( &hNode, &cMessageSorter ) );
    cListenerIMU.subscribe( strTopicIMU );

    std::printf( "(main) starting dumping..\n" );

    //ds main loop - callback pump
    ros::spin( );

    std::printf( "\n(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}
