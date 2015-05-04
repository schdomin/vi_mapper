//ds std
#include <iostream>
#include <thread>
#include <functional>

//ds ROS
#include <ros/ros.h>
#include <utility/CNaiveStereoDetector.h>
#include "txt_io/message_reader.h"
#include "utility/CMessageSynchronizer.h"

int main( int argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strTopicCamera_0  = "/thin_visensor_node/camera_0/image_raw";
    std::string strTopicCamera_1  = "/thin_visensor_node/camera_1/image_raw";
    std::string strTopicIMU       = "/thin_visensor_node/imu_adis16448";

    //ds image resolution
    const uint32_t uImageRows = 480;
    const uint32_t uImageCols = 752;

    //ds setup node
    ros::init( argc, argv, "stereo_detector_node" );
    std::shared_ptr< ros::NodeHandle > pNode( new ros::NodeHandle( "~" ) );

    //ds escape here on failure
    if( !pNode->ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds default files
    std::string strInfileMessageDump = "/home/dominik/ros_bags/dis_ingresso_walk.txt";

    //ds overwrite the string if present
    if( 0 != argv[1] )
    {
        strInfileMessageDump = argv[1];
    }

    //ds message loop
    txt_io::MessageReader cMessageReader;
    cMessageReader.open( strInfileMessageDump );

    //ds set message frequency
    const uint32_t uFrequencyPlaybackHz = 200;
    const uint32_t uSleepMicroseconds   = ( 1.0/uFrequencyPlaybackHz )*1e6;

    //ds log configuration
    std::printf( "(main) ---------------------------------------------- CONFIGURATION ----------------------------------------------\n" );
    std::printf( "(main) ROS Node namespace   := '%s'\n", pNode->getNamespace( ).c_str( ) );
    std::printf( "(main) uImageRows (height)  := '%u'\n", uImageRows );
    std::printf( "(main) uImageCols (width)   := '%u'\n", uImageCols );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) uFrequencyPlaybackHz := '%u'\n", uFrequencyPlaybackHz );
    std::printf( "(main) uSleepMicroseconds   := '%u'\n", uSleepMicroseconds );
    std::printf( "(main) -----------------------------------------------------------------------------------------------------------\n" );
    std::fflush( stdout );

    //ds feature detector
    CNaiveStereoDetector cDetector( uImageRows, uImageCols, true, uFrequencyPlaybackHz );

    //ds allocate a message synchronizer
    CMessageSynchronizer cSynchronizer;
    cSynchronizer.setSynchronizedMessageCallback( std::bind( &CNaiveStereoDetector::receivevDataVI, &cDetector, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3 ) );

    //ds start synchronization
    std::thread tSynchronization( cSynchronizer.startSynchronization( ) );

    //ds playback the dump
    while( cMessageReader.good( ) && ros::ok( ) )
    {
        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
        {
            //ds trigger callbacks artificially - check for imu input first
            if( "IMU_MESSAGE" == msgBase->tag( ) )
            {
                //ds IMU message
                const txt_io::CIMUMessage* msgIMU = dynamic_cast< txt_io::CIMUMessage* >( msgBase );

                //ds pump it into the synchronizer
                cSynchronizer.addMessageIMU( msgIMU );
            }
            else
            {
                //ds camera message
                txt_io::PinholeImageMessage* msgImage = dynamic_cast< txt_io::PinholeImageMessage* >( msgBase );

                //ds if its the right camera
                if( "camera_0" == msgImage->frameId( ) )
                {
                    cSynchronizer.addMessageCamera_0( msgImage );
                }
                else
                {
                    cSynchronizer.addMessageCamera_1( msgImage );
                }
            }
        }

        usleep( uSleepMicroseconds );
    }

    //ds stop synchronization
    cSynchronizer.stopSynchronization( );

    //ds detach thread
    tSynchronization.detach( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}
