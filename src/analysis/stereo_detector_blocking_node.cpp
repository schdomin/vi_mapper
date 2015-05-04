//ds std
#include <iostream>
#include <thread>
#include <functional>

//ds ROS
#include <ros/ros.h>
#include <utility/CNaiveStereoDetector.h>
#include "txt_io/message_reader.h"
#include "utility/CStack.h"

//ds data vectors
CStack< txt_io::CIMUMessage > g_vecMessagesIMU;
CStack< txt_io::PinholeImageMessage > g_vecMessagesCamera_0;
CStack< txt_io::PinholeImageMessage > g_vecMessagesCamera_1;

void readNextMessageFromFile( txt_io::MessageReader& p_cMessageReader, const uint32_t& p_uSleepMicroseconds );

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
    ros::init( argc, argv, "stereo_detector_blocking_node" );
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

    //ds playback the dump
    while( cMessageReader.good( ) && ros::ok( ) )
    {
        //ds read a message
        readNextMessageFromFile( cMessageReader, uSleepMicroseconds );

        //ds as long as we have data in all the stacks - process
        if( !g_vecMessagesCamera_0.isEmpty( ) && !g_vecMessagesCamera_1.isEmpty( ) && !g_vecMessagesIMU.isEmpty( ) )
        {
            //ds pop the camera images
            txt_io::PinholeImageMessage cImageCamera_0( g_vecMessagesCamera_0.pop( ) );
            txt_io::PinholeImageMessage cImageCamera_1( g_vecMessagesCamera_1.pop( ) );

            //ds current triplet timestamp
            double dTimestamp_0( cImageCamera_0.timestamp( ) );
            double dTimestamp_1( cImageCamera_1.timestamp( ) );

            //ds sequence numbers have to match
            if( dTimestamp_0 == dTimestamp_1 )
            {
                //ds get the most recent imu measurement
                txt_io::CIMUMessage cMessageIMU( g_vecMessagesIMU.pop( ) );

                //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                while( dTimestamp_0 < cMessageIMU.timestamp( ) && !g_vecMessagesIMU.isEmpty( ) )
                {
                    readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                    cMessageIMU = g_vecMessagesIMU.pop( );
                }

                //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                while( dTimestamp_0 > cMessageIMU.timestamp( ) )
                {
                    //ds pop the most recent imu measurement and check
                    if( !g_vecMessagesIMU.isEmpty( ) )
                    {
                        readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                        cMessageIMU = g_vecMessagesIMU.pop( );
                    }
                }

                //ds callback with triplet
                cDetector.receivevDataVI( cImageCamera_1, cImageCamera_0, cMessageIMU );
            }
            else
            {
                //ds skipped timestamp
                double dTimestampSkipped = 0.0;

                //ds if the first timestamp is older
                if( dTimestamp_0 < dTimestamp_1 )
                {
                    //ds save timestamp
                    dTimestampSkipped = dTimestamp_0;

                    //ds wait for the next stamp from 0
                    while( g_vecMessagesCamera_0.isEmpty( ) )
                    {
                        readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                    }

                    //ds pop the next item
                    cImageCamera_0 = g_vecMessagesCamera_0.pop( );

                    //ds update the timestamp
                    dTimestamp_0 = cImageCamera_0.timestamp( );
                }
                else
                {
                    //ds save timestamp
                    dTimestampSkipped = dTimestamp_1;

                    //ds wait for the next stamp from 1
                    while( g_vecMessagesCamera_1.isEmpty( ) )
                    {
                        readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                    }

                    //ds pop the next item
                    cImageCamera_1 = g_vecMessagesCamera_1.pop( );

                    //ds update the timestamp
                    dTimestamp_1 = cImageCamera_1.timestamp( );
                }

                //ds sequence numbers have to match now TODO: remove boilerplate code
                if( dTimestamp_0 == dTimestamp_1 )
                {
                    //ds log skipped frame
                    std::printf( "(main) WARNING: skipped single frame - timestamp: %.5lf s\n", dTimestampSkipped );

                    //ds get the most recent imu measurement
                    txt_io::CIMUMessage cMessageIMU( g_vecMessagesIMU.pop( ) );

                    //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                    while( dTimestamp_0 < cMessageIMU.timestamp( ) && !g_vecMessagesIMU.isEmpty( ) )
                    {
                        readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                        cMessageIMU = g_vecMessagesIMU.pop( );
                    }

                    //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                    while( dTimestamp_0 > cMessageIMU.timestamp( ) )
                    {
                        //ds pop the most recent imu measurement and check
                        if( !g_vecMessagesIMU.isEmpty( ) )
                        {
                            readNextMessageFromFile( cMessageReader, uSleepMicroseconds );
                            cMessageIMU = g_vecMessagesIMU.pop( );
                        }
                    }

                    //ds callback with triplet
                    cDetector.receivevDataVI( cImageCamera_1, cImageCamera_0, cMessageIMU );
                }
                else
                {
                    std::printf( "(main) ERROR: could not find matching frames, aborting\n" );
                }
            }
        }
    }

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

void readNextMessageFromFile( txt_io::MessageReader& p_cMessageReader, const uint32_t& p_uSleepMicroseconds )
{
    //ds retrieve a message
    txt_io::BaseMessage* msgBase = p_cMessageReader.readMessage( );

    //ds if set
    if( 0 != msgBase )
    {
        //ds trigger callbacks artificially - check for imu input first
        if( "IMU_MESSAGE" == msgBase->tag( ) )
        {
            //ds IMU message
            const txt_io::CIMUMessage* msgIMU = dynamic_cast< txt_io::CIMUMessage* >( msgBase );

            //ds pump it into the synchronizer
            g_vecMessagesIMU.push( *msgIMU );
        }
        else
        {
            //ds camera message
            txt_io::PinholeImageMessage* msgImage = dynamic_cast< txt_io::PinholeImageMessage* >( msgBase );

            //ds if its the right camera
            if( "camera_0" == msgImage->frameId( ) )
            {
                g_vecMessagesCamera_0.push( *msgImage );
            }
            else
            {
                g_vecMessagesCamera_1.push( *msgImage );
            }
        }
    }

    usleep( p_uSleepMicroseconds );
}
