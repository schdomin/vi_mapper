#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>

//ds ROS
#include <ros/ros.h>

//ds custom
#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "utility/CStack.h"
#include "core/CTrackerStereo.h"
//#include "core/CDetectorTestStereoDepth.h"
#include "utility/CMiniTimer.h"
#include "vision/CMiniVisionToolbox.h"

//ds data vectors
CStack< txt_io::CIMUMessage > g_vecMessagesIMU;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCamera_0 = 0;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCamera_1 = 0;
std::shared_ptr< txt_io::CPoseMessage > g_pActiveMessagesPose           = 0;

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCamera_0 = 0;
uint64_t g_uFrameIDCamera_1 = 0;
uint64_t g_uFrameIDPose     = 0;

//ds initialize statics
std::vector< std::chrono::time_point< std::chrono::system_clock > > CMiniTimer::vec_tmStart;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder, const uint32_t& p_uSleepMicroseconds );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds parameters TODO clean implementation
    EPlaybackMode eMode( ePlaybackInteractive );

    //ds check parameters
    if( 2 == argc )
    {
        //ds stepping desired
        if( 0 != argv[1] && 0 == std::strcmp( "-stepwise", argv[1] ) )
        {
            std::printf( "(main)[parameters] stepwise mode desired\n" );
            eMode = ePlaybackStepwise;
        }

        //ds benchmark
        else if( 0 != argv[1] && 0 == std::strcmp( "-benchmark", argv[1] ) )
        {
            std::printf( "(main)[parameters] benchmark mode desired\n" );
            eMode = ePlaybackBenchmark;
        }
    }

    //ds image resolution
    const uint32_t uImageRows = 480;
    const uint32_t uImageCols = 752;

    //ds setup node
    ros::init( argc, argv, "stereo_detector_alberto_node" );
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
    //std::string strInfileMessageDump = "/home/dominik/ros_bags/datasets4dominik/good_solution/solution.log";
    //std::string strImageFolder       = "/home/dominik/ros_bags/datasets4dominik/good_solution/images/";
    std::string strInfileMessageDump = "/home/dominik/workspace_catkin/src/stereo_vins_ros/stereo_vins/bin/solution.log";
    std::string strImageFolder       = "/home/dominik/workspace_catkin/src/stereo_vins_ros/stereo_vins/bin/data";

    //ds open the file
    std::ifstream ifMessages( strInfileMessageDump, std::ifstream::in );

    //ds internals
    uint32_t uFrequencyPlaybackHz( 100 );
    uint32_t uSleepMicroseconds( ( 1.0/uFrequencyPlaybackHz )*1e6 );
    uint32_t uWaitKeyTimeout( 1 );

    //ds adjust depending on mode
    switch( eMode )
    {
        case ePlaybackStepwise:
        {
            uSleepMicroseconds = 0;
            uWaitKeyTimeout    = 0;
            break;
        }
        case ePlaybackBenchmark:
        {
            uSleepMicroseconds = 0;
            uWaitKeyTimeout    = 1;
            break;
        }
        default:
        {
            break;
        }
    }

    //ds log configuration
    std::printf( "(main) ROS Node namespace   := '%s'\n", pNode->getNamespace( ).c_str( ) );
    std::printf( "(main) uImageRows (height)  := '%u'\n", uImageRows );
    std::printf( "(main) uImageCols (width)   := '%u'\n", uImageCols );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) uFrequencyPlaybackHz := '%u'\n", uFrequencyPlaybackHz );
    std::printf( "(main) uSleepMicroseconds   := '%u'\n", uSleepMicroseconds );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds feature detector
    //CDetectorTestStereoDepth cDetector( uImageRows, uImageCols, true, uFrequencyPlaybackHz );
    CTrackerStereo cDetector( uFrequencyPlaybackHz, eMode, uWaitKeyTimeout );

    //ds get start time
    const uint64_t uToken( CMiniTimer::tic( ) );

    //ds playback the dump
    while( ifMessages.good( ) && ros::ok( ) && !cDetector.isShutdownRequested( ) )
    {
        //ds update the frequency
        const uint32_t uSleepMicroseconds = ( 1.0/cDetector.getPlaybackFrequencyHz( ) )*1e6;

        //ds read a message
        readNextMessageFromFile( ifMessages, strImageFolder, uSleepMicroseconds );

        //ds as long as we have data in all the stacks - process
        if( 0 != g_pActiveMessageCamera_0 && 0 != g_pActiveMessageCamera_1 && !g_vecMessagesIMU.isEmpty( ) && 0 != g_pActiveMessagesPose )
        {
            //ds pop the camera images
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_RIGHT( g_pActiveMessageCamera_0 );
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_LEFT( g_pActiveMessageCamera_1 );

            //ds current triplet timestamp
            double dTimestamp_RIGHT( cImageCamera_RIGHT->timestamp( ) );
            double dTimestamp_LEFT( cImageCamera_LEFT->timestamp( ) );

            //ds sequence numbers have to match
            if( dTimestamp_RIGHT == dTimestamp_LEFT )
            {
                //ds reset holders
                g_pActiveMessageCamera_0 = 0;
                g_pActiveMessageCamera_1 = 0;

                //ds get the most recent imu measurement
                txt_io::CIMUMessage cMessageIMU( g_vecMessagesIMU.pop( ) );

                //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                while( dTimestamp_RIGHT < cMessageIMU.timestamp( ) && !g_vecMessagesIMU.isEmpty( ) )
                {
                    cMessageIMU = g_vecMessagesIMU.pop( );
                }

                //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                while( dTimestamp_RIGHT > cMessageIMU.timestamp( ) )
                {
                    //ds pop the most recent imu measurement and check
                    if( !g_vecMessagesIMU.isEmpty( ) )
                    {
                        readNextMessageFromFile( ifMessages, strImageFolder, uSleepMicroseconds );
                        cMessageIMU = g_vecMessagesIMU.pop( );
                    }
                }

                //ds call success
                bool bCalledDetector( false );

                //ds start looking for the respective pose (as long as we dont get any new camera frames)
                while( 0 == g_pActiveMessageCamera_0 && 0 == g_pActiveMessageCamera_1 )
                {
                    //ds if we have a pose
                    if( 0 != g_pActiveMessagesPose )
                    {
                        //ds check if we can use this pose
                        if( dTimestamp_RIGHT <= g_pActiveMessagesPose->timestamp( ) )
                        {
                            //ds callback with triplet
                            cDetector.receivevDataVIWithPose( cImageCamera_LEFT, cImageCamera_RIGHT, cMessageIMU, g_pActiveMessagesPose );
                            g_pActiveMessagesPose = 0;
                            bCalledDetector = true;
                        }
                        else
                        {
                            //ds log skipped frame
                            std::printf( "(main) WARNING: skipped single frame (outdated pose information) - timestamp camera_right: %.5lf s\n", dTimestamp_RIGHT );
                        }
                    }

                    //ds read a message
                    readNextMessageFromFile( ifMessages, strImageFolder, uSleepMicroseconds );
                }

                if( !bCalledDetector )
                {
                    //ds log skipped frame
                    std::printf( "(main) WARNING: skipped single frame (no pose information) - timestamp camera_right: %.5lf s\n", dTimestamp_RIGHT );
                }
            }
            else
            {
                std::printf( "(main) WARNING: could not find matching frames - timestamp camera_right: %.5lf s\n", dTimestamp_RIGHT );

                //ds reset the respectively elder one
                if( dTimestamp_RIGHT > dTimestamp_LEFT )
                {
                    g_pActiveMessageCamera_1 = 0;
                    g_pActiveMessagesPose    = 0;
                }
                if( dTimestamp_RIGHT < dTimestamp_LEFT )
                {
                    g_pActiveMessageCamera_0 = 0;
                    g_pActiveMessagesPose    = 0;
                }
            }
        }
    }

    //ds get end time
    const double dDuration( CMiniTimer::toc( uToken ) );
    const uint64_t uFrameCount( cDetector.getFrameCount( ) );

    std::printf( "(main) dataset completed\n" );
    std::printf( "(main) duration: %fs\n", dDuration );
    std::printf( "(main) total frames: %lu\n", uFrameCount );
    std::printf( "(main) frame rate (avg): %f FPS\n", uFrameCount/dDuration );

    if( 1 < uFrameCount )
    {
        //ds generate full file name
        const std::string strG2ODump( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + ".g2o" );

        //ds dump file
        cDetector.saveToG2O( strG2ODump );

        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODump.c_str( ) );
    }

    //ds if detector was not manually shut down
    if( !cDetector.isShutdownRequested( ) )
    {
        std::printf( "(main) press [ENTER] to exit" );
        std::getchar( );
    }

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder, const uint32_t& p_uSleepMicroseconds )
{
    //ds line buffer
    std::string strLineBuffer;

    //ds read one line
    std::getline( p_ifMessages, strLineBuffer );

    //ds get it to a stringstream
    std::istringstream issLine( strLineBuffer );

    //ds information fields
    double dTimeSeconds;
    std::string strToken;
    std::string strMessageType;

    //ds fetch first part of the message
    issLine >> strToken >> strToken >> dTimeSeconds >> strMessageType;

    //ds get time to seconds
    dTimeSeconds /= 1000000;

    //ds set message information depending on type
    if( "IMU" == strMessageType )
    {
        //ds IMU message
        txt_io::CIMUMessage msgIMU( "/imu", "imu", g_uFrameIDIMU, dTimeSeconds );

        //ds fields
        Eigen::Vector3d vecAngularVelocity;
        Eigen::Vector3d vecLinearAcceleration;

        //ds parse the values (order x/z/y) TODO align coordinate systems
        issLine >> strToken >> vecLinearAcceleration[0] >> vecLinearAcceleration[1] >> vecLinearAcceleration[2] >> vecAngularVelocity[0] >> vecAngularVelocity[1] >> vecAngularVelocity[2];

        //ds flip the z and y value for consistency with the world frame
        vecLinearAcceleration[1] = -vecLinearAcceleration[1];
        vecLinearAcceleration[2] = -vecLinearAcceleration[2];

        //ds compensate gravitational component (http://en.wikipedia.org/wiki/ISO_80000-3)
        //vecLinearAcceleration[1] += 9.80665;

        //ds set message fields
        msgIMU.setAngularVelocity( vecAngularVelocity );
        msgIMU.setLinearAcceleration( vecLinearAcceleration );

        //ds pump it into the synchronizer
        g_vecMessagesIMU.push( msgIMU );
    }
    else if( "IMAGE0" == strMessageType )
    {
        //ds camera_0 message
        g_pActiveMessageCamera_0 = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_right", "camera_right", g_uFrameIDCamera_0, dTimeSeconds ) );

        //ds set image information
        std::string strImageFile;

        //ds get parameter
        issLine >> strToken >> strImageFile;

        //ds fix filepath
        strImageFile = p_strImageFolder + strImageFile.substr( 7, 18 );

        //ds read the image
        cv::Mat matImage = cv::imread( strImageFile, cv::IMREAD_GRAYSCALE );

        //ds set image to message
        g_pActiveMessageCamera_0->setImage( matImage );
    }
    else if( "IMAGE1" == strMessageType )
    {
        //ds camera_1 message
        g_pActiveMessageCamera_1 = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_left", "camera_left", g_uFrameIDCamera_1, dTimeSeconds ) );

        //ds set image information
        std::string strImageFile;

        //ds get parameter
        issLine >> strToken >> strImageFile;

        //ds fix filepath
        strImageFile = p_strImageFolder + strImageFile.substr( 7, 18 );

        //ds read the image
        cv::Mat matImage = cv::imread( strImageFile, cv::IMREAD_GRAYSCALE );

        //ds set image to message
        g_pActiveMessageCamera_1->setImage( matImage );
    }
    else
    {
        //ds pose message
        g_pActiveMessagesPose = std::shared_ptr< txt_io::CPoseMessage >( new txt_io::CPoseMessage( "/pose", "pose", g_uFrameIDPose, dTimeSeconds ) );

        //ds get the info
        Eigen::Vector3d vecPosition;
        Eigen::Vector3d vecOrientation;

        //ds parse the values
        issLine >> strToken >> vecPosition[0] >> vecPosition[1] >> vecPosition[2] >> vecOrientation[0] >> vecOrientation[1] >> vecOrientation[2];

        //ds set message field
        g_pActiveMessagesPose->setPosition( vecPosition );

        //ds compute quaternion from angles TODO: remove necessity of casts to avoid eclipse errors
        //Eigen::Quaterniond vecOrientationQuaternion( CMiniVisionToolbox::fromEulerAngles( vecOrientation ) );

        //ds set it
        //g_pActiveMessagesPose->setOrientationQuaternion( vecOrientationQuaternion );
        //g_pActiveMessagesPose->setOrientationEulerAngles( vecOrientation );
        g_pActiveMessagesPose->setOrientationMatrix( CMiniVisionToolbox::fromOrientationRodrigues( vecOrientation ) );

        //std::cout << g_pActiveMessagesPose->getOrientationMatrix( ) << std::endl;
    }

    //usleep( p_uSleepMicroseconds );
}
