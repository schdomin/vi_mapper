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
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraRIGHT = 0;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraLEFT = 0;
std::shared_ptr< txt_io::CPoseMessage > g_pActiveMessagesPose           = 0;

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCameraRIGHT = 0;
uint64_t g_uFrameIDCameraLEFT = 0;
uint64_t g_uFrameIDPose     = 0;

//ds initialize statics
std::vector< std::chrono::time_point< std::chrono::system_clock > > CMiniTimer::vec_tmStart;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder, const uint32_t& p_uSleepMicroseconds );

//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages,
                         std::string& p_strImageFolder );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds defaults
    std::string strMode              = "interactive";
    std::string strInfileMessageDump = "/home/dominik/ros_bags/datasets4dominik/good_solution/solution.log";
    std::string strImageFolder       = "/home/dominik/ros_bags/datasets4dominik/good_solution/images/";

    //ds get params
    setParametersNaive( argc, argv, strMode, strInfileMessageDump, strImageFolder );

    //ds get playback mode to enum
    EPlaybackMode eMode( ePlaybackInteractive );

    if( "stepwise" == strMode )
    {
        eMode = ePlaybackStepwise;
    }
    else if( "benchmark" == strMode )
    {
        eMode = ePlaybackBenchmark;
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
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) uFrequencyPlaybackHz := '%u'\n", uFrequencyPlaybackHz );
    std::printf( "(main) uSleepMicroseconds   := '%u'\n", uSleepMicroseconds );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) strImageFolder       := '%s'\n", strImageFolder.c_str( ) );
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
        if( 0 != g_pActiveMessageCameraRIGHT && 0 != g_pActiveMessageCameraLEFT && !g_vecMessagesIMU.isEmpty( ) && 0 != g_pActiveMessagesPose )
        {
            //ds pop the camera images
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_RIGHT( g_pActiveMessageCameraRIGHT );
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_LEFT( g_pActiveMessageCameraLEFT );

            //ds current triplet timestamp
            double dTimestamp_RIGHT( cImageCamera_RIGHT->timestamp( ) );
            double dTimestamp_LEFT( cImageCamera_LEFT->timestamp( ) );

            //ds sequence numbers have to match
            if( dTimestamp_RIGHT == dTimestamp_LEFT )
            {
                //ds reset holders
                g_pActiveMessageCameraRIGHT = 0;
                g_pActiveMessageCameraLEFT = 0;

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
                while( 0 == g_pActiveMessageCameraRIGHT && 0 == g_pActiveMessageCameraLEFT )
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
                    g_pActiveMessageCameraLEFT = 0;
                    g_pActiveMessagesPose    = 0;
                }
                if( dTimestamp_RIGHT < dTimestamp_LEFT )
                {
                    g_pActiveMessageCameraRIGHT = 0;
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

        //ds change y and z sign
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
        g_pActiveMessageCameraRIGHT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_right", "camera_right", g_uFrameIDCameraRIGHT, dTimeSeconds ) );

        //ds set image information
        std::string strImageFile;

        //ds get parameter
        issLine >> strToken >> strImageFile;

        //ds fix filepath
        strImageFile = p_strImageFolder + strImageFile.substr( 7, 18 );

        //ds read the image
        cv::Mat matImage = cv::imread( strImageFile, cv::IMREAD_GRAYSCALE );

        //ds set image to message
        g_pActiveMessageCameraRIGHT->setImage( matImage );
        g_pActiveMessageCameraRIGHT->untaint( );
    }
    else if( "IMAGE1" == strMessageType )
    {
        //ds camera_1 message
        g_pActiveMessageCameraLEFT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_left", "camera_left", g_uFrameIDCameraLEFT, dTimeSeconds ) );

        //ds set image information
        std::string strImageFile;

        //ds get parameter
        issLine >> strToken >> strImageFile;

        //ds fix filepath
        strImageFile = p_strImageFolder + strImageFile.substr( 7, 18 );

        //ds read the image
        cv::Mat matImage = cv::imread( strImageFile, cv::IMREAD_GRAYSCALE );

        //ds set image to message
        g_pActiveMessageCameraLEFT->setImage( matImage );
        g_pActiveMessageCameraLEFT->untaint( );
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

    usleep( p_uSleepMicroseconds );
}

void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages,
                         std::string& p_strImageFolder )
{
    //ds attribute names (C style for printf)
    const char* arrParameter1( "-mode" );
    const char* arrParameter2( "-messages" );
    const char* arrParameter3( "-images" );

    try
    {
        //ds parse optional command line arguments: -mode=interactive -messages="/asdasd/" -images="/asdasdgfa/"
        std::vector< std::string > vecCommandLineArguments;
        for( uint32_t u = 1; u < static_cast< uint32_t >( p_iArgc ); ++u )
        {
            //ds get parameter to string
            const std::string strParameter( p_pArgv[u] );

            //ds find '=' sign
            const std::string::size_type uStart( strParameter.find( '=' ) );

            vecCommandLineArguments.push_back( strParameter.substr( 0, uStart ) );
            vecCommandLineArguments.push_back( strParameter.substr( uStart+1, strParameter.length( )-uStart ) );
        }

        //ds check possible parameters
        const std::vector< std::string >::const_iterator itParameter1( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter1 ) );
        const std::vector< std::string >::const_iterator itParameter2( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter2 ) );
        const std::vector< std::string >::const_iterator itParameter3( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter3 ) );

        //ds set parameters if found
        if( vecCommandLineArguments.end( ) != itParameter1 ){ p_strMode                    = *( itParameter1+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter2 ){ p_strInfileCameraIMUMessages = *( itParameter2+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter3 ){ p_strImageFolder             = *( itParameter3+1 ); }
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_detector_mocked %s %s %s\n", arrParameter1, arrParameter2, arrParameter3 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_detector_mocked %s %s %s\n", arrParameter1, arrParameter2, arrParameter3 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
}
