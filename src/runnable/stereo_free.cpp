#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>
#include <qapplication.h>

//ds ROS
#include <ros/ros.h>

//ds custom
#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "utility/CStack.h"
#include "core/CTrackerStereoMotionModel.h"
#include "utility/CMiniTimer.h"
#include "exceptions/CExceptionEndOfFile.h"
#include "gui/CViewerScene.h"
//#include "types/CIMUInterpolator.h"

//ds data vectors
CStack< txt_io::CIMUMessage > g_vecMessagesIMU;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCamera_0 = 0;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCamera_1 = 0;

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCamera_0 = 0;
uint64_t g_uFrameIDCamera_1 = 0;

//ds interpolator
//CIMUInterpolator g_cInterpolator;

//ds initialize statics
std::vector< std::chrono::time_point< std::chrono::system_clock > > CMiniTimer::vec_tmStart;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder );

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
    ros::init( argc, argv, "stereo_detector_free" );
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
    uint32_t uWaitKeyTimeout( 1 );

    //ds adjust depending on mode
    switch( eMode )
    {
        case ePlaybackStepwise:
        {
            uWaitKeyTimeout    = 0;
            break;
        }
        case ePlaybackBenchmark:
        {
            uWaitKeyTimeout    = 1;
            break;
        }
        default:
        {
            //ds exit
            std::printf( "(main) interactive mode not supported, aborting\n" );
            std::printf( "(main) terminated: %s\n", argv[0] );
            std::fflush( stdout);
            return 0;
        }
    }

    //ds start qt application
    std::shared_ptr< QApplication > pQT( std::make_shared< QApplication >( argc, argv ) );

    //ds instantiate the viewer
    std::shared_ptr< CViewerScene > pViewer( std::make_shared< CViewerScene >( ) );
    pViewer->setWindowTitle( "CViewerScene: WORLD" );
    pViewer->show( );

    //ds log configuration
    std::printf( "(main) ROS Node namespace   := '%s'\n", pNode->getNamespace( ).c_str( ) );
    std::printf( "(main) uImageRows (height)  := '%u'\n", uImageRows );
    std::printf( "(main) uImageCols (width)   := '%u'\n", uImageCols );
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) strImageFolder       := '%s'\n", strImageFolder.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds feature detector
    CTrackerStereoMotionModel cDetector( eMode, pViewer, uWaitKeyTimeout );

    //ds get start time
    const uint64_t uToken( CMiniTimer::tic( ) );

    try
    {
        //ds playback the dump
        while( ifMessages.good( ) && ros::ok( ) && !cDetector.isShutdownRequested( ) )
        {
            //ds read a message
            readNextMessageFromFile( ifMessages, strImageFolder );

            //ds as long as we have data in all the stacks - process
            if( 0 != g_pActiveMessageCamera_0 && 0 != g_pActiveMessageCamera_1 && !g_vecMessagesIMU.isEmpty( ) )
            {
                //ds pop the camera images
                std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_0( g_pActiveMessageCamera_0 );
                std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_1( g_pActiveMessageCamera_1 );

                //ds current triplet timestamp
                double dTimestamp_0( cImageCamera_0->timestamp( ) );
                double dTimestamp_1( cImageCamera_1->timestamp( ) );

                //ds sequence numbers have to match
                if( dTimestamp_0 == dTimestamp_1 )
                {
                    //ds reset holders
                    g_pActiveMessageCamera_0 = 0;
                    g_pActiveMessageCamera_1 = 0;

                    //ds get the most recent imu measurement
                    txt_io::CIMUMessage cMessageIMU( g_vecMessagesIMU.pop( ) );

                    //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                    while( dTimestamp_0 < cMessageIMU.timestamp( ) && !g_vecMessagesIMU.isEmpty( ) )
                    {
                        cMessageIMU = g_vecMessagesIMU.pop( );
                    }

                    //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no matching IMU data arriving
                    while( dTimestamp_0 > cMessageIMU.timestamp( ) )
                    {
                        //ds pop the most recent imu measurement
                        readNextMessageFromFile( ifMessages, strImageFolder );

                        //ds check if we got a new imu measurement
                        if( !g_vecMessagesIMU.isEmpty( ) )
                        {
                            cMessageIMU = g_vecMessagesIMU.pop( );
                        }
                    }

                    //ds check if we can call the detector (no other camera images arrived in the meantime)
                    if( 0 == g_pActiveMessageCamera_0 && 0 == g_pActiveMessageCamera_1 && dTimestamp_0 == cMessageIMU.timestamp( ) )
                    {
                        assert( 0 != cImageCamera_1 );
                        assert( 0 != cImageCamera_0 );

                        //ds callback with triplet
                        //cDetector.receivevDataVI( cImageCamera_1, cImageCamera_0, cMessageIMU, g_cInterpolatorAngular.getTransformation( dTimestamp_0 ) );
                        cDetector.receivevDataVI( cImageCamera_1, cImageCamera_0, cMessageIMU );
                    }
                    else
                    {
                        //ds log skipped frame
                        std::printf( "(main) WARNING: skipped frame - timestamp [LEFT|RIGHT|IMU]: [%f|%f|%f]\n", dTimestamp_1, dTimestamp_0, cMessageIMU.timestamp( ) );
                    }
                }
                else
                {
                    std::printf( "(main) WARNING: could not find matching frames - timestamp [LEFT|RIGHT]: [%f|%f]\n", dTimestamp_1, dTimestamp_0 );

                    //ds reset the respectively elder one
                    if( dTimestamp_0 > dTimestamp_1 )
                    {
                        g_pActiveMessageCamera_1 = 0;
                    }
                    if( dTimestamp_0 < dTimestamp_1 )
                    {
                        g_pActiveMessageCamera_0 = 0;
                    }
                }
            }
        }
    }
    catch( const CExceptionEndOfFile& p_cException )
    {
        std::printf( "(main) caught exception: %s\n", p_cException.what( ) );
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
        //ds generate full file names
        const std::string strG2ODumpComplete( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + "_uvdepthdisparity.g2o" );
        const std::string strG2ODumpXYZ( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + "_xyz.g2o" );
        const std::string strG2ODumpUVDepth( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + "_uvdepth.g2o" );
        const std::string strG2ODumpUVDisparity( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + "_uvdisparity.g2o" );
        const std::string strG2ODumpCOMBO( "/home/dominik/libs/g2o/bin/graphs/graph_" + CMiniTimer::getTimestamp( ) + "_combo.g2o" );

        //ds dump file
        cDetector.saveUVDepthOrDisparity( strG2ODumpComplete );
        cDetector.saveXYZ( strG2ODumpXYZ );
        cDetector.saveUVDepth( strG2ODumpUVDepth );
        cDetector.saveUVDisparity( strG2ODumpUVDisparity );
        cDetector.saveCOMBO( strG2ODumpCOMBO );

        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODumpComplete.c_str( ) );
        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODumpXYZ.c_str( ) );
        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODumpUVDepth.c_str( ) );
        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODumpUVDisparity.c_str( ) );
        std::printf( "(main) successfully written g2o dump to: %s\n", strG2ODumpCOMBO.c_str( ) );
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

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder )
{
    //ds line buffer
    std::string strLineBuffer;

    //ds read one line
    std::getline( p_ifMessages, strLineBuffer );

    if( strLineBuffer.empty( ) ){ throw CExceptionEndOfFile( "received empty string - file end reached" ); }

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
        CLinearAccelerationInIMUFrame vecLinearAcceleration;

        //ds parse the values (order x/z/y) TODO align coordinate systems
        issLine >> strToken >> vecLinearAcceleration[0] >> vecLinearAcceleration[1] >> vecLinearAcceleration[2] >> vecAngularVelocity[0] >> vecAngularVelocity[1] >> vecAngularVelocity[2];

        //ds rotate around X axis by 180 degrees
        vecLinearAcceleration.y( ) = -vecLinearAcceleration.y( );
        vecLinearAcceleration.z( ) = -vecLinearAcceleration.z( );
        vecAngularVelocity.y( )    = -vecAngularVelocity.y( );
        vecAngularVelocity.z( )    = -vecAngularVelocity.z( );

        //ds add to interpolator
        //g_cInterpolator.addMeasurement( vecLinearAcceleration, vecAngularVelocity, dTimeSeconds );

        //ds set message fields
        msgIMU.setAngularVelocity( vecAngularVelocity );
        msgIMU.setLinearAcceleration( vecLinearAcceleration );

        //ds pump it into the synchronizer
        g_vecMessagesIMU.push( msgIMU );
    }
    else if( "IMAGE0" == strMessageType )
    {
        //ds camera_0 message
        g_pActiveMessageCamera_0 = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_0", "camera_0", g_uFrameIDCamera_0, dTimeSeconds ) );

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
        g_pActiveMessageCamera_0->untaint( );
    }
    else if( "IMAGE1" == strMessageType )
    {
        //ds camera_1 message
        g_pActiveMessageCamera_1 = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_1", "camera_1", g_uFrameIDCamera_1, dTimeSeconds ) );

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
        g_pActiveMessageCamera_1->untaint( );
    }
    else
    {
        //std::printf( "(main) unknown message type: %s\n", strMessageType.c_str( ) );
    }
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
