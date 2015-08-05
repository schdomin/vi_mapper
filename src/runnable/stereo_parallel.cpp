#include <iostream>
#include <opencv/highgui.h>
#include <stack>
#include <qapplication.h>
#include <thread>

//ds custom
#include "core/CTrackerStereoMotionModel.h"
#include "exceptions/CExceptionEndOfFile.h"
#include "gui/CViewerScene.h"
#include "parallelization/CThreadGUI.h"

//ds data vectors
std::stack< txt_io::CIMUMessage > g_vecMessagesIMU;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraRIGHT = 0;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraLEFT = 0;

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCamera_0 = 0;
uint64_t g_uFrameIDCamera_1 = 0;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder );

//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages,
                         std::string& p_strImageFolder );

int main( int argc, char **argv )
{
    //assert( false ); //ds check RELEASE build

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

    //ds start the qt application
    QApplication cApplicationQT( argc, argv );

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

    //ds open the file
    std::ifstream ifMessages( strInfileMessageDump, std::ifstream::in );

    //ds escape here on failure
    if( !ifMessages.good( ) )
    {
        std::printf( "\n(main) ERROR: unable to open message file: %s\n", strInfileMessageDump.c_str( ) );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) uImageRows (height)  := '%u'\n", uImageRows );
    std::printf( "(main) uImageCols (width)   := '%u'\n", uImageCols );
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) strImageFolder       := '%s'\n", strImageFolder.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds allocate the tracker
    CTrackerStereoMotionModel cTracker( eMode, uWaitKeyTimeout );

    //ds instantiate a Qt thread
    CThreadGUI *pThreadGUI( new CThreadGUI( cTracker.getLandmarksHandle( ) ) );
    pThreadGUI->start( QThread::Priority::HighestPriority );

    //ds stop time
    const double dTimeStart = CLogger::getTimeSeconds( );

    try
    {
        //ds playback the dump
        while( ifMessages.good( ) && !cTracker.isShutdownRequested( ) )
        {
            //ds check for frames to update the viewer
            if( cTracker.isFrameAvailable( ) ){ pThreadGUI->updateFrame( cTracker.getFrameLEFTtoWORLD( ) ); }

            //ds read a message
            readNextMessageFromFile( ifMessages, strImageFolder );

            //ds as long as we have data in all the stacks - process
            if( 0 != g_pActiveMessageCameraRIGHT && 0 != g_pActiveMessageCameraLEFT && !g_vecMessagesIMU.empty( ) )
            {
                //ds pop the camera images
                std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_0( g_pActiveMessageCameraRIGHT );
                std::shared_ptr< txt_io::PinholeImageMessage > cImageCamera_1( g_pActiveMessageCameraLEFT );

                //ds current triplet timestamp
                double dTimestamp_0( cImageCamera_0->timestamp( ) );
                double dTimestamp_1( cImageCamera_1->timestamp( ) );

                //ds sequence numbers have to match
                if( dTimestamp_0 == dTimestamp_1 )
                {
                    //ds reset holders
                    g_pActiveMessageCameraRIGHT = 0;
                    g_pActiveMessageCameraLEFT = 0;

                    //ds get the most recent imu measurement
                    txt_io::CIMUMessage cMessageIMU( g_vecMessagesIMU.top( ) );
                    g_vecMessagesIMU.pop( );

                    //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                    while( dTimestamp_0 < cMessageIMU.timestamp( ) && !g_vecMessagesIMU.empty( ) )
                    {
                        cMessageIMU = g_vecMessagesIMU.top( );
                        g_vecMessagesIMU.pop( );
                    }

                    //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no matching IMU data arriving
                    while( dTimestamp_0 > cMessageIMU.timestamp( ) )
                    {
                        //ds pop the most recent imu measurement
                        readNextMessageFromFile( ifMessages, strImageFolder );

                        //ds check if we got a new imu measurement
                        if( !g_vecMessagesIMU.empty( ) )
                        {
                            cMessageIMU = g_vecMessagesIMU.top( );
                            g_vecMessagesIMU.pop( );
                        }
                    }

                    //ds check if we can call the detector (no other camera images arrived in the meantime)
                    if( 0 == g_pActiveMessageCameraRIGHT && 0 == g_pActiveMessageCameraLEFT && dTimestamp_0 == cMessageIMU.timestamp( ) )
                    {
                        assert( 0 != cImageCamera_1 );
                        assert( 0 != cImageCamera_0 );

                        //ds callback with triplet
                        cTracker.receivevDataVI( cImageCamera_1, cImageCamera_0, cMessageIMU );
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
                        g_pActiveMessageCameraLEFT = 0;
                    }
                    if( dTimestamp_0 < dTimestamp_1 )
                    {
                        g_pActiveMessageCameraRIGHT = 0;
                    }
                }
            }
        }
    }
    catch( const CExceptionEndOfFile& p_cException )
    {
        std::printf( "(main) caught exception: %s\n", p_cException.what( ) );
    }

    //ds stop threads
    pThreadGUI->close( );

    //ds wait until all threads are closed
    while( pThreadGUI->isRunning( ) )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        std::printf( "(main) waiting for threads to shut down ..\n" );
    }
    pThreadGUI->quit( );
    pThreadGUI->wait( );
    std::printf( "(main) all threads terminated successfully\n" );

    //ds get end time
    const double dDurationSeconds = CLogger::getTimeSeconds( )-dTimeStart;
    const uint64_t uFrameCount    = cTracker.getFrameCount( );

    std::printf( "(main) dataset completed\n" );
    std::printf( "(main) duration: %fs\n", dDurationSeconds );
    std::printf( "(main) total frames: %lu\n", uFrameCount );
    std::printf( "(main) frame rate (avg): %f FPS\n", uFrameCount/dDurationSeconds );

    if( 1 < uFrameCount )
    {
        //ds generate full file names
        const std::string strG2ODump( "/home/dominik/libs/g2o/bin/graphs/graph_" + CLogger::getTimestamp( ) );

        //ds dump file
        cTracker.saveUVDepthOrDisparity( strG2ODump );
        cTracker.saveXYZ( strG2ODump );
        cTracker.saveUVDepth( strG2ODump );
        cTracker.saveUVDisparity( strG2ODump );
        cTracker.saveCOMBO( strG2ODump );

        std::printf( "(main) successfully written g2o dump to: %s_TYPE.g2o\n", strG2ODump.c_str( ) );
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
        CLinearAccelerationIMU vecLinearAcceleration;

        //ds parse the values (order x/z/y) TODO align coordinate systems
        issLine >> strToken >> vecLinearAcceleration[0] >> vecLinearAcceleration[1] >> vecLinearAcceleration[2] >> vecAngularVelocity[0] >> vecAngularVelocity[1] >> vecAngularVelocity[2];

        //ds rotate around X axis by 180 degrees
        vecLinearAcceleration.y( ) = -vecLinearAcceleration.y( );
        vecLinearAcceleration.z( ) = -vecLinearAcceleration.z( );
        vecAngularVelocity.y( )    = -vecAngularVelocity.y( );
        vecAngularVelocity.z( )    = -vecAngularVelocity.z( );

        //ds set message fields
        msgIMU.setAngularVelocity( vecAngularVelocity );
        msgIMU.setLinearAcceleration( vecLinearAcceleration );

        //ds pump it into the synchronizer
        g_vecMessagesIMU.push( msgIMU );
    }
    else if( "IMAGE0" == strMessageType )
    {
        //ds camera_0 message
        g_pActiveMessageCameraRIGHT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_0", "camera_0", g_uFrameIDCamera_0, dTimeSeconds ) );

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
        g_pActiveMessageCameraLEFT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_1", "camera_1", g_uFrameIDCamera_1, dTimeSeconds ) );

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
