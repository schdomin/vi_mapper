#include <opencv/highgui.h>
#include <qapplication.h>
#include <stack>

//ds custom
#include "txt_io/message_reader.h"
#include "core/CTrackerStereoMotionModel.h"
#include "gui/CViewerScene.h"

//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds defaults
    std::string strMode              = "benchmark";
    std::string strInfileMessageDump = "/home/dominik/ros_dumps/dump_printer_room_sideways.txt";

    //ds get params
    setParametersNaive( argc, argv, strMode, strInfileMessageDump );

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

    //ds start the qt application
    QApplication cApplicationQT( argc, argv );

    //ds internals
    uint32_t uWaitKeyTimeout( 1 );

    //ds adjust depending on mode
    switch( eMode )
    {
        case ePlaybackStepwise:
        {
            uWaitKeyTimeout = 0;
            break;
        }
        case ePlaybackBenchmark:
        {
            uWaitKeyTimeout = 1;
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

    //ds message loop
    txt_io::MessageReader cMessageReader;
    cMessageReader.open( strInfileMessageDump );

    //ds escape here on failure
    if( !cMessageReader.good( ) )
    {
        std::printf( "(main) unable to open message file: %s\n", strInfileMessageDump.c_str( ) );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds allocated loggers
    CLogger::CLogDetectionEpipolar::open( );
    CLogger::CLogLandmarkCreation::open( );
    CLogger::CLogLandmarkFinal::open( );
    CLogger::CLogLandmarkFinalOptimized::open( );
    CLogger::CLogOptimizationOdometry::open( );
    CLogger::CLogTrajectory::open( );
    CLogger::CLogLinearAcceleration::open( );

    //ds log configuration
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds allocate the tracker
    CTrackerStereoMotionModel cTracker( eMode, uWaitKeyTimeout );

    //ds allocate a libqglviewer
    CViewerScene cViewer( cTracker.getLandmarksHandle( ) );
    cViewer.setWindowTitle( "CViewerScene: WORLD" );
    cViewer.showMaximized( );

    //ds stop time
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds message holders
    std::shared_ptr< txt_io::CIMUMessage > pMessageIMU( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

    //ds playback the dump
    while( cMessageReader.good( ) && !cTracker.isShutdownRequested( ) )
    {
        std::fflush( stdout );

        //ds check if viewer is still active
        if( cViewer.isVisible( ) )
        {
            //ds check for frames to update the viewer
            if( cTracker.isFrameAvailable( ) ){ cViewer.addFrame( cTracker.getFrameLEFTtoWORLD( ) ); }
        }

        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
        {
            //ds trigger callbacks artificially - check for imu input first
            if( "IMU_MESSAGE" == msgBase->tag( ) )
            {
                //ds IMU message
                pMessageIMU = std::shared_ptr< txt_io::CIMUMessage >( dynamic_cast< txt_io::CIMUMessage* >( msgBase ) );
            }
            else
            {
                //ds camera message
                std::shared_ptr< txt_io::PinholeImageMessage > pMessageImage( dynamic_cast< txt_io::PinholeImageMessage* >( msgBase ) );

                //ds if its the left camera
                if( "camera_left" == pMessageImage->frameId( ) )
                {
                    pMessageCameraLEFT  = pMessageImage;
                }
                else
                {
                    pMessageCameraRIGHT = pMessageImage;
                }
            }
        }

        //ds as soon as we have data in all the stacks - process
        if( 0 != pMessageCameraLEFT && 0 != pMessageCameraRIGHT && 0 != pMessageIMU )
        {
            //ds synchronization expected
            assert( pMessageCameraLEFT->timestamp( ) == pMessageCameraRIGHT->timestamp( ) );
            assert( pMessageIMU->timestamp( )        == pMessageCameraLEFT->timestamp( ) );
            assert( pMessageIMU->timestamp( )        == pMessageCameraRIGHT->timestamp( ) );

            //ds callback with triplet
            cTracker.receivevDataVI( pMessageCameraLEFT, pMessageCameraRIGHT, pMessageIMU );

            //ds reset holders
            pMessageCameraLEFT.reset( );
            pMessageCameraRIGHT.reset( );
            pMessageIMU.reset( );

            //ds check reset
            assert( 0 == pMessageCameraLEFT );
            assert( 0 == pMessageCameraRIGHT );
            assert( 0 == pMessageIMU );
        }
    }

    //ds get end time
    const double dDuration     = CLogger::getTimeSeconds( )-dTimeStartSeconds;
    const uint64_t uFrameCount = cTracker.getFrameCount( );

    std::printf( "(main) dataset completed\n" );
    std::printf( "(main) duration: %fs\n", dDuration );
    std::printf( "(main) total frames: %lu\n", uFrameCount );
    std::printf( "(main) frame rate (avg): %f FPS\n", uFrameCount/dDuration );

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
    if( cViewer.isVisible( ) )
    {
        return cApplicationQT.exec( );
    }
    else
    {
        return 0;
    }
}

void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages )
{
    //ds attribute names (C style for printf)
    const char* arrParameter1 = "-mode";
    const char* arrParameter2 = "-messages";

    try
    {
        //ds parse optional command line arguments
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

        //ds set parameters if found
        if( vecCommandLineArguments.end( ) != itParameter1 ){ p_strMode                    = *( itParameter1+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter2 ){ p_strInfileCameraIMUMessages = *( itParameter2+1 ); }
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_slam %s %s\n", arrParameter1, arrParameter2 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_slam %s %s\n", arrParameter1, arrParameter2 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
}
