#include <opencv/highgui.h>
#include <qapplication.h>
#include <stack>

//ds custom
#include "txt_io/message_reader.h"
#include "exceptions/CExceptionLogfileTree.h"
#include "core/CTrackerStereoMotionModelKITTI.h"
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
    std::string strInfileMessageDump = "/home/n551jw/ros_dumps/dump_printer_room_sideways.txt";

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
        std::printf( "(main) terminated: %s\n", argv[0] );
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

    //ds log configuration
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    //std::printf( "(main) openCV build information: \n%s", cv::getBuildInformation( ).c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds stop time
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds message holders
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

    //ds allocate the tracker
    CTrackerStereoMotionModelKITTI cTracker( eMode, uWaitKeyTimeout );
    try
    {
        //ds prepare file structure
        cTracker.sanitizeFiletree( );
    }
    catch( const CExceptionLogfileTree& p_cException )
    {
        std::printf( "(main) unable to sanitize filetree - exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds count invalid frames
    UIDFrame uInvalidFrames = 0;

    //ds allocate a libqglviewer
    CViewerScene cViewer( cTracker.getLandmarksHandle( ), cTracker.getKeyFramesHandle( ), cTracker.getLoopClosingRadius( ) );
    cViewer.setWindowTitle( "CViewerScene: WORLD" );
    cViewer.showMaximized( );

    //ds playback the dump
    while( cMessageReader.good( ) && !cTracker.isShutdownRequested( ) )
    {
        //ds check if viewer is still active
        if( cViewer.isVisible( ) )
        {
            //ds check for frames to update the viewer
            if( cTracker.isFrameAvailable( ) ){ cViewer.addFrame( cTracker.getFrameLEFTtoWORLD( ) ); }
            //if( cTracker.isFrameAvailableSlidingWindow( ) ){ cViewer.updateFrame( cTracker.getFrameLEFTtoWORLDSlidingWindow( ), true ); }
            cViewer.manualDraw( );
        }

        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
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

        //ds as soon as we have data in all the stacks - process
        if( 0 != pMessageCameraLEFT && 0 != pMessageCameraRIGHT )
        {
            //ds if the timestamps match (optimally the case)
            if( pMessageCameraLEFT->timestamp( ) == pMessageCameraRIGHT->timestamp( ) )
            {
                //ds callback with pair
                cTracker.receivevDataVI( pMessageCameraLEFT, pMessageCameraRIGHT );

                //ds reset holders
                pMessageCameraLEFT.reset( );
                pMessageCameraRIGHT.reset( );

                //ds check reset
                assert( 0 == pMessageCameraLEFT );
                assert( 0 == pMessageCameraRIGHT );
            }
            else
            {
                //ds check timestamp mismatch
                if( pMessageCameraLEFT->timestamp( ) < pMessageCameraRIGHT->timestamp( ) )
                {
                    std::printf( "(main) timestamp mismatch LEFT: %f < RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraLEFT.reset( );
                }
                else
                {
                    std::printf( "(main) timestamp mismatch LEFT: %f > RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraRIGHT.reset( );
                }

                ++uInvalidFrames;
            }
        }
    }

    //ds get end time
    const double dDuration       = CLogger::getTimeSeconds( )-dTimeStartSeconds;
    const UIDFrame uFrameCount   = cTracker.getFrameCount( );
    const double dDurationOnline = uFrameCount/20.0;
    const double dDistance       = cTracker.getDistanceTraveled( );

    if( 1 < uFrameCount )
    {
        //ds finalize tracker (e.g. do a last optimization)
        cTracker.finalize( );

        //ds and update the viewer
        if( cViewer.isVisible( ) )
        {
            cViewer.manualDraw( );
        }
    }

    //ds summary
    CLogger::openBox( );
    std::printf( "(main) dataset completed\n" );
    std::printf( "(main) distance traveled: %fm\n", dDistance );
    std::printf( "(main) duration: %fs (online: %fs, x%f)\n", dDuration, dDurationOnline, dDuration/dDurationOnline );
    std::printf( "(main) traveling speed (online): %fm/s\n", dDistance/dDurationOnline );
    std::printf( "(main) total frames: %lu\n", uFrameCount );
    std::printf( "(main) frame rate (avg): %f FPS\n", uFrameCount/dDuration );
    std::printf( "(main) invalid frames: %li (%4.2f)\n", uInvalidFrames, static_cast< double >( uInvalidFrames )/uFrameCount );
    CLogger::closeBox( );

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
