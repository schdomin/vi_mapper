#include <opencv/highgui.h>

//ds custom
#include "core/CMockedTrackerStereo.h"
#include "vision/CMiniVisionToolbox.h"

//ds data vectors
std::stack< txt_io::CIMUMessage > g_vecMessagesIMU;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraRIGHT = 0;
std::shared_ptr< txt_io::PinholeImageMessage > g_pActiveMessageCameraLEFT = 0;
std::shared_ptr< txt_io::CPoseMessage > g_pActiveMessagesPose           = 0;

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCameraRIGHT = 0;
uint64_t g_uFrameIDCameraLEFT = 0;
uint64_t g_uFrameIDPose     = 0;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder );

//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages,
                         std::string& p_strImageFolder,
                         std::string& p_strInfileMockedLandmarks );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds defaults
    std::string strMode              = "stepwise";
    std::string strInfileMessageDump = "/home/dominik/ros_bags/datasets4dominik/good_solution/solution.log";
    std::string strImageFolder       = "/home/dominik/ros_bags/datasets4dominik/good_solution/images/";
    std::string strMockedLandmarks   = "/home/dominik/workspace_catkin/src/vi_mapper/mocking/landmarks_clean_level2.txt";

    //ds get params
    setParametersNaive( argc, argv, strMode, strInfileMessageDump, strImageFolder, strMockedLandmarks );

    //ds get playback mode to enum
    EPlaybackMode eMode( ePlaybackStepwise );

    if( "stepwise" == strMode )
    {
        eMode = ePlaybackStepwise;
    }
    else if( "benchmark" == strMode )
    {
        eMode = ePlaybackBenchmark;
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
            break;
        }
    }

    //ds logfiles used (will be closed independently)
    CLogger::CLogLandmarkCreationMocked::open( );
    CLogger::CLogDetectionEpipolar::open( );
    CLogger::CLogLandmarkFinal::open( );
    CLogger::CLogLandmarkFinalOptimized::open( );
    CLogger::CLogOptimizationOdometry::open( );
    CLogger::CLogTrajectory::open( );

    //ds log configuration
    std::printf( "(main) strMode              := '%s'\n", strMode.c_str( ) );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) strImageFolder       := '%s'\n", strImageFolder.c_str( ) );
    std::printf( "(main) strMockedLandmarks   := '%s'\n", strMockedLandmarks.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds feature detector
    CMockedTrackerStereo cDetector( eMode, strMockedLandmarks, uWaitKeyTimeout );

    //ds get start time
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds playback the dump
    while( ifMessages.good( ) && !cDetector.isShutdownRequested( ) )
    {
        //ds read a message
        readNextMessageFromFile( ifMessages, strImageFolder );

        //ds as long as we have data in all the stacks - process
        if( 0 != g_pActiveMessageCameraRIGHT && 0 != g_pActiveMessageCameraLEFT && !g_vecMessagesIMU.empty( ) && 0 != g_pActiveMessagesPose )
        {
            //ds pop the camera images
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCameraRIGHT( g_pActiveMessageCameraRIGHT );
            std::shared_ptr< txt_io::PinholeImageMessage > cImageCameraLEFT( g_pActiveMessageCameraLEFT );

            //ds current triplet timestamp
            double dTimestamp_0( cImageCameraRIGHT->timestamp( ) );
            double dTimestamp_1( cImageCameraLEFT->timestamp( ) );

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

                //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                while( dTimestamp_0 > cMessageIMU.timestamp( ) )
                {
                    //ds pop the most recent imu measurement and check
                    if( !g_vecMessagesIMU.empty( ) )
                    {
                        readNextMessageFromFile( ifMessages, strImageFolder );
                        cMessageIMU = g_vecMessagesIMU.top( );
                        g_vecMessagesIMU.pop( );
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
                        if( dTimestamp_0 <= g_pActiveMessagesPose->timestamp( ) )
                        {
                            //ds callback with triplet
                            cDetector.receivevDataVIWithPose( cImageCameraLEFT, cImageCameraRIGHT, cMessageIMU, g_pActiveMessagesPose );
                            g_pActiveMessagesPose = 0;
                            bCalledDetector = true;
                        }
                        else
                        {
                            //ds log skipped frame
                            std::printf( "(main) WARNING: skipped single frame (outdated pose information) - timestamp camera_right: %.5lf s\n", dTimestamp_0 );
                        }
                    }

                    //ds read a message
                    readNextMessageFromFile( ifMessages, strImageFolder );
                }

                if( !bCalledDetector )
                {
                    //ds log skipped frame
                    std::printf( "(main) WARNING: skipped single frame (no pose information) - timestamp camera_right: %.5lf s\n", dTimestamp_0 );
                }
            }
            else
            {
                std::printf( "(main) WARNING: could not find matching frames - timestamp camera_right: %.5lf s\n", dTimestamp_0 );

                //ds reset the respectively elder one
                if( dTimestamp_0 > dTimestamp_1 )
                {
                    g_pActiveMessageCameraLEFT = 0;
                    g_pActiveMessagesPose    = 0;
                }
                if( dTimestamp_0 < dTimestamp_1 )
                {
                    g_pActiveMessageCameraRIGHT = 0;
                    g_pActiveMessagesPose    = 0;
                }
            }
        }
    }

    //ds get end time
    const double dDuration     = CLogger::getTimeSeconds( )-dTimeStartSeconds;
    const uint64_t uFrameCount = cDetector.getFrameCount( );

    std::printf( "(main) dataset completed\n" );
    std::printf( "(main) duration: %fs\n", dDuration );
    std::printf( "(main) total frames: %lu\n", uFrameCount );
    std::printf( "(main) frame rate (avg): %f FPS\n", uFrameCount/dDuration );

    if( 1 < uFrameCount )
    {
        //ds generate full file names
        const std::string strG2ODump( "/home/dominik/libs/g2o/bin/graphs/mocked/graph_" + CLogger::getTimestamp( ) );

        //ds dump file
        cDetector.saveUVDepthOrDisparity( strG2ODump );
        cDetector.saveXYZ( strG2ODump );
        cDetector.saveUVDepth( strG2ODump );
        cDetector.saveUVDisparity( strG2ODump );
        cDetector.saveCOMBO( strG2ODump );

        std::printf( "(main) successfully written g2o dump to: %s_TYPE.g2o\n", strG2ODump.c_str( ) );
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

inline int8_t sign( const float& p_fNumber )
{
    assert( 0.0 != p_fNumber );

    if( 0.0 < p_fNumber ){ return 1; }
    if( 0.0 > p_fNumber ){ return -1; }

    //ds never gets called, just pleasing the compiler
    assert( false );
    return 0;
}

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder )
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

        //ds parse the values
        issLine >> strToken >> vecLinearAcceleration[0] >> vecLinearAcceleration[1] >> vecLinearAcceleration[2] >> vecAngularVelocity[0] >> vecAngularVelocity[1] >> vecAngularVelocity[2];

        //ds flip Y and Z values (TODO VERIFY)
        vecLinearAcceleration[1] = -vecLinearAcceleration[1];
        vecLinearAcceleration[2] = -vecLinearAcceleration[2];
        vecAngularVelocity[1] = -vecAngularVelocity[1];
        vecAngularVelocity[2] = -vecAngularVelocity[2];

        //ds set message fields
        msgIMU.setAngularVelocity( vecAngularVelocity );
        msgIMU.setLinearAcceleration( vecLinearAcceleration );

        //ds pump it into the synchronizer
        g_vecMessagesIMU.push( msgIMU );
    }
    else if( "IMAGE0" == strMessageType )
    {
        //ds camera_0 message
        g_pActiveMessageCameraRIGHT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_0", "camera_0", g_uFrameIDCameraRIGHT, dTimeSeconds ) );

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
        g_pActiveMessageCameraLEFT = std::shared_ptr< txt_io::PinholeImageMessage >( new txt_io::PinholeImageMessage( "/camera_1", "camera_1", g_uFrameIDCameraLEFT, dTimeSeconds ) );

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
}

void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages,
                         std::string& p_strImageFolder,
                         std::string& p_strInfileMockedLandmarks )
{
    //ds attribute names (C style for printf)
    const char* arrParameter1( "-mode" );
    const char* arrParameter2( "-messages" );
    const char* arrParameter3( "-images" );
    const char* arrParameter4( "-landmarks" );

    try
    {
        //ds parse optional command line arguments: -mode=interactive -messages="/asdasd/" -images="/asdasdgfa/" -landmarks="/asdasddsd/"
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
        const std::vector< std::string >::const_iterator itParameter4( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter4 ) );

        //ds set parameters if found
        if( vecCommandLineArguments.end( ) != itParameter1 ){ p_strMode                    = *( itParameter1+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter2 ){ p_strInfileCameraIMUMessages = *( itParameter2+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter3 ){ p_strImageFolder             = *( itParameter3+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter4 ){ p_strInfileMockedLandmarks   = *( itParameter4+1 ); }
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_detector_mocked %s %s %s %s\n", arrParameter1, arrParameter2, arrParameter3, arrParameter4 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: stereo_detector_mocked %s %s %s %s\n", arrParameter1, arrParameter2, arrParameter3, arrParameter4 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
}
