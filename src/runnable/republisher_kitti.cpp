#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>
#include <sys/stat.h>

//ds ROS
//#include <ros/ros.h>
//#include <image_transport/image_transport.h>
//#include <cv_bridge/cv_bridge.h>

//ds custom
#include "txt_io/pinhole_image_message.h"
#include "utility/CLogger.h"

//ds fake session counters
uint64_t g_uFrameIDCameraRIGHT = 0;
uint64_t g_uFrameIDCameraLEFT  = 0;

//ds publishers
//image_transport::Publisher g_cPublisherCameraLEFT;
//image_transport::Publisher g_cPublisherCameraRIGHT;

//ds message buffer
std::ofstream g_strOutfile;

//ds camera projection matrices - constant for KITTI (2015-10-19)
const double arrProjectionLEFT[12]  = { 7.188560000000e+02,
                                        0.000000000000e+00,
                                        6.071928000000e+02,
                                        0.000000000000e+00,
                                        0.000000000000e+00,
                                        7.188560000000e+02,
                                        1.852157000000e+02,
                                        0.000000000000e+00,
                                        0.000000000000e+00,
                                        0.000000000000e+00,
                                        1.000000000000e+00,
                                        0.000000000000e+00 };
const double arrProjectionRIGHT[12] = { 7.188560000000e+02,
                                        0.000000000000e+00,
                                        6.071928000000e+02,
                                        -3.861448000000e+02,
                                        0.000000000000e+00,
                                        7.188560000000e+02,
                                        1.852157000000e+02,
                                        0.000000000000e+00,
                                        0.000000000000e+00,
                                        0.000000000000e+00,
                                        1.000000000000e+00,
                                        0.000000000000e+00 };

const Eigen::Vector3d vecTranslationToRIGHT( -0.54, 0, 0 );
const MatrixProjection matProjectionLEFT( Eigen::Matrix< double, 4, 3 >( arrProjectionLEFT ).transpose( ) );
const MatrixProjection matProjectionRIGHT( Eigen::Matrix< double, 4, 3 >( arrProjectionRIGHT ).transpose( ) );

inline void readNextMessageFromFile( const double& p_dTimestampSeconds,
                                     const std::string& p_strImageFolderLEFT,
                                     const std::string& p_strImageFolderRIGHT,
                                     const uint32_t& p_uSleepMicroseconds,
                                     const std::string& p_strOutfile );

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    if( 2 != argc )
    {
        std::printf( "(main) please specify the outfile\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds setup node
    //ros::init( argc, argv, "republisher_node_kitti" );
    //ros::NodeHandle hNode;

    /*ds escape here on failure
    if( !hNode.ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }*/

    //ds default files
    std::string strInfileTimestamps = "/media/n551jw/data/n551jw/Downloads/dataset/sequences/21/times.txt";
    std::string strImageFolderLEFT  = "/media/n551jw/data/n551jw/Downloads/dataset/sequences/21/image_0/";
    std::string strImageFolderRIGHT = "/media/n551jw/data/n551jw/Downloads/dataset/sequences/21/image_1/";
    std::string strOutfile          = argv[1];

    //ds open outfile
    g_strOutfile.open( strOutfile, std::ofstream::out );

    //ds on failure
    if( !g_strOutfile.good( ) )
    {
        std::printf( "(main) unable to create outfile\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds create image file directory
    const std::string strOutfileDirectory( strOutfile+".d/" );
    if( 0 != mkdir( strOutfileDirectory.c_str( ), 0700 ) )
    {
        std::printf( "(main) unable to create image directory\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds open the timestamp file
    std::ifstream ifTimestamps( strInfileTimestamps, std::ifstream::in );

    //ds timestamps (changing)
    std::vector< double > vecTimestampsSeconds;

    try
    {
        //ds compute timing
        while( ifTimestamps.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifTimestamps, strLineBuffer );

            //ds check if nothing was read
            if( strLineBuffer.empty( ) )
            {
                //ds escape
                break;
            }
            else
            {
                //ds add it to the vector
                vecTimestampsSeconds.push_back( std::stod( strLineBuffer ) );
            }
        }
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "\n(main) ERROR: unable to parse timestamps file, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    assert( 1 < vecTimestampsSeconds.size( ) );
    std::printf( "(main) successfully loaded timestamps: %lu\n", vecTimestampsSeconds.size( ) );

    /*ds instantiate publishers
    image_transport::ImageTransport itTransportCameraLEFT( hNode );
    image_transport::ImageTransport itTransportCameraRIGHT( hNode );
    g_cPublisherCameraLEFT  = itTransportCameraLEFT.advertise( "/thin_visensor_node/camera_left/image_raw", 1 );
    g_cPublisherCameraRIGHT = itTransportCameraRIGHT.advertise( "/thin_visensor_node/camera_right/image_raw", 1 );*/

    //ds log configuration
    CLogger::openBox( );
    //std::printf( "(main) ROS Node namespace   := '%s'\n", hNode.getNamespace( ).c_str( ) );
    std::printf( "(main) strInfileTimestamps  := '%s'\n", strInfileTimestamps.c_str( ) );
    std::printf( "(main) strImageFolderLEFT   := '%s'\n", strImageFolderLEFT.c_str( ) );
    std::printf( "(main) strImageFolderRIGHT  := '%s'\n", strImageFolderRIGHT.c_str( ) );
    std::printf( "(main) strOutfile           := '%s'\n", strOutfile.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds current index in the pump
    std::vector< double >::size_type uIndex = 0;

    std::printf( "(main) press [ENTER] to start playback\n" );
    while( -1 == getchar( ) )
    {
        usleep( 1 );
    }
    std::printf( "\n(main) streaming to file\n" );

    //ds playback the dump
    while( uIndex < vecTimestampsSeconds.size( )-1 )
    {
        //ds compute delta timestamp
        const double dTimestampDeltaSeconds = vecTimestampsSeconds[uIndex+1]-vecTimestampsSeconds[uIndex];

        //ds to microseconds for sleeping
        const uint32_t uSleepMicroseconds = dTimestampDeltaSeconds*1e6;

        //ds read a message
        readNextMessageFromFile( vecTimestampsSeconds[uIndex], strImageFolderLEFT, strImageFolderRIGHT, uSleepMicroseconds, strOutfileDirectory );

        //ds info
        std::printf( "remaining time: %f\n", vecTimestampsSeconds.back( )-vecTimestampsSeconds[uIndex] );
        std::fflush( stdout );

        //ds go on
        ++uIndex;
    }

    //ds done
    g_strOutfile.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

inline void readNextMessageFromFile( const double& p_dTimestampSeconds,
                                     const std::string& p_strImageFolderLEFT,
                                     const std::string& p_strImageFolderRIGHT,
                                     const uint32_t& p_uSleepMicroseconds,
                                     const std::string& p_strOutfile )
{
    //ds parse LEFT image - build image file name
    char chBufferLEFT[10];
    std::snprintf( chBufferLEFT, 7, "%06lu", g_uFrameIDCameraLEFT );
    const std::string strImageFileLEFT = p_strImageFolderLEFT + chBufferLEFT + ".png";

    //ds read the image
    cv::Mat matImageLEFT = cv::imread( strImageFileLEFT, cv::IMREAD_GRAYSCALE );

    /*ds image header
    std_msgs::Header msgHeaderLEFT;
    msgHeaderLEFT.stamp    = ros::Time( p_dTimestampSeconds );
    msgHeaderLEFT.seq      = g_uFrameIDCameraLEFT;
    msgHeaderLEFT.frame_id = "camera_left";*/

    //ds parse RIGHT image - build image file name
    char chBufferRIGHT[10];
    std::snprintf( chBufferRIGHT, 7, "%06lu", g_uFrameIDCameraRIGHT );
    const std::string strImageFileRIGHT = p_strImageFolderRIGHT + chBufferRIGHT + ".png";

    //ds read the image
    cv::Mat matImageRIGHT = cv::imread( strImageFileRIGHT, cv::IMREAD_GRAYSCALE );

    /*ds image header
    std_msgs::Header msgHeaderRIGHT;
    msgHeaderRIGHT.stamp    = ros::Time( p_dTimestampSeconds );
    msgHeaderRIGHT.seq      = g_uFrameIDCameraRIGHT;
    msgHeaderRIGHT.frame_id = "camera_right";*/

    //ds synchronization enforced
    assert( g_uFrameIDCameraLEFT == g_uFrameIDCameraRIGHT );

    //ds create pinhole messages
    txt_io::PinholeImageMessage cMessageLEFT( "/thin_visensor_node/camera_left/image_raw", "camera_left", g_uFrameIDCameraLEFT, p_dTimestampSeconds );
    txt_io::PinholeImageMessage cMessageRIGHT( "/thin_visensor_node/camera_right/image_raw", "camera_right", g_uFrameIDCameraRIGHT, p_dTimestampSeconds );

    //ds set images
    cMessageLEFT.setBinaryFilePrefix( p_strOutfile );
    cMessageRIGHT.setBinaryFilePrefix( p_strOutfile );
    cMessageLEFT.setImage( matImageLEFT );
    cMessageRIGHT.setImage( matImageRIGHT );

    //ds write to stream
    g_strOutfile << "PINHOLE_IMAGE_MESSAGE ";
    cMessageLEFT.toStream( g_strOutfile );
    g_strOutfile << "\nPINHOLE_IMAGE_MESSAGE ";
    cMessageRIGHT.toStream( g_strOutfile );
    g_strOutfile << "\n";

    //ds publish the images
    //g_cPublisherCameraLEFT.publish( cv_bridge::CvImage( msgHeaderLEFT, "mono8", matImageLEFT ).toImageMsg( ) );
    //g_cPublisherCameraRIGHT.publish( cv_bridge::CvImage( msgHeaderRIGHT, "mono8", matImageRIGHT ).toImageMsg( ) );
    ++g_uFrameIDCameraLEFT;
    ++g_uFrameIDCameraRIGHT;

    //ds maintain playback speed
    //usleep( p_uSleepMicroseconds );
}
