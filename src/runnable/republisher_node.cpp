#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>

//ds ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

//ds custom
#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "utility/CStack.h"
#include "utility/CLogger.h"

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCamera_0 = 0;
uint64_t g_uFrameIDCamera_1 = 0;
uint64_t g_uFrameIDPose     = 0;

//ds publishers
image_transport::Publisher g_cPublisherCameraLEFT;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder, const uint32_t& p_uSleepMicroseconds );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds image resolution
    const uint32_t uImageRows = 480;
    const uint32_t uImageCols = 752;

    //ds setup node
    ros::init( argc, argv, "republisher_alberto" );
    ros::NodeHandle hNode;

    //ds escape here on failure
    if( !hNode.ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds default files
    std::string strInfileMessageDump = "/home/dominik/ros_bags/datasets4dominik/good_solution/solution.log";
    std::string strImageFolder       = "/home/dominik/ros_bags/datasets4dominik/good_solution/images/";

    //ds open the file
    std::ifstream ifMessages( strInfileMessageDump, std::ifstream::in );

    //ds internals
    uint32_t uFrequencyPlaybackHz( 200 );
    uint32_t uSleepMicroseconds( ( 1.0/uFrequencyPlaybackHz )*1e6 );

    //ds instantiate publishers
    image_transport::ImageTransport itTransportCameraLEFT( hNode );
    g_cPublisherCameraLEFT = itTransportCameraLEFT.advertise( "camera/image_raw", 1 );

    //ds log configuration
    std::printf( "(main) ROS Node namespace   := '%s'\n", hNode.getNamespace( ).c_str( ) );
    std::printf( "(main) uImageRows (height)  := '%u'\n", uImageRows );
    std::printf( "(main) uImageCols (width)   := '%u'\n", uImageCols );
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) uFrequencyPlaybackHz := '%u'\n", uFrequencyPlaybackHz );
    std::printf( "(main) uSleepMicroseconds   := '%u'\n", uSleepMicroseconds );
    std::fflush( stdout );
    CLogger::closeBox( );

    //ds playback the dump
    while( ifMessages.good( ) && ros::ok( ) )
    {
        //ds read a message
        readNextMessageFromFile( ifMessages, strImageFolder, uSleepMicroseconds );
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
    }
    else if( "IMAGE0" == strMessageType )
    {
        //ds camera_0 message
    }
    else if( "IMAGE1" == strMessageType )
    {
        //ds set image information
        std::string strImageFile;

        //ds get parameter
        issLine >> strToken >> strImageFile;

        //ds fix filepath
        strImageFile = p_strImageFolder + strImageFile.substr( 7, 18 );

        //ds read the image
        cv::Mat matImage = cv::imread( strImageFile, cv::IMREAD_GRAYSCALE );

        //ds image header
        std_msgs::Header msgHeader;
        msgHeader.stamp    = ros::Time( dTimeSeconds );
        msgHeader.seq      = g_uFrameIDCamera_1;
        msgHeader.frame_id = g_uFrameIDCamera_1;

        //ds publish the image
        g_cPublisherCameraLEFT.publish( cv_bridge::CvImage( msgHeader, "mono8", matImage ).toImageMsg( ) );

        ++g_uFrameIDCamera_1;
    }
    else
    {
        //ds pose message
    }

    usleep( p_uSleepMicroseconds );
}
