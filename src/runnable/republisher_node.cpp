#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>

//ds ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>

//ds custom
#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "utility/CLogger.h"

//ds fake session counters
uint64_t g_uFrameIDIMU      = 0;
uint64_t g_uFrameIDCameraRIGHT = 0;
uint64_t g_uFrameIDCameraLEFT  = 0;
uint64_t g_uFrameIDPose     = 0;

//ds publishers
image_transport::Publisher g_cPublisherCameraLEFT;
image_transport::Publisher g_cPublisherCameraRIGHT;
ros::Publisher g_cPublisherIMU;

inline void readNextMessageFromFile( std::ifstream& p_ifMessages, const std::string& p_strImageFolder, const uint32_t& p_uSleepMicroseconds );

int main( int argc, char **argv )
{
    //assert( false );

    //ds pwd info
    CLogger::openBox( );
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds setup node
    ros::init( argc, argv, "republisher_node" );
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
    std::string strInfileMessageDump = "/home/n551jw/ros_dumps_alberto/good_solution/data.log";
    std::string strImageFolder       = "/home/n551jw/ros_dumps_alberto/good_solution/images/";

    //ds open the file
    std::ifstream ifMessages( strInfileMessageDump, std::ifstream::in );

    //ds internals
    uint32_t uFrequencyPlaybackHz( 100 );
    uint32_t uSleepMicroseconds( ( 1.0/uFrequencyPlaybackHz )*1e6 );

    //ds instantiate publishers
    image_transport::ImageTransport itTransportCameraLEFT( hNode );
    image_transport::ImageTransport itTransportCameraRIGHT( hNode );
    g_cPublisherCameraLEFT  = itTransportCameraLEFT.advertise( "/thin_visensor_node/camera_left/image_raw", 1 );
    g_cPublisherCameraRIGHT = itTransportCameraRIGHT.advertise( "/thin_visensor_node/camera_right/image_raw", 1 );
    g_cPublisherIMU         = hNode.advertise< sensor_msgs::Imu >( "/thin_visensor_node/imu_adis16448", 1 );

    //ds log configuration
    std::printf( "(main) ROS Node namespace   := '%s'\n", hNode.getNamespace( ).c_str( ) );
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
        //ds fields
        Eigen::Vector3d vecAngularVelocity;
        Eigen::Vector3d vecLinearAcceleration;

        //ds parse the values
        issLine >> strToken >> vecLinearAcceleration[0] >> vecLinearAcceleration[1] >> vecLinearAcceleration[2] >> vecAngularVelocity[0] >> vecAngularVelocity[1] >> vecAngularVelocity[2];

        //ds build the IMU message
        sensor_msgs::Imu msgIMU;
        msgIMU.header.stamp    = ros::Time( dTimeSeconds );
        msgIMU.header.seq      = g_uFrameIDIMU;
        msgIMU.header.frame_id = "imu_adis16448";

        //ds COMPASS: orientation
        msgIMU.orientation.x   = 0.0;
        msgIMU.orientation.y   = 0.0;
        msgIMU.orientation.z   = 0.0;
        msgIMU.orientation.w   = 1.0;
        msgIMU.orientation_covariance[0] = 99999.9;
        msgIMU.orientation_covariance[1] = 0.0;
        msgIMU.orientation_covariance[2] = 0.0;
        msgIMU.orientation_covariance[3] = 0.0;
        msgIMU.orientation_covariance[4] = 99999.9;
        msgIMU.orientation_covariance[5] = 0.0;
        msgIMU.orientation_covariance[6] = 0.0;
        msgIMU.orientation_covariance[7] = 0.0;
        msgIMU.orientation_covariance[8] = 99999.9;

        //ds GYROSCOPE: angular Velocity
        msgIMU.angular_velocity.x = vecAngularVelocity[0];
        msgIMU.angular_velocity.y = vecAngularVelocity[1];
        msgIMU.angular_velocity.z = vecAngularVelocity[2];
        msgIMU.angular_velocity_covariance[0] = 0.0;
        msgIMU.angular_velocity_covariance[1] = 0.0;
        msgIMU.angular_velocity_covariance[2] = 0.0;
        msgIMU.angular_velocity_covariance[3] = 0.0;
        msgIMU.angular_velocity_covariance[4] = 0.0;
        msgIMU.angular_velocity_covariance[5] = 0.0;
        msgIMU.angular_velocity_covariance[6] = 0.0;
        msgIMU.angular_velocity_covariance[7] = 0.0;
        msgIMU.angular_velocity_covariance[8] = 0.0;

        //ds ACCELEROMETER: linear Acceleration
        msgIMU.linear_acceleration.x = vecLinearAcceleration[0];
        msgIMU.linear_acceleration.y = vecLinearAcceleration[1];
        msgIMU.linear_acceleration.z = vecLinearAcceleration[2];
        msgIMU.linear_acceleration_covariance[0] = 0.0;
        msgIMU.linear_acceleration_covariance[1] = 0.0;
        msgIMU.linear_acceleration_covariance[2] = 0.0;
        msgIMU.linear_acceleration_covariance[3] = 0.0;
        msgIMU.linear_acceleration_covariance[4] = 0.0;
        msgIMU.linear_acceleration_covariance[5] = 0.0;
        msgIMU.linear_acceleration_covariance[6] = 0.0;
        msgIMU.linear_acceleration_covariance[7] = 0.0;
        msgIMU.linear_acceleration_covariance[8] = 0.0;

        //ds publish all messages
        g_cPublisherIMU.publish( msgIMU );

        ++g_uFrameIDIMU;
    }
    else if( "IMAGE0" == strMessageType )
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
        msgHeader.seq      = g_uFrameIDCameraRIGHT;
        msgHeader.frame_id = "camera_right";

        //ds publish the image
        g_cPublisherCameraRIGHT.publish( cv_bridge::CvImage( msgHeader, "mono8", matImage ).toImageMsg( ) );

        ++g_uFrameIDCameraRIGHT;
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
        msgHeader.seq      = g_uFrameIDCameraLEFT;
        msgHeader.frame_id = "camera_left";

        //ds publish the image
        g_cPublisherCameraLEFT.publish( cv_bridge::CvImage( msgHeader, "mono8", matImage ).toImageMsg( ) );

        ++g_uFrameIDCameraLEFT;
    }
    else
    {
        //ds pose message
    }

    usleep( p_uSleepMicroseconds );
}
