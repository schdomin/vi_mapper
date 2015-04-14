//ds std
#include <iostream>
#include <thread>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>

//ds ROS
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

//ds custom
#include <utility/CStack.h>

//ds UGLY global stacks
CStack< sensor_msgs::ImageConstPtr > g_stackMessagesCamera_0;
CStack< sensor_msgs::ImageConstPtr > g_stackMessagesCamera_1;
CStack< sensor_msgs::ImuConstPtr > g_stackMessagesIMU;

//ds sensor topic callbacks
void callbackCamera_0( const sensor_msgs::ImageConstPtr p_msgImage ){ g_stackMessagesCamera_0.push( p_msgImage ); }
void callbackCamera_1( const sensor_msgs::ImageConstPtr p_msgImage ){ g_stackMessagesCamera_1.push( p_msgImage ); }
void callbackIMU( const  sensor_msgs::ImuConstPtr p_msgIMU ){ g_stackMessagesIMU.push( p_msgIMU ); }

//ds synchronization
void handleMessageTriplets( );

int main( int argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strTopicCamera_0   = "/thin_visensor_node/camera_0/image_raw";
    std::string strTopicCamera_1   = "/thin_visensor_node/camera_1/image_raw";
    std::string strTopicIMU        = "/thin_visensor_node/imu_adis16448";
    std::string strBaseLinkFrameID = "";
    std::string strTopicOdometry   = "/odom";
    uint32_t uMaximumQueueSize     = 1000;

    //ds setup node
    ros::init( argc, argv, "vi_message_dumper_node" );
    boost::shared_ptr< ros::NodeHandle > hNode( new ros::NodeHandle( "~" ) );

    //ds escape here on failure
    if( !hNode->ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) ---------------------------------------------- CONFIGURATION ----------------------------------------------\n" );
    std::printf( "(main) ROS Node namespace := '%s'\n", hNode->getNamespace( ).c_str( ) );
    std::printf( "(main) strTopicCamera_0   := '%s'\n", strTopicCamera_0.c_str( ) );
    std::printf( "(main) strTopicCamera_1   := '%s'\n", strTopicCamera_1.c_str( ) );
    std::printf( "(main) strTopicIMU        := '%s'\n", strTopicIMU.c_str( ) );
    std::printf( "(main) strBaseLinkFrameID := '%s'\n", strBaseLinkFrameID.c_str( ) );
    std::printf( "(main) strTopicOdometry   := '%s'\n", strTopicOdometry.c_str( ) );
    std::printf( "(main) uMaximumQueueSize  := '%u'\n", uMaximumQueueSize );
    std::printf( "(main) -----------------------------------------------------------------------------------------------------------\n" );
    std::fflush( stdout );

    //ds subscribe to visensor topics
    ros::Subscriber cSubscriberCamera0 = hNode->subscribe( strTopicCamera_0, uMaximumQueueSize, callbackCamera_0 );
    ros::Subscriber cSubscriberCamera1 = hNode->subscribe( strTopicCamera_1, uMaximumQueueSize, callbackCamera_1 );
    ros::Subscriber cSubscribeIMU0     = hNode->subscribe( strTopicIMU, uMaximumQueueSize, callbackIMU );

    //ds launch threads
    std::thread hTFPublisher( handleMessageTriplets );

    //ds main loop - pump callbacks
    ros::spin( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

void handleMessageTriplets( )
{
    std::printf( "(handleTripletMessages) thread launched\n" );

    //ds hold steady
    while( ros::ok( ) )
    {
        //ds as long as we have data in all the stacks - process
        while( !g_stackMessagesCamera_0.isEmpty( ) && !g_stackMessagesCamera_1.isEmpty( ) && !g_stackMessagesIMU.isEmpty( ) && ros::ok( ) )
        {
            //ds pop the camera images
            sensor_msgs::ImageConstPtr pImageCamera_0( g_stackMessagesCamera_0.pop( ) );
            sensor_msgs::ImageConstPtr pImageCamera_1( g_stackMessagesCamera_1.pop( ) );

            //ds current triplet timestamp
            const ros::Time tmTimestamp( pImageCamera_0->header.stamp );

            //ds timestamps have to match
            if( tmTimestamp == pImageCamera_1->header.stamp )
            {
                //ds get the most recent imu measurement
                sensor_msgs::ImuConstPtr pMessageIMU( g_stackMessagesIMU.pop( ) );

                //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                while( tmTimestamp < pMessageIMU->header.stamp && !g_stackMessagesIMU.isEmpty( ) && ros::ok( ) )
                {
                    pMessageIMU = g_stackMessagesIMU.pop( );
                }

                //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                while( tmTimestamp > pMessageIMU->header.stamp && ros::ok( ) )
                {
                    //ds pop the most recent imu measurement and check
                    if( !g_stackMessagesIMU.isEmpty( ) )
                    {
                        pMessageIMU = g_stackMessagesIMU.pop( );
                    }
                }

                //ds verify timestamps (TODO remove check for better performance)
                if( tmTimestamp == pMessageIMU->header.stamp )
                {
                    //ds progress
                    std::printf( "x" );
                }
                else
                {
                    //ds unable to find matching IMU measurement
                    std::printf( "o" );
                }
            }
            else
            {
                std::printf( "\n(handleTripletMessages) ERROR: frame timestamp mismatch\n" );
            }
        }

        //ds flush buffer
        std::fflush( stdout );
    }

    std::printf( "\n(handleTripletMessages) thread terminated\n" );
}
