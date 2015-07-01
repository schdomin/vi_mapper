#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <omp.h>

#include "configuration/CConfigurationCamera.h"
#include "configuration/CConfigurationOpenCV.h"
#include "core/CTriangulator.h"
#include "vision/CStereoCamera.h"
#include "utility/CLogger.h"
#include "exceptions/CExceptionNoMatchFound.h"

//ds UGLY globals for testing
//ds camera setup
const std::shared_ptr< CStereoCamera > g_pCameraSTEREO( std::make_shared< CStereoCamera >( CConfigurationCamera::LEFT::cPinholeCamera, CConfigurationCamera::RIGHT::cPinholeCamera ) );

//ds feature handling
const uint32_t g_uKeyPointSize( 7 );
std::shared_ptr< cv::GoodFeaturesToTrackDetector > g_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 100, 0.01, 20.0, g_uKeyPointSize, true ) );
std::shared_ptr< cv::BriefDescriptorExtractor > g_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) );
std::shared_ptr< cv::BFMatcher > g_pMatcher( std::make_shared< cv::BFMatcher >( ) );

//ds triangulator
std::shared_ptr< CTriangulator > g_pTriangulator( 0 );

//ds display
cv::Mat g_matDisplay( cv::Mat( g_pCameraSTEREO->m_uPixelHeight, 2*g_pCameraSTEREO->m_uPixelWidth, CV_8UC3 ) );

//ds single buffers
sensor_msgs::ImageConstPtr g_pImageRIGHTLast( 0 );
sensor_msgs::ImageConstPtr g_pImageLEFTLast( 0 );

void computeDepth( const cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT )
{
    //ds preprocessed images
    cv::Mat matPreprocessedLEFT;
    cv::Mat matPreprocessedRIGHT;

    //ds preprocess images
    cv::equalizeHist( p_matImageLEFT, matPreprocessedLEFT );
    cv::equalizeHist( p_matImageRIGHT, matPreprocessedRIGHT );
    g_pCameraSTEREO->undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLEFT;
    cv::Mat matDisplayRIGHT;

    //ds get images to triple channel for colored display
    cv::cvtColor( matPreprocessedLEFT, matDisplayLEFT, cv::COLOR_GRAY2BGR );
    cv::cvtColor( matPreprocessedRIGHT, matDisplayRIGHT, cv::COLOR_GRAY2BGR );

    //ds detect features
    std::vector< cv::KeyPoint > vecKeyPoints;
    g_pDetector->detect( matPreprocessedLEFT, vecKeyPoints );

    //ds compute descriptors for the keypoints
    CDescriptor matReferenceDescriptors;
    g_pExtractor->compute( matPreprocessedLEFT, vecKeyPoints, matReferenceDescriptors );

    //ds count active points
    uint64_t uActivePoints( 0 );

    //ds errors
    double dAverageErrorSVDLS( 0.0 );
    double dAverageErrorQRLS( 0.0 );
    double dAverageErrorEDS( 0.0 );
    double dAverageErrorSVDDLT( 0.0 );
    uint64_t uPointsSVDDLS( 0 );
    uint64_t uPointsQRLS( 0 );
    uint64_t uPointsEDS( 0 );
    uint64_t uPointsSVDDLT( 0 );

    //ds process the keypoints and see if we can use them as landmarks
    for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
    {
        //ds current points
        const cv::KeyPoint cKeyPoint( vecKeyPoints[u] );
        const cv::Point2f ptLandmarkLEFT( cKeyPoint.pt );
        const CDescriptor& matReferenceDescriptorLEFT( matReferenceDescriptors.row(u) );

        //ds draw detected point
        cv::circle( matDisplayLEFT, ptLandmarkLEFT, 2, CColorCodeBGR( 255, 0, 0 ), -1 );

        //ds triangulations - set on success else null
        const CPoint3DInCameraFrame* vecPointTriangulatedSTEREOSVDLS( 0 );
        const CPoint3DInCameraFrame* vecPointTriangulatedSTEREOQRLS( 0 );
        const CPoint3DInCameraFrame* vecPointTriangulatedSTEREOEDS( 0 );
        const CPoint3DInCameraFrame* vecPointTriangulatedSTEREOSVDDLT( 0 );

        try
        {
            //ds triangulate the point
            vecPointTriangulatedSTEREOSVDLS = new CPoint3DInCameraFrame( g_pTriangulator->getPointTriangulatedLimitedSVDLS( matDisplayRIGHT, matPreprocessedRIGHT, cKeyPoint, matReferenceDescriptorLEFT ) );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            //std::cout << "exception: " << p_cException.what( ) << std::endl;
        }

        /*try
        {
            vecPointTriangulatedSTEREOQRLS  = new CPoint3DInCameraFrame( g_pTriangulator->getPointTriangulatedLimitedQRLS( matPreprocessedRIGHT, cKeyPoint, matReferenceDescriptorLEFT ) );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {

        }

        try
        {
            vecPointTriangulatedSTEREOSVDDLT = new CPoint3DInCameraFrame( g_pTriangulator->getPointTriangulatedLimitedSVDDLT( matPreprocessedRIGHT, cKeyPoint, matReferenceDescriptorLEFT ) );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {

        }*/

        /*try
        {
            std::vector< cv::KeyPoint > vecKeyPointsEDS;
            std::vector< CPoint3DInCameraFrame > vecTriangulatedPoints;

            //ds normalized coords
            const CPoint2DInCameraFrame vecLandmarkLEFTNormalized( g_pCameraSTEREO->m_pCameraLEFT->getNormalized( ptLandmarkLEFT ) );
            //const CPoint2DInCameraFrame vecLandmarkLEFTNormalized( ptLandmarkLEFT.x/g_pCameraSTEREO->m_pCameraLEFT->m_dFx, ptLandmarkLEFT.y/g_pCameraSTEREO->m_pCameraLEFT->m_dFy );*/

            /*ds calibrate scaling factor for depth of one meter (EPIPOLAR CONSTRAINT)
            double dErrorPixel( 1000.0 );
            double dVerticalFactor( 1.0 );

            #pragma omp parallel for shared( dErrorPixel, dVerticalFactor )
            for( uint32_t u = 1; u < 20; ++u )
            {
                const double dDepthSampling( u/static_cast< double >( 10.0 ) );

                for( int32_t i = 1; i < 15; ++i )
                {
                    //ds sample current solution
                    const double dVerticalFactorCurrent = i/static_cast< double >( 10.0 );
                    const CPoint3DInCameraFrame vecPointUncalibratedEDSLEFT( dVerticalFactorCurrent*dDepthSampling*vecLandmarkLEFTNormalized(0), dVerticalFactorCurrent*dDepthSampling*vecLandmarkLEFTNormalized(1), dDepthSampling );
                    //const CPoint3DInCameraFrame vecPointUncalibratedEDSRIGHT( g_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointUncalibratedEDSLEFT );
                    const double dErrorPixelCurrent( std::fabs( g_pCameraSTEREO->m_pCameraRIGHT->getProjection( vecPointUncalibratedEDSLEFT ).y-ptLandmarkLEFT.y ) );

                    if( dErrorPixel > dErrorPixelCurrent )
                    {
                        dVerticalFactor = dVerticalFactorCurrent;
                        dErrorPixel     = dErrorPixelCurrent;
                    }
                }
            }

            assert( 1000.0 != dErrorPixel );

            if( 0.1 == dVerticalFactor || 1.5 == dVerticalFactor )
            {
                throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not calibrate offset" );
            }*/

            //std::cout << "best min error: " << dErrorPixel << " (scale factor: " << dVerticalFactor << " )" << std::endl;

            //ds exponential depth sampling triangulation EDS
            //for( int32_t iExponent = -50; iExponent < 50; ++iExponent )
            /*#pragma omp parallel for
            for( uint32_t uDepthDecimeter = 1; uDepthDecimeter < 100; ++uDepthDecimeter )
            {
                //ds current depth
                const double dDepthMeter( uDepthDecimeter/static_cast< double >( 10.0 ) );

                //ds "compute triangulated point"
                const CPoint3DInCameraFrame vecPointTriangulatedEDSLEFT( vecLandmarkLEFTNormalized(0)*dDepthMeter, vecLandmarkLEFTNormalized(1)*dDepthMeter, dDepthMeter );
                //const CPoint3DInCameraFrame vecPointTriangulatedEDSRIGHT( g_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointTriangulatedEDSLEFT );

                //std::cout << vecPointTriangulatedEDSLEFT.transpose( ) << " - " << vecPointTriangulatedEDSRIGHT.transpose( ) << std::endl;

                //ds project the point into the right camera frame
                const cv::Point2d ptLandmarkRIGHTEDS( g_pCameraSTEREO->m_pCameraRIGHT->getProjection( vecPointTriangulatedEDSLEFT ) );

                if( g_pCameraSTEREO->m_cVisibleRange.contains( ptLandmarkRIGHTEDS ) )
                {
                    //ds TODO shift vector combination
                    #pragma omp critical
                    {
                        vecKeyPointsEDS.push_back( cv::KeyPoint( ptLandmarkRIGHTEDS, cKeyPoint.size ) );
                        vecTriangulatedPoints.push_back( vecPointTriangulatedEDSLEFT );
                    }
                    cv::circle( matDisplayRIGHT, ptLandmarkRIGHTEDS, 2, CColorCodeBGR( 200, 200, 200 ), -1 );
                }
                else
                {
                    //ds stop condition
                    if( !vecKeyPointsEDS.empty( ) )
                    {
                        //break;
                        uDepthDecimeter = 100;
                    }
                }
            }

            if( !vecKeyPointsEDS.empty( ) )
            {
                //ds compute descriptors for the keypoints
                CDescriptor matDescriptorsEDS;
                g_pExtractor->compute( matPreprocessedRIGHT, vecKeyPointsEDS, matDescriptorsEDS );

                //ds match the descriptors
                std::vector< cv::DMatch > vecMatches;
                g_pMatcher->match( matReferenceDescriptorLEFT, matDescriptorsEDS, vecMatches );

                if( vecMatches.empty( ) )
                {
                    throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
                }

                //ds check match quality
                if( g_dMatchingDistanceCutoffTriangulation > vecMatches[0].distance )
                {
                    vecPointTriangulatedSTEREOEDS = new CPoint3DInCameraFrame( vecTriangulatedPoints[vecMatches[0].trainIdx] );
                }
                else
                {
                    throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
                }
            }
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {

        }*/

        //ds escape if no depth has been computed
        if( 0 == vecPointTriangulatedSTEREOEDS && 0 == vecPointTriangulatedSTEREOQRLS && 0 == vecPointTriangulatedSTEREOSVDLS && 0 == vecPointTriangulatedSTEREOSVDDLT )
        {
            //ds UGLY
            continue;
        }

        std::string strDepthInfo( "" );

        if( 0 != vecPointTriangulatedSTEREOSVDLS )
        {
            //ds compute error
            const cv::Point2d ptLandmarkRIGHT( g_pCameraSTEREO->m_pCameraRIGHT->getProjection( *vecPointTriangulatedSTEREOSVDLS ) );
            const double dDisparityPixels( ptLandmarkLEFT.x-ptLandmarkRIGHT.x );
            const double dDepthMeters( ( *vecPointTriangulatedSTEREOSVDLS )(2) );
            const double dDepthErrorMeters( ( *vecPointTriangulatedSTEREOSVDLS )(2)*( *vecPointTriangulatedSTEREOSVDLS )(2)/( g_pCameraSTEREO->m_dBaselineMeters*g_pCameraSTEREO->m_pCameraLEFT->m_dFx )*dDisparityPixels );
            dAverageErrorSVDLS += dDepthErrorMeters/dDepthMeters;
            ++uPointsSVDDLS;

            char chBufferDepth[20];
            std::snprintf( chBufferDepth, 20, "|%.2f", ( *vecPointTriangulatedSTEREOSVDLS )(2) );
            strDepthInfo += chBufferDepth;

            cv::circle( matDisplayRIGHT, ptLandmarkRIGHT, 10, CColorCodeBGR( 255, 0, 255 ), 1 );
        }
        else
        {
            strDepthInfo += "|X";
        }

        /*if( 0 != vecPointTriangulatedSTEREOQRLS )
        {
            //ds compute error
            const cv::Point2d ptLandmarkRIGHT( g_pCameraSTEREO->m_pCameraRIGHT->getProjection(*vecPointTriangulatedSTEREOQRLS ) );
            const double dDisparityPixels( ptLandmarkLEFT.x-ptLandmarkRIGHT.x );
            const double dDepthMeters( ( *vecPointTriangulatedSTEREOQRLS )(2) );
            const double dDepthErrorMeters( ( *vecPointTriangulatedSTEREOQRLS )(2)*( *vecPointTriangulatedSTEREOQRLS )(2)/g_pCameraSTEREO->m_dBaselineMeters*dDisparityPixels/g_pCameraSTEREO->m_pCameraLEFT->m_dFx );
            dAverageErrorQRLS += dDepthErrorMeters/dDepthMeters;
            ++uPointsQRLS;

            char chBufferDepth[20];
            std::snprintf( chBufferDepth, 20, "|%.2f", ( *vecPointTriangulatedSTEREOQRLS )(2) );
            strDepthInfo += chBufferDepth;

            cv::circle( matDisplayRIGHT, ptLandmarkRIGHT, 12, CColorCodeBGR( 128, 0, 128 ), 1 );
        }
        else
        {
            strDepthInfo += "|X";
        }

        if( 0 != vecPointTriangulatedSTEREOSVDDLT )
        {
            //ds compute error
            const cv::Point2d ptLandmarkRIGHT( g_pCameraSTEREO->m_pCameraRIGHT->getProjection( *vecPointTriangulatedSTEREOSVDDLT ) );
            const double dDisparityPixels( ptLandmarkLEFT.x-ptLandmarkRIGHT.x );
            const double dDepthMeters( ( *vecPointTriangulatedSTEREOSVDDLT )(2) );
            const double dDepthErrorMeters( ( *vecPointTriangulatedSTEREOSVDDLT )(2)*( *vecPointTriangulatedSTEREOSVDDLT )(2)/( g_pCameraSTEREO->m_dBaselineMeters*g_pCameraSTEREO->m_pCameraLEFT->m_dFx )*dDisparityPixels );
            dAverageErrorSVDDLT += dDepthErrorMeters/dDepthMeters;
            ++uPointsSVDDLT;

            char chBufferDepth[20];
            std::snprintf( chBufferDepth, 20, "|%.2f", ( *vecPointTriangulatedSTEREOSVDDLT )(2) );
            strDepthInfo += chBufferDepth;

            cv::circle( matDisplayRIGHT, ptLandmarkRIGHT, 14, CColorCodeBGR( 255, 0, 0 ), 1 );
        }
        else
        {
            strDepthInfo += "|X";
        }

        if( 0 != vecPointTriangulatedSTEREOEDS )
        {
            //ds compute error
            const cv::Point2d ptLandmarkRIGHT( g_pCameraSTEREO->m_pCameraRIGHT->getProjection( *vecPointTriangulatedSTEREOEDS  ) );
            const double dDisparityPixels( ptLandmarkLEFT.x-ptLandmarkRIGHT.x );
            const double dDepthMeters( ( *vecPointTriangulatedSTEREOEDS )(2) );
            const double dDepthErrorMeters( dDepthMeters*dDepthMeters/( g_pCameraSTEREO->m_dBaselineMeters*g_pCameraSTEREO->m_pCameraLEFT->m_dFx )*dDisparityPixels );
            dAverageErrorEDS += dDepthErrorMeters/dDepthMeters;
            ++uPointsEDS;

            char chBufferDepth[20];
            std::snprintf( chBufferDepth, 20, "|%.2f", ( *vecPointTriangulatedSTEREOEDS )(2) );
            strDepthInfo += chBufferDepth;

            cv::circle( matDisplayRIGHT, ptLandmarkRIGHT, 16, CColorCodeBGR( 0, 255, 0 ), 1 );
        }
        else
        {
            strDepthInfo += "|X";
        }*/

        cv::putText( matDisplayLEFT, strDepthInfo, cv::Point2d( ptLandmarkLEFT.x+cKeyPoint.size, ptLandmarkLEFT.y+cKeyPoint.size ), cv::FONT_HERSHEY_PLAIN, 0.75, CColorCodeBGR( 0, 0, 255 ) );

        ++uActivePoints;

        //ds free memory
        if( 0 != vecPointTriangulatedSTEREOSVDLS ){ delete vecPointTriangulatedSTEREOSVDLS; }
        if( 0 != vecPointTriangulatedSTEREOQRLS ){ delete vecPointTriangulatedSTEREOQRLS; }
        if( 0 != vecPointTriangulatedSTEREOSVDDLT ){ delete vecPointTriangulatedSTEREOSVDDLT; }
        if( 0 != vecPointTriangulatedSTEREOEDS ){ delete vecPointTriangulatedSTEREOEDS; }
    }

    //ds compute averages
    if( 0 != uPointsSVDDLS ){ dAverageErrorSVDLS  = dAverageErrorSVDLS/uPointsSVDDLS; }
    if( 0 != uPointsQRLS ){ dAverageErrorQRLS     = dAverageErrorQRLS/uPointsQRLS; }
    if( 0 != uPointsSVDDLT ){ dAverageErrorSVDDLT = dAverageErrorSVDDLT/uPointsSVDDLT; }
    if( 0 != uPointsEDS ){ dAverageErrorEDS       = dAverageErrorEDS/uPointsEDS; }















    //ds build display mat
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, g_matDisplay );

    char chBuffer[100];
    std::snprintf( chBuffer, 100, "ACTIVE POINTS: %lu ERRORS[ SVDLS: %4.2f(%2lu) QRLS: %4.2f(%2lu) SVDDLT: %4.2f(%2lu) EDS: %4.2f(%2lu) ]", uActivePoints,
                   dAverageErrorSVDLS, uPointsSVDDLS, dAverageErrorQRLS, uPointsQRLS, dAverageErrorSVDDLT, uPointsSVDDLT, dAverageErrorEDS, uPointsEDS );
    g_matDisplay( cv::Rect( 0, 0, g_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( g_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

    //ds display
    cv::imshow( "stereo depth calibration", g_matDisplay );
    cv::waitKey( 1 );
}

//ds sensor topic callbacks
void callbackCameraRIGHT( const sensor_msgs::ImageConstPtr p_pImageRIGHT );
void callbackCameraLEFT( const sensor_msgs::ImageConstPtr p_pImageLEFT );
/*void callbackIMU0( const sensor_msgs::ImuConstPtr p_msgIMU )
{
    //ds not used
}*/

int main( int argc, char** argv )
{
    //assert( false );

    //ds pwd info
    std::fflush( stdout);
    std::printf( "(main) launched: %s\n", argv[0] );
    CLogger::openBox( );

    //ds OpenMP
    const uint8_t uOpenMPNumberOfThreads( 2 );
    omp_set_num_threads( uOpenMPNumberOfThreads );
    uint8_t uOpenMPThreadsActive( 0 );

    //ds measure active threads
    #pragma omp parallel shared( uOpenMPThreadsActive )
    {
        ++uOpenMPThreadsActive;
    }

    std::printf( "(main) OpenMP set threads: %i | cur threads: %u | max threads: %i\n", uOpenMPNumberOfThreads, uOpenMPThreadsActive, omp_get_max_threads( ) );

    //ds cutoff distance
    double dMatchingDistanceCutoffTriangulation( 300.0 );

    if( 2 == argc )
    {
        dMatchingDistanceCutoffTriangulation = std::atof( argv[1] );
    }

    //ds configuration parameters
    std::string strTopicCameraRIGHT( "/thin_visensor_node/camera_right/image_raw" );
    std::string strTopicCameraLEFT ( "/thin_visensor_node/camera_left/image_raw" );
    std::string strTopicIMU        ( "/thin_visensor/imu_adis16448" );
    uint32_t uMaximumQueueSizeCamera( 10 );
    //uint32_t uMaximumQueueSizeIMU   ( 100 );

    //ds setup node
    ros::init( argc, argv, "test_depth" );
    std::shared_ptr< ros::NodeHandle > pNode( new ros::NodeHandle( "~" ) );

    //ds escape here on failure
    if( !pNode->ok( ) )
    {
        std::printf( "\n(main) ERROR: unable to instantiate node\n" );
        std::printf( "(main) terminated: %s\n", argv[0]);
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) ROS node namespace     := '%s'\n", pNode->getNamespace( ).c_str( ) );
    std::printf( "(main) ROS topic camera LEFT  := '%s'\n", strTopicCameraRIGHT.c_str( ) );
    std::printf( "(main) ROS topic camera RIGHT := '%s'\n", strTopicCameraLEFT.c_str( ) );
    std::printf( "(main) ROS topic IMU          := '%s'\n", strTopicIMU.c_str( ) );
    std::printf( "(main) matching distance cut  := '%f'\n", dMatchingDistanceCutoffTriangulation );
    std::fflush( stdout );
    CLogger::closeBox( );

    g_pTriangulator = std::make_shared< CTriangulator >( g_pCameraSTEREO, g_pExtractor, g_pMatcher, dMatchingDistanceCutoffTriangulation );

    //ds subscribe to topics
    ros::Subscriber cSubscriberCameraRIGHT = pNode->subscribe( strTopicCameraRIGHT, uMaximumQueueSizeCamera, callbackCameraRIGHT );
    ros::Subscriber cSubscriberCameraLEFT  = pNode->subscribe( strTopicCameraLEFT , uMaximumQueueSizeCamera, callbackCameraLEFT );
    //ros::Subscriber cSubscribeIMU0         = pNode->subscribe( strTopicIMU        , uMaximumQueueSizeIMU   , callbackIMU0 );

    //ds start callback pump
    ros::spin( );

    //ds exit
    std::printf( "\n(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

//ds sensor topic callbacks
void callbackCameraRIGHT( const sensor_msgs::ImageConstPtr p_pImageRIGHT )
{
    //ds check if we already have a left frame to compare against
    if( 0 != g_pImageLEFTLast )
    {
        //ds only if timestamps match
        if( g_pImageLEFTLast->header.stamp == p_pImageRIGHT->header.stamp )
        {
            //ds process both images
            computeDepth( cv_bridge::toCvCopy( g_pImageLEFTLast, g_pImageLEFTLast->encoding )->image,
                          cv_bridge::toCvCopy( p_pImageRIGHT, p_pImageRIGHT->encoding )->image );

            //ds reset buffers
            g_pImageRIGHTLast = 0;
            g_pImageLEFTLast  = 0;
        }

        //ds if the timestamp is newer drop the other image
        else if( g_pImageLEFTLast->header.stamp < p_pImageRIGHT->header.stamp )
        {
            g_pImageRIGHTLast = p_pImageRIGHT;
            g_pImageLEFTLast  = 0;
        }
    }
    else
    {
        g_pImageRIGHTLast = p_pImageRIGHT;
    }
}

void callbackCameraLEFT( const sensor_msgs::ImageConstPtr p_pImageLEFT )
{
    //ds check if we already have a right frame to compare against
    if( 0 != g_pImageRIGHTLast )
    {
        //ds only if timestamps match
        if( g_pImageRIGHTLast->header.stamp == p_pImageLEFT->header.stamp )
        {
            //ds process both images
            computeDepth( cv_bridge::toCvCopy( p_pImageLEFT, p_pImageLEFT->encoding )->image,
                          cv_bridge::toCvCopy( g_pImageRIGHTLast, g_pImageRIGHTLast->encoding )->image );

            //ds reset buffers
            g_pImageRIGHTLast = 0;
            g_pImageLEFTLast  = 0;
        }

        //ds if the timestamp is newer
        else if( g_pImageRIGHTLast->header.stamp < p_pImageLEFT->header.stamp )
        {
            g_pImageRIGHTLast = 0;
            g_pImageLEFTLast  = p_pImageLEFT;
        }
    }
    else
    {
        g_pImageLEFTLast = p_pImageLEFT;
    }
}
