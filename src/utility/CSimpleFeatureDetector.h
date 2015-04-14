#ifndef CSIMPLEFEATUREDETECTOR_H_
#define CSIMPLEFEATUREDETECTOR_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "configuration/CConfigurationStereoCamera.h"

class CSimpleFeatureDetector
{

//ds ctor/dtor
public:

    CSimpleFeatureDetector( const uint32_t& p_uImageRows, const uint32_t& p_uImageCols, const bool& p_bDisplayImages ): m_uImageRows( p_uImageRows ), m_uImageCols( p_uImageCols ), m_uFeaturesCap( 100 ), m_bDisplayImages( p_bDisplayImages )
    {
        //ds initialize reference frames with black images
        m_matReferenceFrameLeft  = cv::Mat( m_uImageRows, m_uImageCols, CV_8UC1, cv::Scalar( 0 ) );
        m_matReferenceFrameRight = cv::Mat( m_uImageRows, m_uImageCols, CV_8UC1, cv::Scalar( 0 ) );
    }
    ~CSimpleFeatureDetector( )
    {

    }

//ds members
private:

    //ds resolution
    const uint32_t m_uImageRows;
    const uint32_t m_uImageCols;

    //ds reference images
    cv::Mat m_matReferenceFrameLeft;
    cv::Mat m_matReferenceFrameRight;

    //ds feature related
    std::vector< cv::KeyPoint > m_vecKeypointsActive;
    const uint32_t m_uFeaturesCap;

    //ds control
    bool m_bDisplayImages;

//ds accessors
public:

    void receivevDataVI( txt_io::PinholeImageMessage& p_cImageLeft, txt_io::PinholeImageMessage& p_cImageRight, const txt_io::CIMUMessage& p_cIMU )
    {
        //ds get images into opencv format
        const cv::Mat matImageLeft( p_cImageLeft.image( ) );
        const cv::Mat matImageRight( p_cImageRight.image( ) );

        //ds detect features
        _detectFeaturesCorner( matImageLeft, matImageRight );
    }

//ds helpers
private:

    void _detectFeaturesCorner( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight )
    {
        //ds input mats
        cv::Mat matLeft;
        cv::Mat matRight;

        //ds normalize monochrome input
        cv::equalizeHist( p_matImageLeft, matLeft );
        cv::equalizeHist( p_matImageRight, matRight );

        //ds detected keypoints
        std::vector< cv::KeyPoint > vecKeyPointsLeft;
        std::vector< cv::KeyPoint > vecKeyPointsRight;

        //ds configuration
        const double dQualityLevel = 0.01;
        const double dMinimumDistance = 20.0;
        const uint32_t uBlockSize = 10;
        const bool bUseHarrisCorners = false;
        const double dK = 0.04;
        const double dMaximumVerticalDistortion = 25.0;

        //ds allocate a features detector
        cv::GoodFeaturesToTrackDetector cDetectorGFTT( m_uFeaturesCap, dQualityLevel, dMinimumDistance, uBlockSize, bUseHarrisCorners, dK );

        //ds detect features and create keypoints
        cDetectorGFTT.detect( matLeft, vecKeyPointsLeft );
        cDetectorGFTT.detect( matRight, vecKeyPointsRight );

        //ds get a descriptor extractor
        cv::OrbDescriptorExtractor cExtractorORB;

        cv::Mat matDescriptorsLeft;
        cv::Mat matDescriptorsRight;

        //ds compute descriptor images
        cExtractorORB.compute( matLeft, vecKeyPointsLeft, matDescriptorsLeft );
        cExtractorORB.compute( matRight, vecKeyPointsRight, matDescriptorsRight );

        //ds get a bruteforce matcher
        cv::BFMatcher cMatcher;
        std::vector< cv::DMatch > vecMatches;
        cMatcher.match( matDescriptorsLeft, matDescriptorsRight, vecMatches );

        std::vector< cv::DMatch > vecGoodMatches;

        //ds filter the matches for vertical skews
        for( cv::DMatch cMatch: vecMatches )
        {
            //ds check if the match is vertically plausible (thanks to stereo vision)
            if( dMaximumVerticalDistortion > std::fabs( vecKeyPointsLeft[cMatch.queryIdx].pt.y - vecKeyPointsRight[cMatch.trainIdx].pt.y ) )
            {
                //ds add the match
                vecGoodMatches.push_back( cMatch );
            }
        }

        //ds if display is desired
        if( m_bDisplayImages )
        {
            //ds get images into triple channel mats (display only)
            cv::Mat matDisplayLeft;
            cv::Mat matDisplayRight;

            cv::cvtColor( p_matImageLeft, matDisplayLeft, cv::COLOR_GRAY2BGR );
            cv::cvtColor( p_matImageRight, matDisplayRight, cv::COLOR_GRAY2BGR );

            //ds good matches
            std::vector< cv::DMatch > vecMatchesFiltered;

            //ds draw the matches
            for( cv::DMatch cMatch: vecGoodMatches )
            {
                cv::circle( matDisplayLeft, vecKeyPointsLeft[cMatch.queryIdx].pt, 5, cv::Scalar( 0, 255, 0 ), 1 );
                cv::circle( matDisplayRight, vecKeyPointsRight[cMatch.trainIdx].pt, 5, cv::Scalar( 0, 255, 0 ), 1 );
            }

            //ds display mat
            cv::Mat matDisplay = cv::Mat( m_uImageRows, 2*m_uImageCols, CV_8UC3 );
            cv::hconcat( matDisplayLeft, matDisplayRight, matDisplay );

            //ds draw matched feature connection
            for( cv::DMatch cMatch: vecGoodMatches )
            {
                cv::line( matDisplay, vecKeyPointsLeft[cMatch.queryIdx].pt, cv::Point2f( m_uImageCols + vecKeyPointsRight[cMatch.trainIdx].pt.x, vecKeyPointsRight[cMatch.trainIdx].pt.y ) , cv::Scalar( 255, 0, 0 ) );
            }

            //ds show the image
            cv::imshow( "stereo matching", matDisplay );
            cv::waitKey( 1 );
        }

        //ds update references
        m_matReferenceFrameLeft  = p_matImageLeft;
        m_matReferenceFrameRight = p_matImageRight;
    }
};

#endif //#define CSIMPLEFEATUREDETECTOR_H_
