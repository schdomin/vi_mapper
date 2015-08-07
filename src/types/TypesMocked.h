#ifndef TYPESMOCKED_H
#define TYPESMOCKED_H

#include "Types.h"

struct CMockedLandmark
{
    const UIDLandmark uID;
    const CPoint3DWORLD vecPointXYZWORLD;
    const cv::Rect cRangeVisible;
    const double dNoiseMean;
    const double dNoiseStandardDeviation;

    CMockedLandmark( const UIDLandmark& p_uID,
                     const CPoint3DWORLD& p_vecPointXYZWORLD,
                     const double& p_dULCornerX,
                     const double& p_dULCornerY,
                     const double& p_dLRCornerX,
                     const double& p_dLRCornerY,
                     const double& p_dNoiseMean,
                     const double& p_dNoiseVariance ): uID( p_uID ),
                                                       vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                       cRangeVisible( cv::Point2d( p_dULCornerX, p_dULCornerY ), cv::Point2d( p_dLRCornerX, p_dLRCornerY ) ),
                                                       dNoiseMean( p_dNoiseMean ),
                                                       dNoiseStandardDeviation( p_dNoiseVariance )
    {
        //ds nothing to do
    }
};

struct CMockedDetection
{
    const UIDLandmark uID;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const double dNoiseULEFT;
    const double dNoiseURIGHT;
    const double dNoiseV;

    CMockedDetection( const UIDLandmark& p_uID,
                      const cv::Point2d& p_ptUVLEFT,
                      const cv::Point2d& p_ptUVRIGHT,
                      const double& p_dNoiseULEFT,
                      const double& p_dNoiseURIGHT,
                      const double& p_dNOiseV ): uID( p_uID ),
                                                 ptUVLEFT( p_ptUVLEFT ),
                                                 ptUVRIGHT( p_ptUVRIGHT ),
                                                 dNoiseULEFT( p_dNoiseULEFT ),
                                                 dNoiseURIGHT( p_dNoiseURIGHT ),
                                                 dNoiseV( p_dNOiseV )
    {
        //ds nothing to do
    }
};

#endif //TYPESMOCKED_H
