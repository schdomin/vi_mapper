#ifndef TYPES_H
#define TYPES_H

#include <opencv/cv.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

enum EPlaybackMode
{
    ePlaybackInteractive,
    ePlaybackStepwise,
    ePlaybackBenchmark
};

//typedef Eigen::Vector3d CPoint2DNormalized;
typedef Eigen::Vector3d CPoint2DHomogenized;
typedef Eigen::Vector3d CPoint2DInCameraFrameHomogenized;
typedef Eigen::Vector4d CPoint3DHomogenized;
typedef Eigen::Vector2d CPoint2DInCameraFrame;
typedef Eigen::Vector3d CPoint3DInCameraFrame;
typedef Eigen::Vector3d CPoint3DInWorldFrame;
typedef Eigen::Vector4d CPoint3DInWorldFrameHomogenized;
typedef cv::Scalar      CColorCodeBGR;
typedef cv::Mat         CDescriptor;
typedef double          TFloatingPointNumber;
typedef uint64_t        UIDLandmark;

struct CMatchTracking
{
    const cv::Point2d ptPosition;
    const CDescriptor matDescriptor;

    CMatchTracking( const cv::Point2d& p_ptPosition, const CDescriptor& p_matDescriptor ): ptPosition( p_ptPosition ), matDescriptor( p_matDescriptor )
    {
        //ds nothing to do
    }
    ~CMatchTracking( )
    {
        //ds nothing to do
    }
};

struct CLandmarkMeasurement
{
    const UIDLandmark uID;
    const CPoint2DInCameraFrame vecPositionUV;
    const cv::Point2i ptPositionUV;

    CLandmarkMeasurement( const UIDLandmark& p_uID, const CPoint2DInCameraFrame& p_vecPositionUV, const cv::Point2i& p_ptPositionUV ): uID( p_uID ), vecPositionUV( p_vecPositionUV ), ptPositionUV( p_ptPositionUV )
    {
        //ds nothing to do
    }
    CLandmarkMeasurement( const UIDLandmark& p_uID, const CPoint2DInCameraFrameHomogenized& p_vecPositionUV, const cv::Point2i& p_ptPositionUV ): uID( p_uID ), vecPositionUV( p_vecPositionUV.head( 2 ) ), ptPositionUV( p_ptPositionUV )
    {
        //ds nothing to do
    }
};

struct CPositionRaw
{
    const CPoint3DInWorldFrame vecPointXYZ;
    const cv::Point2d ptPosition;
    const Eigen::Vector3d vecCameraPosition;
    const Eigen::Matrix3d matKRotation;
    const Eigen::Vector3d vecKTranslation;

    CPositionRaw( const CPoint3DInWorldFrame& p_vecPointXYZ,
               const cv::Point2d& p_ptPosition,
               const Eigen::Vector3d& p_vecCameraPosition,
               const Eigen::Matrix3d& p_matKRotation,
               const Eigen::Vector3d& p_vecKTranslation ): vecPointXYZ( p_vecPointXYZ ),
                                                           ptPosition( p_ptPosition ),
                                                           vecCameraPosition( p_vecCameraPosition ),
                                                           matKRotation( p_matKRotation ),
                                                           vecKTranslation( p_vecKTranslation )
    {

    }
};

#endif //TYPES_H
