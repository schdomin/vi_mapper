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

struct CMeasurementLandmark
{
    const UIDLandmark uID;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const CPoint3DInCameraFrame vecPointXYZ;
    const Eigen::Vector3d vecCameraPosition;
    const Eigen::Matrix3d matKRotationLEFT;
    const Eigen::Vector3d vecKTranslationLEFT;

    CMeasurementLandmark( const UIDLandmark& p_uID,
                          const cv::Point2d& p_ptUVLEFT,
                          const cv::Point2d& p_ptUVRIGHT,
                          const CPoint3DInCameraFrame& p_vecPointXYZ,
                          const Eigen::Vector3d& p_vecCameraPosition,
                          const Eigen::Matrix3d& p_matKRotation,
                          const Eigen::Vector3d& p_vecKTranslation ): uID( p_uID ),
                                                                      ptUVLEFT( p_ptUVLEFT ),
                                                                      ptUVRIGHT( p_ptUVRIGHT ),
                                                                      vecPointXYZ( p_vecPointXYZ ),
                                                                      vecCameraPosition( p_vecCameraPosition ),
                                                                      matKRotationLEFT( p_matKRotation ),
                                                                      vecKTranslationLEFT( p_vecKTranslation )
    {
        //ds nothing to do
    }

};

struct CMeasurementPose
{
    const Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const Eigen::Vector3d vecLinearAccelerationNormalized;
    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecLandmarks;

    CMeasurementPose( const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
                      const Eigen::Vector3d& p_vecLinearAcceleration,
                      const std::shared_ptr< std::vector< const CMeasurementLandmark* > > p_vecLandmarks ): matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                                            vecLinearAccelerationNormalized( p_vecLinearAcceleration ),
                                                                                                            vecLandmarks( p_vecLandmarks )
    {
        //ds nothing to do
    }
};

struct CMatchTracking
{
    const cv::KeyPoint cKeyPoint;
    const CDescriptor matDescriptor;

    CMatchTracking( const cv::KeyPoint& p_cKeyPoint, const CDescriptor& p_matDescriptor ): cKeyPoint( p_cKeyPoint ), matDescriptor( p_matDescriptor )
    {
        //ds nothing to do
    }
    ~CMatchTracking( )
    {
        //ds nothing to do
    }
};

#endif //TYPES_H
