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
typedef uint64_t        UIDMeasurementPoint;
typedef Eigen::Matrix< double, 3, 4 > MatrixProjection;
typedef Eigen::Matrix< double, 1, 3 > Vector3dT;
typedef Eigen::Vector3d CLinearAccelerationInIMUFrame;

struct CMeasurementLandmark
{
    const UIDLandmark uID;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const CPoint3DInCameraFrame vecPointXYZLEFT;
    const CPoint3DInWorldFrame  vecPointXYZWORLD;
    const Eigen::Vector3d vecCameraPosition;
    const Eigen::Matrix3d matPRotationWORLDtoLEFT;
    const Eigen::Vector3d vecPTranslationWORLDtoLEFT;
    const MatrixProjection matProjectionWORLDtoLEFT;

    CMeasurementLandmark( const UIDLandmark& p_uID,
                          const cv::Point2d& p_ptUVLEFT,
                          const cv::Point2d& p_ptUVRIGHT,
                          const CPoint3DInCameraFrame& p_vecPointXYZ,
                          const CPoint3DInWorldFrame& p_vecPointXYZWORLD,
                          const Eigen::Vector3d& p_vecCameraPosition,
                          const Eigen::Matrix3d& p_matKRotation,
                          const Eigen::Vector3d& p_vecKTranslation,
                          const MatrixProjection& p_matProjectionWORLDtoLEFT ): uID( p_uID ),
                                                                      ptUVLEFT( p_ptUVLEFT ),
                                                                      ptUVRIGHT( p_ptUVRIGHT ),
                                                                      vecPointXYZLEFT( p_vecPointXYZ ),
                                                                      vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                                      vecCameraPosition( p_vecCameraPosition ),
                                                                      matPRotationWORLDtoLEFT( p_matKRotation ),
                                                                      vecPTranslationWORLDtoLEFT( p_vecKTranslation ),
                                                                      matProjectionWORLDtoLEFT( p_matProjectionWORLDtoLEFT )
    {
        //ds nothing to do
    }

};

struct CMeasurementPose
{
    const Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const CLinearAccelerationInIMUFrame vecLinearAccelerationNormalized;
    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecLandmarks;

    CMeasurementPose( const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
                      const CLinearAccelerationInIMUFrame& p_vecLinearAcceleration,
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

struct CMockedLandmark
{
    const UIDLandmark uID;
    const CPoint3DInWorldFrame vecPointXYZWORLD;
    const cv::Rect cRangeVisible;
    const double dNoiseMean;
    const double dNoiseStandardDeviation;

    CMockedLandmark( const UIDLandmark& p_uID,
                     const CPoint3DInWorldFrame& p_vecPointXYZWORLD,
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
    ~CMockedLandmark( )
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
    ~CMockedDetection( )
    {
        //ds nothing to do
    }
};

#endif //TYPES_H
