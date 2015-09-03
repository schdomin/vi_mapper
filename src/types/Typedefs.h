#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <opencv/cv.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <cassert>

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
typedef Eigen::Vector3d CPoint3DCAMERA;
typedef Eigen::Vector3d CPoint3DWORLD;
typedef Eigen::Vector4d CPoint3DInWorldFrameHomogenized;
typedef cv::Scalar      CColorCodeBGR;
typedef cv::Mat         CDescriptor;
typedef double          TFloatingPointNumber;
typedef uint64_t        UIDLandmark;
typedef uint64_t        UIDDescriptor;
typedef uint64_t        UIDDetectionPoint;
typedef int64_t         UIDKeyFrame;
typedef uint64_t        UIDCloud;
typedef uint64_t        UIDDescriptorPoint3D;
typedef int64_t         UIDFrame;
typedef Eigen::Matrix< double, 3, 4 > MatrixProjection;
typedef Eigen::Matrix< double, 1, 3 > Vector3dT;
typedef Eigen::Vector3d CLinearAccelerationIMU;
typedef Eigen::Vector3d CLinearAccelerationLEFT;
typedef Eigen::Vector3d CAngularVelocityIMU;
typedef Eigen::Vector3d CAngularVelocityLEFT;
typedef Eigen::Vector3d CAngularVelocityWORLD;
typedef Eigen::Vector3d CLinearAccelerationWORLD;

typedef Eigen::Matrix< double, 4, 4 > EigenMatrix4d;
typedef Eigen::Matrix< double, 6, 1 > EigenVector6d;
typedef Eigen::Matrix< double, 6, 6 > EigenMatrix6d;

#endif //TYPEDEFS_H
