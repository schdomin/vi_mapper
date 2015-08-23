#ifndef TYPES_H
#define TYPES_H

#include "Typedefs.h"

struct CMeasurementLandmark
{
    const UIDLandmark uID;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const CPoint3DCAMERA vecPointXYZLEFT;
    const CPoint3DWORLD  vecPointXYZWORLD;
    const CPoint3DWORLD  vecPointXYZWORLDOptimized;
    const Eigen::Vector3d vecCameraPosition;
    //const Eigen::Matrix3d matPRotationWORLDtoLEFT;
    //const Eigen::Vector3d vecPTranslationWORLDtoLEFT;
    const MatrixProjection matProjectionWORLDtoLEFT;

    CMeasurementLandmark( const UIDLandmark& p_uID,
                          const cv::Point2d& p_ptUVLEFT,
                          const cv::Point2d& p_ptUVRIGHT,
                          const CPoint3DCAMERA& p_vecPointXYZ,
                          const CPoint3DWORLD& p_vecPointXYZWORLD,
                          const CPoint3DWORLD& p_vecPointXYZWORLDOptimized,
                          const Eigen::Vector3d& p_vecCameraPosition,
                          //const Eigen::Matrix3d& p_matKRotation,
                          //const Eigen::Vector3d& p_vecKTranslation,
                          const MatrixProjection& p_matProjectionWORLDtoLEFT ): uID( p_uID ),
                                                                      ptUVLEFT( p_ptUVLEFT ),
                                                                      ptUVRIGHT( p_ptUVRIGHT ),
                                                                      vecPointXYZLEFT( p_vecPointXYZ ),
                                                                      vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                                      vecPointXYZWORLDOptimized( p_vecPointXYZWORLDOptimized ),
                                                                      vecCameraPosition( p_vecCameraPosition ),
                                                                      //matPRotationWORLDtoLEFT( p_matKRotation ),
                                                                      //vecPTranslationWORLDtoLEFT( p_vecKTranslation ),
                                                                      matProjectionWORLDtoLEFT( p_matProjectionWORLDtoLEFT )
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
};

struct CMatchTriangulation
{
    const CPoint3DCAMERA vecPointXYZCAMERA;
    const cv::Point2f ptUVCAMERA;
    const CDescriptor matDescriptorCAMERA;

    CMatchTriangulation( const CPoint3DCAMERA& p_vecPointXYZCAMERA,
                         const cv::Point2f& p_ptUVCAMERA,
                         const CDescriptor& p_matDescriptorCAMERA ): vecPointXYZCAMERA( p_vecPointXYZCAMERA ),
                                                                     ptUVCAMERA( p_ptUVCAMERA ),
                                                                     matDescriptorCAMERA( p_matDescriptorCAMERA )
    {
        //ds nothing to do
    }
};

#endif //TYPES_H
