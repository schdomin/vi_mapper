#ifndef TYPESCLOUD_H
#define TYPESCLOUD_H

#include "Types.h"

struct CDescriptorPoint3DWORLD
{
    const UIDLandmark uID;
    const CPoint3DWORLD vecPointXYZWORLD;
    const CDescriptor matDescriptor;

    CDescriptorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const CDescriptor& p_matDescriptor ): uID( p_uID ), vecPointXYZWORLD( p_vecPointXYZWORLD ), matDescriptor( p_matDescriptor )
    {
        //ds nothing to do
    }
};

struct CDescriptorVectorPoint3DWORLD
{
    const UIDLandmark uID;
    const CPoint3DWORLD vecPointXYZWORLD;
    const CPoint3DCAMERA vecPointXYZCAMERA;
    const cv::Point2d ptUVLEFT;
    const cv::Point2d ptUVRIGHT;
    const std::vector< CDescriptor > vecDescriptors;

    CDescriptorVectorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const CPoint3DCAMERA& p_vecPointXYZCAMERA,
                             const cv::Point2d& p_ptUVLEFT,
                             const cv::Point2d& p_ptUVRIGHT,
                             const std::vector< CDescriptor >& p_vecDescriptors ): uID( p_uID ),
                                                     vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                     vecPointXYZCAMERA( p_vecPointXYZCAMERA ),
                                                     ptUVLEFT( p_ptUVLEFT ),
                                                     ptUVRIGHT( p_ptUVRIGHT ),
                                                     vecDescriptors( p_vecDescriptors )
    {
        //ds nothing to do
    }
};

struct CDescriptorVectorPointCloud
{
    const UIDCloud uID;
    const Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

    CDescriptorVectorPointCloud( const UIDCloud& p_uID, const Eigen::Isometry3d& p_matPose, const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecPoints ): uID( p_uID ), matTransformationLEFTtoWORLD( p_matPose ), vecPoints( p_vecPoints )
    {
        //ds nothing to do
    }
};

struct CMatchCloud
{
    const CDescriptorVectorPoint3DWORLD cPointQuery;
    const CDescriptorVectorPoint3DWORLD cPointReference;
    const double dMatchingDistance;

    CMatchCloud( const CDescriptorVectorPoint3DWORLD& p_cPointQuery,
                 const CDescriptorVectorPoint3DWORLD& p_cPointReference,
                 const double& p_dMatchingDistance ): cPointQuery( p_cPointQuery ), cPointReference( p_cPointReference ), dMatchingDistance( p_dMatchingDistance )
    {
        //ds nothing to do
    }
};

#endif //TYPESCLOUD_H
