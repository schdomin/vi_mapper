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
    const std::vector< CDescriptor > vecDescriptors;

    CDescriptorVectorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const CPoint3DCAMERA& p_vecPointXYZCAMERA,
                             const std::vector< CDescriptor >& p_vecDescriptors ): uID( p_uID ),
                                                     vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                     vecPointXYZCAMERA( p_vecPointXYZCAMERA ),
                                                     vecDescriptors( p_vecDescriptors )
    {
        //ds nothing to do
    }
};

struct CDescriptorPointCloud
{
    const UIDCloud uID;
    const Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

    CDescriptorPointCloud( const UIDCloud& p_uID, const Eigen::Isometry3d& p_matPose, const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecPoints ): uID( p_uID ), matTransformationLEFTtoWORLD( p_matPose ), vecPoints( p_vecPoints )
    {
        //ds nothing to do
    }
};

struct CMatchCloud
{
    const UIDLandmark uIDQuery;
    const UIDLandmark uIDMatch;
    const CPoint3DCAMERA vecPointXYZCAMERAQuery;
    const CPoint3DCAMERA vecPointXYZCAMERAMatch;

    CMatchCloud( const UIDLandmark& p_uIDQuery,
            const UIDLandmark& p_uIDMatch,
            const CPoint3DCAMERA& p_vecPointXYZCAMERAQuery,
            const CPoint3DCAMERA& p_vecPointXYZCAMERAMatch ): uIDQuery( p_uIDQuery ),
                    uIDMatch( p_uIDMatch ),
                    vecPointXYZCAMERAQuery( p_vecPointXYZCAMERAQuery ),
                    vecPointXYZCAMERAMatch( p_vecPointXYZCAMERAMatch )
    {
        //ds nothing to do
    }
};

#endif //TYPESCLOUD_H
