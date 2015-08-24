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
    const std::vector< CDescriptor > vecDescriptors;

    CDescriptorVectorPoint3DWORLD( const UIDLandmark& p_uID,
                             const CPoint3DWORLD& p_vecPointXYZWORLD,
                             const std::vector< CDescriptor >& p_vecDescriptors ): uID( p_uID ), vecPointXYZWORLD( p_vecPointXYZWORLD ), vecDescriptors( p_vecDescriptors )
    {
        //ds nothing to do
    }
};

struct CDescriptorPointCloud
{
    const UIDCloud uID;
    const Eigen::Isometry3d matPose;
    const std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

    CDescriptorPointCloud( const UIDCloud& p_uID, const Eigen::Isometry3d& p_matPose, const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecPoints ): uID( p_uID ), matPose( p_matPose ), vecPoints( p_vecPoints )
    {
        //ds nothing to do
    }
};

struct CMatchCloud
{
    const UIDLandmark uIDQuery;
    const UIDLandmark uIDMatch;

    CMatchCloud( const UIDLandmark& p_uIDQuery, const UIDLandmark& p_uIDMatch ): uIDQuery( p_uIDQuery ), uIDMatch( p_uIDMatch )
    {
        //ds nothing to do
    }
};

#endif //TYPESCLOUD_H
