#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"
#include "TypesCloud.h"

class CKeyFrame
{

public:

    //ds keyframe loop closing
    struct CMatchICP
    {
        const CKeyFrame* pKeyFrameReference;
        const Eigen::Isometry3d matTransformationToClosure;

        CMatchICP( const CKeyFrame* p_pKeyFrameReference,
                   const Eigen::Isometry3d& p_matTransformationToReference ): pKeyFrameReference( p_pKeyFrameReference ),
                                                                              matTransformationToClosure( p_matTransformationToReference )
        {
            //ds nothing to do
        }
    };

public:

    CKeyFrame( const UIDKeyFrame& p_uID,
               const UIDFrame& p_uFrame,
               const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
               const CLinearAccelerationIMU& p_vecLinearAcceleration,
               const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
               const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloud,
               const CMatchICP* p_pLoopClosure ): uID( p_uID ),
                                                  uFrameOfCreation( p_uFrame ),
                                                  matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                  vecLinearAccelerationNormalized( p_vecLinearAcceleration ),
                                                  vecMeasurements( p_vecMeasurements ),
                                                  vecCloud( p_vecCloud ),
                                                  pLoopClosure( p_pLoopClosure )
    {
        //ds nothing to do
    }
    ~CKeyFrame( )
    {
        //ds free loop closure if set
        if( 0 != pLoopClosure )
        {
            delete pLoopClosure;
        }
    }

public:

    const UIDKeyFrame uID;
    const UIDFrame uFrameOfCreation;
    Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const CLinearAccelerationIMU vecLinearAccelerationNormalized;
    const std::vector< const CMeasurementLandmark* > vecMeasurements;
    bool bIsOptimized = false;
    const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > vecCloud;
    const CMatchICP* pLoopClosure;

};

#endif //CKEYFRAME_H
