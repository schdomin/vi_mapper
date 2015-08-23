#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"

struct CKeyFrame
{
    const UIDKeyFrame uID;
    Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const CLinearAccelerationIMU vecLinearAccelerationNormalized;
    const std::vector< const CMeasurementLandmark* > vecMeasurements;
    bool bIsOptimized = false;
    const UIDFrame uFrame;
    const CKeyFrame* pLoopClosure;

    CKeyFrame( const UIDKeyFrame& p_uID,
               const Eigen::Isometry3d p_matTransformationLEFTtoWORLD,
               const CLinearAccelerationIMU& p_vecLinearAcceleration,
               const std::vector< const CMeasurementLandmark* >& p_vecMeasurements,
               const UIDFrame& p_uFrame,
               const CKeyFrame* p_pLoopClosure ): uID( p_uID ),
                                                  matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                  vecLinearAccelerationNormalized( p_vecLinearAcceleration ),
                                                  vecMeasurements( p_vecMeasurements ),
                                                  uFrame( p_uFrame ),
                                                  pLoopClosure( p_pLoopClosure )
    {
        //ds nothing to do
    }

    //ds copy constructor (however with unique id)
    CKeyFrame( const UIDKeyFrame& p_uID, const CKeyFrame* p_pKeyFrame, const CKeyFrame* p_pLoopClosure ): uID( p_uID ),
                                                                                                          matTransformationLEFTtoWORLD( p_pKeyFrame->matTransformationLEFTtoWORLD ),
                                                                                                          vecLinearAccelerationNormalized( p_pKeyFrame->vecLinearAccelerationNormalized ),
                                                                                                          vecMeasurements( p_pKeyFrame->vecMeasurements ),
                                                                                                          uFrame( p_pKeyFrame->uFrame ),
                                                                                                          pLoopClosure( p_pLoopClosure )
    {
        //ds nothing to do
    }
};

#endif //CKEYFRAME_H
