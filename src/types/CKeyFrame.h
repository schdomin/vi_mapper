#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"
#include "TypesCloud.h"
#include "utility/CLogger.h"
#include "g2o/types/slam3d/types_slam3d.h"

class CKeyFrame
{

public:

    //ds keyframe loop closing
    struct CMatchICP
    {
        const CKeyFrame* pKeyFrameReference;
        const Eigen::Isometry3d matTransformationToClosure;
        const std::shared_ptr< const std::vector< CMatchCloud > > vecMatches;

        CMatchICP( const CKeyFrame* p_pKeyFrameReference,
                   const Eigen::Isometry3d& p_matTransformationToReference,
                   const std::shared_ptr< const std::vector< CMatchCloud > > p_vecMatches ): pKeyFrameReference( p_pKeyFrameReference ),
                                                                                             matTransformationToClosure( p_matTransformationToReference ),
                                                                                             vecMatches( p_vecMatches )
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
               const std::vector< const CMatchICP* > p_vecLoopClosures );

    //ds keyframe loading from file (used for offline cloud matching)
    CKeyFrame( const std::string& p_strFile );

    ~CKeyFrame( );

public:

    const UIDKeyFrame uID;
    const UIDFrame uFrameOfCreation;
    Eigen::Isometry3d matTransformationLEFTtoWORLD;
    const CLinearAccelerationIMU vecLinearAccelerationNormalized;
    const std::vector< const CMeasurementLandmark* > vecMeasurements;
    bool bIsOptimized = false;
    const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > vecCloud;
    const std::vector< const CMatchICP* > vecLoopClosures;

private:

    //ds cloud matching
    static constexpr double m_dCloudMatchingWeightEuclidian        = 10.0;  //10.0
    static constexpr double m_dCloudMatchingMatchingDistanceCutoff = 125.0; //75.0

public:

    void saveCloudToFile( ) const;
    std::shared_ptr< const std::vector< CMatchCloud > > getMatches( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudQuery ) const;

    //ds offline loading
    std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > getCloudFromFile( const std::string& p_strFile );

};

#endif //CKEYFRAME_H
