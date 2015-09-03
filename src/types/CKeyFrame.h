#ifndef CKEYFRAME_H
#define CKEYFRAME_H

#include "CLandmark.h"
#include "TypesCloud.h"
#include "utility/CLogger.h"

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
        assert( !vecCloud->empty( ) );

        //ds save the cloud to a file
        saveCloudToFile( );
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

public:

    void saveCloudToFile( )
    {
        //ds construct filestring and open dump file
        char chBuffer[256];
        std::snprintf( chBuffer, 256, "clouds/keyframe_%06lu.cloud", uID );
        std::ofstream ofCloud( chBuffer, std::ofstream::out );

        //ds dump pose and number of points information
        for( uint8_t u = 0; u < 4; ++u )
        {
            for( uint8_t v = 0; v < 4; ++v )
            {
                CLogger::writeDatum( ofCloud, matTransformationLEFTtoWORLD(u,v) );
            }
        }
        CLogger::writeDatum( ofCloud, vecCloud->size( ) );

        for( const CDescriptorVectorPoint3DWORLD& pPoint: *vecCloud )
        {
            //ds dump position and descriptor number info
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZWORLD.x( ) );
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZWORLD.y( ) );
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZWORLD.z( ) );
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZCAMERA.x( ) );
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZCAMERA.y( ) );
            CLogger::writeDatum( ofCloud, pPoint.vecPointXYZCAMERA.z( ) );

            assert( pPoint.ptUVLEFT.y == pPoint.ptUVRIGHT.y );

            CLogger::writeDatum( ofCloud, pPoint.ptUVLEFT.x );
            CLogger::writeDatum( ofCloud, pPoint.ptUVLEFT.y );
            CLogger::writeDatum( ofCloud, pPoint.ptUVRIGHT.x );
            CLogger::writeDatum( ofCloud, pPoint.ptUVRIGHT.y );

            CLogger::writeDatum( ofCloud, pPoint.vecDescriptors.size( ) );

            //ds dump all descriptors found so far
            for( const CDescriptor& pDescriptorLEFT: pPoint.vecDescriptors )
            {
                //ds buffer descriptor data
                const uchar* pDescriptor = pDescriptorLEFT.data;

                //ds print the descriptor elements
                for( uint8_t u = 0; u < 64; ++u ){ CLogger::writeDatum( ofCloud, pDescriptor[u] ); }
            }
        }

        ofCloud.close( );
    }

};

#endif //CKEYFRAME_H
