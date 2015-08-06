#ifndef CCLOUDSTREAMER_H
#define CCLOUDSTREAMER_H

#include "types/CLandmark.h"

class CCloudstreamer
{

public:

    static void saveLandmarksToCloud( const UIDKeyFrame& p_uIDKeyFrame, const std::vector< CLandmark* >& p_vecLandmarks )
    {
        //ds construct filestring and open dump file
        char chBuffer[256];
        std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/clouds/keyframe_%06lu.txt", p_uIDKeyFrame );
        std::FILE* pFile = std::fopen( chBuffer, "w" );

        for( const CLandmark* pLandmark: p_vecLandmarks )
        {
            //ds dump position
            std::fprintf( pFile, "%6.2f %6.2f %6.2f", pLandmark->vecPointXYZOptimized.x( ), pLandmark->vecPointXYZOptimized.y( ), pLandmark->vecPointXYZOptimized.z( ) );

            //ds dump all descriptors found so far
            for( const CMeasurementLandmark* pMeasurement: pLandmark->m_vecMeasurements )
            {
                //ds buffer descriptor data
                const uchar* pDescriptor = pMeasurement->matDescriptorLEFT.data;

                //ds print the descriptor elements
                for( uint8_t u = 0; u < 64; ++u )
                {
                    std::fprintf( pFile, " %03i", pDescriptor[u] );
                }
            }

            std::fprintf( pFile, "\n" );
        }

        std::fclose( pFile );
    }

};

#endif //CCLOUDSTREAMER_H
