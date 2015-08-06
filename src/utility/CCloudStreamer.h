#ifndef CCLOUDSTREAMER_H
#define CCLOUDSTREAMER_H

#include <fstream>

#include "types/CLandmark.h"
#include "exceptions/CExceptionInvalidFile.h"

class CCloudstreamer
{

public:

    static void saveLandmarksToCloud( const UIDKeyFrame& p_uIDKeyFrame, const std::vector< CLandmark* >& p_vecLandmarks )
    {
        //ds construct filestring and open dump file
        char chBuffer[256];
        std::snprintf( chBuffer, 256, "clouds/keyframe_%06lu.cloud", p_uIDKeyFrame );
        std::ofstream ofCloud( chBuffer, std::ofstream::out );
        CCloudstreamer::writeDatum( ofCloud, p_vecLandmarks.size( ) );

        char chBufferFPS[256];
        std::snprintf( chBufferFPS, 256, "clouds/keyframe_%06lu.cloud_fps", p_uIDKeyFrame );
        std::ofstream ofCloudFPS( chBufferFPS, std::ofstream::out );
        CCloudstreamer::writeDatum( ofCloudFPS, p_vecLandmarks.size( ) );

        for( const CLandmark* pLandmark: p_vecLandmarks )
        {
            //ds dump position and descriptor number info
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.x( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.y( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.z( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->m_vecMeasurements.size( ) );

            //ds fps
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.x( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.y( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.z( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, -1.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 0.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 0.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 0.0f );

            //ds dump all descriptors found so far
            for( const CMeasurementLandmark* pMeasurement: pLandmark->m_vecMeasurements )
            {
                //ds buffer descriptor data
                const uchar* pDescriptor = pMeasurement->matDescriptorLEFT.data;

                //ds print the descriptor elements
                for( uint8_t u = 0; u < 64; ++u ){ CCloudstreamer::writeDatum( ofCloud, pDescriptor[u] ); }
            }
        }

        ofCloud.close( );
        ofCloudFPS.close( );
    }

    static CPointCloud loadCloud( const std::string& p_strFile, const UIDCloud& p_uID )
    {
        //ds open the file
        std::ifstream ifMessages( p_strFile, std::ifstream::in );

        //ds check if opening failed
        if( !ifMessages.is_open( ) )
        {
            throw CExceptionInvalidFile( "<CCloudStreamer>(loadCloud) unable to open file: " + p_strFile );
        }

        //ds parse number of points
        std::vector< CLandmark* >::size_type uNumberOfPoints;
        CCloudstreamer::readDatum( ifMessages, uNumberOfPoints );

        //ds points in the cloud (preallocation ignored since const elements)
        std::vector< CDescriptorPoint3DWORLD > vecPoints;

        //ds for all these points
        for( std::vector< CLandmark* >::size_type u = 0; u < uNumberOfPoints; ++u )
        {
            //ds point field
            CPoint3DWORLD vecPointXYZWORLD;
            CCloudstreamer::readDatum( ifMessages, vecPointXYZWORLD.x( ) );
            CCloudstreamer::readDatum( ifMessages, vecPointXYZWORLD.y( ) );
            CCloudstreamer::readDatum( ifMessages, vecPointXYZWORLD.z( ) );

            //ds number of descriptors
            std::vector< CMeasurementLandmark* >::size_type uNumberOfDescriptors;
            CCloudstreamer::readDatum( ifMessages, uNumberOfDescriptors );

            //ds descriptor vector (preallocate)
            std::vector< CDescriptor > vecDescriptors( uNumberOfDescriptors );

            //ds parse all descriptors
            for( std::vector< CMeasurementLandmark* >::size_type v = 0; v < uNumberOfDescriptors; ++v )
            {
                //ds current descriptor
                CDescriptor matDescriptor( 1, 64, CV_8U );

                //ds buffer descriptor data
                uchar* pDescriptor = matDescriptor.data;

                //ds every descriptor contains 64 fields
                for( uint8_t w = 0; w < 64; ++w )
                {
                    CCloudstreamer::readDatum( ifMessages, pDescriptor[w] );
                }

                vecDescriptors[v] = matDescriptor;
            }

            //ds set vector
            vecPoints.push_back( CDescriptorPoint3DWORLD( u, vecPointXYZWORLD, vecDescriptors ) );
        }

        return CPointCloud( p_uID, vecPoints );
    }

private:

    template < class T > static void writeDatum( std::ostream& p_osStream, const T& p_tValue )
    {
        const char * pValue = reinterpret_cast< const char* >( &p_tValue );
        p_osStream.write( pValue, sizeof(T) );
    }

    template < class T > static void readDatum( std::istream& p_isStream, T& p_tValue )
    {
        char * pValue = reinterpret_cast< char* >( &p_tValue );
        p_isStream.read( pValue, sizeof(T) );
    }

};

#endif //CCLOUDSTREAMER_H
