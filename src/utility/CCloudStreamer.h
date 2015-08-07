#ifndef CCLOUDSTREAMER_H
#define CCLOUDSTREAMER_H

#include <fstream>

#include "types/CLandmark.h"
#include "types/TypesCloud.h"
#include "exceptions/CExceptionInvalidFile.h"

class CCloudstreamer
{

public:

    static CDescriptorPointCloud* getCloud( const UIDKeyFrame& p_uIDKeyFrame, const Eigen::Isometry3d& p_matPose, const std::shared_ptr< const std::vector< CLandmark* > > p_vecVisibleLandmarks )
    {
        //ds points in the cloud
        std::vector< CDescriptorPoint3DWORLD > vecPoints;

        //ds for all these points
        for( const CLandmark* pLandmark: *p_vecVisibleLandmarks )
        {
            vecPoints.push_back( CDescriptorPoint3DWORLD( pLandmark->uID, pLandmark->vecPointXYZOptimized, pLandmark->vecDescriptorsLEFT ) );
        }

        return new CDescriptorPointCloud( p_uIDKeyFrame, p_matPose, vecPoints );
    }

    static void saveLandmarksToCloudFile( const UIDKeyFrame& p_uIDKeyFrame, const Eigen::Isometry3d& p_matPose, const std::shared_ptr< const std::vector< CLandmark* > > p_vecVisibleLandmarks )
    {
        //ds construct filestring and open dump file
        char chBuffer[256];
        std::snprintf( chBuffer, 256, "clouds/keyframe_%06lu.cloud", p_uIDKeyFrame );
        std::ofstream ofCloud( chBuffer, std::ofstream::out );

        //ds dump pose and number of points information
        for( uint8_t u = 0; u < 4; ++u )
        {
            for( uint8_t v = 0; v < 4; ++v )
            {
                CCloudstreamer::writeDatum( ofCloud, p_matPose(u,v) );
            }
        }
        CCloudstreamer::writeDatum( ofCloud, p_vecVisibleLandmarks->size( ) );

        char chBufferFPS[256];
        std::snprintf( chBufferFPS, 256, "clouds/keyframe_%06lu.cloud_fps", p_uIDKeyFrame );
        std::ofstream ofCloudFPS( chBufferFPS, std::ofstream::out );
        CCloudstreamer::writeDatum( ofCloudFPS, p_vecVisibleLandmarks->size( ) );

        for( const CLandmark* pLandmark: *p_vecVisibleLandmarks )
        {
            //ds dump position and descriptor number info
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.x( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.y( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecPointXYZOptimized.z( ) );
            CCloudstreamer::writeDatum( ofCloud, pLandmark->vecDescriptorsLEFT.size( ) );

            //ds fps
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.x( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.y( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, static_cast< float >( pLandmark->vecPointXYZOptimized.z( ) ) );
            CCloudstreamer::writeDatum( ofCloudFPS, -1.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 0.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 0.0f );
            CCloudstreamer::writeDatum( ofCloudFPS, 1.0f );

            //ds dump all descriptors found so far
            for( const CDescriptor& pDescriptorLEFT: pLandmark->vecDescriptorsLEFT )
            {
                //ds buffer descriptor data
                const uchar* pDescriptor = pDescriptorLEFT.data;

                //ds print the descriptor elements
                for( uint8_t u = 0; u < 64; ++u ){ CCloudstreamer::writeDatum( ofCloud, pDescriptor[u] ); }
            }
        }

        ofCloud.close( );
        ofCloudFPS.close( );
    }

    static CDescriptorPointCloud loadCloud( const std::string& p_strFile, const UIDCloud& p_uID )
    {
        //ds open the file
        std::ifstream ifMessages( p_strFile, std::ifstream::in );

        //ds check if opening failed
        if( !ifMessages.is_open( ) )
        {
            throw CExceptionInvalidFile( "<CCloudStreamer>(loadCloud) unable to open file: " + p_strFile );
        }

        //ds parse pose
        Eigen::Isometry3d matPose( Eigen::Matrix4d::Identity( ) );
        for( uint8_t u = 0; u < 4; ++u )
        {
            for( uint8_t v = 0; v < 4; ++v )
            {
                CCloudstreamer::readDatum( ifMessages, matPose(u,v) );
            }
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

        return CDescriptorPointCloud( p_uID, matPose, vecPoints );
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
