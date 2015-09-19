#ifndef CCLOUDSTREAMER_H
#define CCLOUDSTREAMER_H

#include <fstream>

#include "types/CLandmark.h"
#include "types/TypesCloud.h"
#include "types/C67DTree.h"
#include "types/CKeyFrame.h"
#include "exceptions/CExceptionInvalidFile.h"

class CCloudstreamer
{

public:

    static CDescriptorVectorPointCloud* getCloud( const UIDKeyFrame& p_uIDKeyFrame, const Eigen::Isometry3d& p_matPose, const std::shared_ptr< const std::vector< CLandmark* > > p_vecVisibleLandmarks )
    {
        //ds points in the cloud
        std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

        /*ds for all these points
        for( const CLandmark* pLandmark: *p_vecVisibleLandmarks )
        {
            //vecPoints.push_back( CDescriptorVectorPoint3DWORLD( pLandmark->uID, pLandmark->vecPointXYZOptimized, pLandmark->getLastPointXYZLEFT( ), pLandmark->vecDescriptorsLEFT ) );
        }*/

        return new CDescriptorVectorPointCloud( p_uIDKeyFrame, p_matPose, vecPoints );
    }

    static C67DTree* getTree( const UIDKeyFrame& p_uIDKeyFrame, const Eigen::Isometry3d& p_matPose, const std::shared_ptr< const std::vector< CLandmark* > > p_vecVisibleLandmarks )
    {
        //ds count points
        UIDDescriptorPoint3D uPoints = 0;

        //ds count total atomic points
        for( const CLandmark* pLandmark: *p_vecVisibleLandmarks )
        {
            uPoints += pLandmark->getNumberOfMeasurements( );
        }

        //ds allocate final structure (will be freed internally)
        float* arrData = new float[uPoints*67];

        //ds current point
        UIDDescriptorPoint3D uCurrentPoint = 0;

        //ds set points
        for( const CLandmark* pLandmark: *p_vecVisibleLandmarks )
        {
            //ds for all descriptors
            for( const CDescriptor& cDescriptor: pLandmark->vecDescriptorsLEFT )
            {
                const UIDDescriptorPoint3D uCurrentPosition = uCurrentPoint*67;

                arrData[uCurrentPosition] = pLandmark->vecPointXYZOptimized.x( );
                arrData[uCurrentPosition+1] = pLandmark->vecPointXYZOptimized.y( );
                arrData[uCurrentPosition+2] = pLandmark->vecPointXYZOptimized.z( );

                //ds add the descriptor
                for( uint8_t u = 0; u < 64; ++u )
                {
                    arrData[uCurrentPosition+3+u] = cDescriptor.at< uchar >( u );
                }

                //ds added a point
                ++uCurrentPoint;
            }
        }

        return new C67DTree( p_uIDKeyFrame, p_matPose, uPoints, arrData );
    }

    static CDescriptorVectorPointCloud loadCloud( const std::string& p_strFile )
    {
        //ds get cloud id (last 6 digits previous to .txt)
        const UIDKeyFrame uID = std::stoi( p_strFile.substr( p_strFile.length( )-12, 6 ) );

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
                CLogger::readDatum( ifMessages, matPose(u,v) );
            }
        }

        //ds parse number of points
        std::vector< CLandmark* >::size_type uNumberOfPoints;
        CLogger::readDatum( ifMessages, uNumberOfPoints );

        //ds points in the cloud (preallocation ignored since const elements)
        std::vector< CDescriptorVectorPoint3DWORLD > vecPoints;

        //ds for all these points
        for( std::vector< CLandmark* >::size_type u = 0; u < uNumberOfPoints; ++u )
        {
            //ds point field
            CPoint3DWORLD vecPointXYZWORLD;
            CPoint3DCAMERA vecPointXYZCAMERA;
            CLogger::readDatum( ifMessages, vecPointXYZWORLD.x( ) );
            CLogger::readDatum( ifMessages, vecPointXYZWORLD.y( ) );
            CLogger::readDatum( ifMessages, vecPointXYZWORLD.z( ) );
            CLogger::readDatum( ifMessages, vecPointXYZCAMERA.x( ) );
            CLogger::readDatum( ifMessages, vecPointXYZCAMERA.y( ) );
            CLogger::readDatum( ifMessages, vecPointXYZCAMERA.z( ) );

            assert( 0.0 < vecPointXYZCAMERA.z( ) );

            cv::Point2d ptUVLEFT;
            cv::Point2d ptUVRIGHT;
            CLogger::readDatum( ifMessages, ptUVLEFT.x );
            CLogger::readDatum( ifMessages, ptUVLEFT.y );
            CLogger::readDatum( ifMessages, ptUVRIGHT.x );
            CLogger::readDatum( ifMessages, ptUVRIGHT.y );

            assert( ptUVLEFT.y == ptUVRIGHT.y );

            //ds number of descriptors
            std::vector< CMeasurementLandmark* >::size_type uNumberOfDescriptors;
            CLogger::readDatum( ifMessages, uNumberOfDescriptors );

            //ds descriptor vector (preallocate)
            std::vector< CDescriptor > vecDescriptors( uNumberOfDescriptors );

            //ds parse all descriptors
            for( std::vector< CMeasurementLandmark* >::size_type v = 0; v < uNumberOfDescriptors; ++v )
            {
                //ds current descriptor
                CDescriptor matDescriptor( 1, 64, CV_8U );

                //ds every descriptor contains 64 fields
                for( uint8_t w = 0; w < 64; ++w )
                {
                    CLogger::readDatum( ifMessages, matDescriptor.data[w] );
                }

                vecDescriptors[v] = matDescriptor;
            }

            //ds set vector
            vecPoints.push_back( CDescriptorVectorPoint3DWORLD( u, vecPointXYZWORLD, vecPointXYZCAMERA, ptUVLEFT, ptUVRIGHT, vecDescriptors ) );
        }

        return CDescriptorVectorPointCloud( uID, matPose, vecPoints );
    }

};

#endif //CCLOUDSTREAMER_H
