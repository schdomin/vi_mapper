#ifndef CMOCKEDSTEREOCAMERA_H
#define CMOCKEDSTEREOCAMERA_H

#include <fstream>

#include "CStereoCamera.h"

class CMockedStereoCamera: public CStereoCamera
{

public:

    CMockedStereoCamera( const std::string& p_strFilenameLandmarks,
                         const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                         const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT ): CStereoCamera( p_pCameraLEFT, p_pCameraRIGHT ),
                                                                                   m_uAvailableMockedLandmarkID( 0 )
    {
        //ds load mocked landmarks
        std::ifstream ifLandmarks( p_strFilenameLandmarks );

        if( ifLandmarks.good( ) && ifLandmarks.is_open( ) )
        {
            //ds fields to read
            double dPositionX     = 0.0;
            double dPositionY     = 0.0;
            double dPositionZ     = 0.0;
            double dULCornerX     = 0.0;
            double dULCornerY     = 0.0;
            double dLRCornerX     = 0.0;
            double dLRCornerY     = 0.0;
            double dNoiseMean     = 0.0;
            double dNoiseVariance = 0.0;

            //ds read the file
            while( ifLandmarks >> dPositionX >> dPositionY >> dPositionZ >> dULCornerX >> dULCornerY >> dLRCornerX >> dLRCornerY >> dNoiseMean >> dNoiseVariance )
            {
                //ds push back the current landmark
                m_vecLandmarksMocked.push_back( CMockedLandmark( m_uAvailableMockedLandmarkID,
                                                                 CPoint3DInWorldFrame( dPositionX, dPositionY, dPositionZ ),
                                                                 dULCornerX, dULCornerY, dLRCornerX, dLRCornerY,
                                                                 dNoiseMean,
                                                                 dNoiseVariance ) );

                //ds increment id
                ++m_uAvailableMockedLandmarkID;
            }

            //ds done
            std::printf( "<CMockedStereoCamera>(CMockedStereoCamera) loaded %lu landmarks from file: %s\n", m_vecLandmarksMocked.size( ), p_strFilenameLandmarks.c_str( ) );
        }
        else
        {
            std::printf( "<CMockedStereoCamera>(CMockedStereoCamera) unable to load landmarks from file: %s\n", p_strFilenameLandmarks.c_str( ) );
        }

        //ds close stream
        ifLandmarks.close( );

        std::printf( "<CMockedStereoCamera>(CMockedStereoCamera) instance allocated\n" );
    }

    //ds no manual dynamic allocation
    ~CMockedStereoCamera( )
    {
        //ds nothing to do
    }


//ds fields
private:

    UIDLandmark m_uAvailableMockedLandmarkID;
    std::vector< CMockedLandmark > m_vecLandmarksMocked;

    //ds noise
    std::default_random_engine m_cGenerator;

//ds accessors
public:

    const std::map< UIDLandmark, CMockedDetection > getDetectedLandmarks( const cv::Point2d& p_ptPositionXY, const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT, cv::Mat& p_matDisplayTrajectory )
    {
        //ds empty map
        std::map< UIDLandmark, CMockedDetection > mapDetections;

        //ds info counters
        //uint64_t uBehindCamera  = 0;
        //uint64_t uOutOfFOVLEFT  = 0;
        //uint64_t uOutOfFOVRIGHT = 0;

        //ds project all landmarks into the current scene
        for( const CMockedLandmark cLandmark: m_vecLandmarksMocked )
        {
            //ds check if not occluded
            if( cLandmark.cRangeVisible.contains( p_ptPositionXY ) )
            {
                //ds get point into left camera frame
                const CPoint3DInCameraFrame vecPointXYZLEFT( p_matTransformationWORLDtoLEFT*cLandmark.vecPointXYZWORLD );

                //ds check if visible
                if( 0 < vecPointXYZLEFT.z( ) )
                {
                    //ds compute projection in LEFT
                    cv::Point2d ptUVLEFT( m_pCameraLEFT->getProjection( vecPointXYZLEFT ) );

                    //ds allocate normal distribution and compute the respective noises
                    std::normal_distribution< double > cDistributionNormal( cLandmark.dNoiseMean, cLandmark.dNoiseStandardDeviation );

                    //ds noise in V is the same for both to satisfy epipolar constraint
                    const double dNoiseVPixel      = cDistributionNormal( m_cGenerator );
                    const double dNoiseUPixelLEFT  = cDistributionNormal( m_cGenerator );
                    const double dNoiseUPixelRIGHT = cDistributionNormal( m_cGenerator );

                    //std::cout << dNoiseVPixel << " " << dNoiseUPixelLEFT << " " << dNoiseUPixelRIGHT << std::endl;

                    //ds add noise to detected point
                    ptUVLEFT.x += dNoiseUPixelLEFT;
                    ptUVLEFT.y += dNoiseVPixel;

                    //ds if the point is in the camera frame
                    if( m_cVisibleRange.contains( ptUVLEFT ) )
                    {
                        //ds compute projection in RIGHT
                        cv::Point2d ptUVRIGHT( m_pCameraRIGHT->getProjection( vecPointXYZLEFT ) );

                        //ds add noise to detected point
                        ptUVRIGHT.x += dNoiseUPixelRIGHT;
                        ptUVRIGHT.y += dNoiseVPixel;

                        //ds if the point is in the camera frame
                        if( m_cVisibleRange.contains( ptUVRIGHT ) )
                        {
                            //ds epipolar constraint
                            assert( std::round( ptUVLEFT.y ) == std::round( ptUVRIGHT.y ) );

                            mapDetections.insert( std::pair< UIDLandmark, CMockedDetection >( cLandmark.uID, CMockedDetection( cLandmark.uID, ptUVLEFT, ptUVRIGHT ) ) );

                            //ds draw visible landmark on map
                            cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+cLandmark.vecPointXYZWORLD.x( )*10, 360-cLandmark.vecPointXYZWORLD.y( )*10 ), 10, CColorCodeBGR( 0, 175, 0 ), 1 );
                        }
                        else
                        {
                            //++uOutOfFOVRIGHT;
                            cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+cLandmark.vecPointXYZWORLD.x( )*10, 360-cLandmark.vecPointXYZWORLD.y( )*10 ), 10, CColorCodeBGR( 200, 0, 150 ), 1 );
                        }
                    }
                    else
                    {
                        //++uOutOfFOVLEFT;
                        cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+cLandmark.vecPointXYZWORLD.x( )*10, 360-cLandmark.vecPointXYZWORLD.y( )*10 ), 10, CColorCodeBGR( 200, 0, 150 ), 1 );
                    }
                }
                else
                {
                    //++uBehindCamera;
                    cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+cLandmark.vecPointXYZWORLD.x( )*10, 360-cLandmark.vecPointXYZWORLD.y( )*10 ), 10, CColorCodeBGR( 255, 0, 0 ), 1 );
                }
            }
            else
            {
                //ds occluded landmark
                cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+cLandmark.vecPointXYZWORLD.x( )*10, 360-cLandmark.vecPointXYZWORLD.y( )*10 ), 10, CColorCodeBGR( 100, 100, 100 ), 1 );
            }
        }

        //std::printf( "<CMockedStereoCamera>(getDetectedLandmarks) detected landmarks: %lu (behind camera: %lu, out of FOV LEFT: %lu RIGHT: %lu)\n", mapDetections.size( ), uBehindCamera, uOutOfFOVLEFT, uOutOfFOVRIGHT );

        return mapDetections;
    }

};

#endif //CMOCKEDSTEREOCAMERA_H
