#include "CMockedMatcherEpipolar.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CMockedMatcherEpipolar::CMockedMatcherEpipolar( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                                                const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT,
                                                const std::shared_ptr< CMockedStereoCamera > p_pCameraSTEREO,
                                                const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark ):
                                                                                m_pCameraLEFT( p_pCameraLEFT ),
                                                                                m_pCameraRIGHT( p_pCameraRIGHT ),
                                                                                m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                              m_uAvailableMeasurementPointID( 0 ),
                                                                              m_uMaximumFailedSubsequentTrackingsPerLandmark( p_uMaximumFailedSubsequentTrackingsPerLandmark ),
                                                                              m_pFileOdometryError( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/error_odometry.txt", "w" ) ),
                                                                              m_pFileEpipolarDetection( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/epipolar_detection.txt", "w" ) )
{
    //ds dump file format
    std::fprintf( m_pFileOdometryError, "FRAME MEASUREMENT_POINT ITERATION TOTAL_POINTS INLIERS ERROR\n" );
    std::fprintf( m_pFileEpipolarDetection, "FRAME MEASUREMENT_POINT LANDMARKS_TOTAL LANDMARKS_ACTIVE LANDMARKS_VISIBLE\n" );

    CLogger::openBox( );
    std::printf( "<CMockedMatcherEpipolar>(CMockedMatcherEpipolar) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "<CMockedMatcherEpipolar>(CMockedMatcherEpipolar) instance allocated\n" );
    CLogger::closeBox( );
}

CMockedMatcherEpipolar::~CMockedMatcherEpipolar( )
{
    std::fclose( m_pFileOdometryError );
    std::fclose( m_pFileEpipolarDetection );
    std::printf( "<CMockedMatcherEpipolar>(~CMockedMatcherEpipolar) instance deallocated\n" );
}

void CMockedMatcherEpipolar::addMeasurementPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
{
    m_vecLandmarkCreationPointsActive.push_back( CLandmarkCreationPoint( m_uAvailableMeasurementPointID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );

    ++m_uAvailableMeasurementPointID;
}

const Eigen::Isometry3d CMockedMatcherEpipolar::getPoseOptimized( const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT )
{
    //ds optimized pose
    Eigen::Isometry3d matTransformationOptimizedWORLDtoLEFT( p_matTransformationEstimateWORLDtoLEFT );

    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePoints;

    //ds active measurements
    for( const CLandmarkCreationPoint cCreationPoint: m_vecLandmarkCreationPointsActive )
    {
        //ds loop over the points for the current scan
        for( const CLandmark* pLandmarkReference: *cCreationPoint.vecLandmarks )
        {
            //ds world position
            const CPoint3DInWorldFrame& vecPointXYZ( pLandmarkReference->vecPointXYZCalibrated );

            //ds store world position
            vecLandmarksWORLD.push_back( vecPointXYZ );

            //ds project point into current scenery
            vecImagePoints.push_back( m_pCameraLEFT->getUV( p_matTransformationEstimateWORLDtoLEFT*vecPointXYZ ) );
        }
    }

    //ds feed the solver with the 3D points (in camera frame)
    m_cSolverPose.model_points = vecLandmarksWORLD;

    //ds feed the solver with the 2D points
    m_cSolverPose.image_points = vecImagePoints;

    //ds initial guess of the transformation
    m_cSolverPose.T = p_matTransformationEstimateWORLDtoLEFT;

    //double dErrorSolverPosePrevious( 0.0 );
    //const double dDeltaConvergence( 1e-5 );

    m_cSolverPose.init( );

    return matTransformationOptimizedWORLDtoLEFT;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMockedMatcherEpipolar::getVisibleLandmarksEssential( const uint64_t p_uFrame,
                                                                                                                    cv::Mat& p_matDisplayLEFT,
                                                                                                                    cv::Mat& p_matDisplayRIGHT,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                    cv::Mat& p_matDisplayTrajectory )
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Vector3d vecTranslation( p_matTransformationLEFTtoWORLD.translation( ) );
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const Eigen::Matrix3d matKRotation( m_pCameraLEFT->m_matIntrinsic*matTransformationWORLDtoLEFT.linear( ) );
    const Eigen::Vector3d vecCameraPosition( matTransformationWORLDtoLEFT.translation( ) );
    const Eigen::Vector3d vecKTranslation( m_pCameraLEFT->m_matIntrinsic*vecCameraPosition );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds current position
    const cv::Point2d ptPositionXY( vecTranslation.x( ), vecTranslation.y( ) );

    //ds get currently visible landmarks
    const std::map< UIDLandmark, CMockedDetection > mapVisibleLandmarks( m_pCameraSTEREO->getDetectedLandmarks( ptPositionXY, matTransformationWORLDtoLEFT, p_matDisplayTrajectory ) );

    //ds new active measurement points after this matching
    std::vector< CLandmarkCreationPoint > vecMeasurementPointsActive;

    //ds active measurements
    for( const CLandmarkCreationPoint cMeasurementPoint: m_vecLandmarkCreationPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerMeasurementPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerMeasurementPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cMeasurementPoint.vecLandmarks )
        {
            //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DInWorldFrame >( matTransformationWORLDtoLEFT*pLandmarkReference->vecPointXYZCalibrated ) ) );

            //ds draw last position
            cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

            try
            {
                //ds look for the detection of this landmark (HACK: set mocked landmark id into keypoint size field to avoid another landmark class)
                const CMockedDetection cDetection( mapVisibleLandmarks.at( static_cast< UIDLandmark >( pLandmarkReference->dKeyPointSize ) ) );

                //ds triangulate point
                const CPoint3DInCameraFrame vecPointTriangulatedLEFT( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( cDetection.ptUVLEFT,
                                                                                                                                  cDetection.ptUVRIGHT,
                                                                                                                                  m_pCameraLEFT->m_matProjection,
                                                                                                                                  m_pCameraRIGHT->m_matProjection ) );

                //ds get projection
                cv::Point2d ptUVRIGHT( m_pCameraRIGHT->getProjection( vecPointTriangulatedLEFT ) );

                //ds enforce epipolar constraint TODO integrate epipolar error
                const double dEpipolarError( ptUVRIGHT.y-cDetection.ptUVLEFT.y );
                if( 0.1 < dEpipolarError )
                {
                    std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] epipolar error: %f\n", pLandmarkReference->uID, dEpipolarError );
                }
                ptUVRIGHT.y = cDetection.ptUVLEFT.y;

                //ds update landmark
                pLandmarkReference->matDescriptorLastLEFT          = CDescriptor( );
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addPosition( p_uFrame, cDetection.ptUVLEFT, ptUVRIGHT, vecPointTriangulatedLEFT, static_cast< CPoint3DInWorldFrame >( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT ), vecCameraPosition, matProjectionWORLDtoLEFT );

                vecVisibleLandmarksPerMeasurementPoint.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds new positions
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                char chBufferMiniInfo[20];
                std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmarkReference->uID, pLandmarkReference->uCalibrations, pLandmarkReference->dCurrentAverageSquaredError );
                cv::putText( p_matDisplayLEFT, chBufferMiniInfo, cv::Point2d( cDetection.ptUVLEFT.x+10, cDetection.ptUVLEFT.y+10 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplayRIGHT, ptUVRIGHT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                cv::circle( p_matDisplayLEFT, ptProjection, 10, CColorCodeBGR( 0, 0, 255 ), 1 );
            }
            catch( const std::out_of_range& p_eException )
            {
                //ds draw last position
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
                ++pLandmarkReference->uFailedSubsequentTrackings;
                //std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] caught exception: %s\n", pLandmarkReference->uID, p_eException.what( ) );
            }

            //ds check activity
            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings )
            {
                vecActiveLandmarksPerMeasurementPoint->push_back( pLandmarkReference );
            }
            else
            {
                std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] dropped\n", pLandmarkReference->uID );
            }
        }

        //ds log
        std::fprintf( m_pFileEpipolarDetection, "%04lu %03lu %02lu %02lu %02lu\n", p_uFrame, cMeasurementPoint.uID, cMeasurementPoint.vecLandmarks->size( ), vecActiveLandmarksPerMeasurementPoint->size( ), vecVisibleLandmarksPerMeasurementPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerMeasurementPoint->empty( ) )
        {
            //ds number of visible landmarks
            const uint32_t uNumberOfVisibleLandmarks( vecVisibleLandmarksPerMeasurementPoint.size( ) );

            //ds vector for pose solver
            gtools::Vector3dVector vecLandmarksWORLD( uNumberOfVisibleLandmarks );
            gtools::Vector2dVector vecImagePoints( uNumberOfVisibleLandmarks );

            //ds solve pose
            for( uint32_t u = 0; u < uNumberOfVisibleLandmarks; ++u )
            {
                //ds feed it position in world
                vecLandmarksWORLD[u] = vecVisibleLandmarksPerMeasurementPoint[u]->vecPointXYZWORLD;
                vecImagePoints[u]    = CWrapperOpenCV::fromCVVector( vecVisibleLandmarksPerMeasurementPoint[u]->ptUVLEFT );
            }

            //ds feed the solver with the 3D points (in camera frame)
            m_cSolverPose.model_points = vecLandmarksWORLD;

            //ds feed the solver with the 2D points
            m_cSolverPose.image_points = vecImagePoints;

            //ds initial guess of the transformation
            m_cSolverPose.T = matTransformationWORLDtoLEFT;

            //double dErrorSolverPosePrevious( 0.0 );
            //const double dDeltaConvergence( 1e-5 );

            m_cSolverPose.init();

            //ds run LS - SNIPPET
            const uint8_t num_iterations = 10;
            for( uint8_t uIteration = 0; uIteration < num_iterations; ++uIteration )
            {
                const double dErrorSolverPoseCurrent = m_cSolverPose.oneRound( uIteration == num_iterations-1);
                uint32_t uInliersCurrent = m_cSolverPose.num_inliers;
                uint32_t uGoodReprojections = m_cSolverPose.num_reprojected_points;

                //std::cout << "iteration "<< iteration << std::endl;
                //std::cout << " error: " << dErrorSolverPoseCurrent << std::endl;

                /*ds check convergence
                if( dDeltaConvergence > std::fabs( dErrorSolverPosePrevious-dErrorSolverPoseCurrent ) )
                {
                    break;
                }
                else
                {
                    dErrorSolverPosePrevious = dErrorSolverPoseCurrent;
                }*/

                //ds log the error evolution [frame measurement_id iteration error]
                std::fprintf( m_pFileOdometryError, "%04lu %03lu %01u %02u %02u %02u %6.2f\n",
                            p_uFrame,
                            cMeasurementPoint.uID,
                            uIteration,
                            uNumberOfVisibleLandmarks,
                            uInliersCurrent,
                            uGoodReprojections,
                            dErrorSolverPoseCurrent );
            }

            const Eigen::Isometry3d matTransformationLEFTtoWORLDCorrected( m_cSolverPose.T.inverse( ) );
            const Eigen::Vector3d vecTranslationCorrected( matTransformationLEFTtoWORLDCorrected.translation( ) );

            //ds draw position on trajectory mat
            cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+vecTranslationCorrected( 0 )*10, 360-vecTranslationCorrected( 1 )*10 ), 2, CColorCodeBGR( 0, 0, 255 ), -1 );

            //ds register the measurement point and its visible landmarks anew
            vecMeasurementPointsActive.push_back( CLandmarkCreationPoint( cMeasurementPoint.uID, cMeasurementPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerMeasurementPoint ) );
            //vecMeasurementPointsActive.push_back( CMeasurementPoint( matTransformationLEFTtoWORLDCorrected, vecActiveLandmarksPerMeasurementPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerMeasurementPoint.begin( ), vecVisibleLandmarksPerMeasurementPoint.end( ) );
        }
        else
        {
            std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point\n" );
        }
    }

    //ds info
    //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) visible landmarks: %lu (active measurement points: %lu)\n", vecVisibleLandmarks->size( ), vecMeasurementPointsActive.size( ) );

    //ds update active measurement points
    m_vecLandmarkCreationPointsActive.swap( vecMeasurementPointsActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}
