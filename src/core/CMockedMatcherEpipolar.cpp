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
                                                                              m_cSolverPose( m_pCameraLEFT->m_matProjection ),
                                                                              m_cSolverPoseSTEREO( m_pCameraSTEREO )
{
    CLogger::openBox( );
    std::printf( "<CMockedMatcherEpipolar>(CMockedMatcherEpipolar) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "<CMockedMatcherEpipolar>(CMockedMatcherEpipolar) instance allocated\n" );
    CLogger::closeBox( );
}

CMockedMatcherEpipolar::~CMockedMatcherEpipolar( )
{
    //ds close logfiles
    CLogger::CLogDetectionEpipolar::close( );
    CLogger::CLogOptimizationOdometry::close( );

    std::printf( "<CMockedMatcherEpipolar>(~CMockedMatcherEpipolar) instance deallocated\n" );
}

void CMockedMatcherEpipolar::addMeasurementPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
{
    m_vecDetectionPointsActive.push_back( CDetectionPoint( m_uAvailableMeasurementPointID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );

    ++m_uAvailableMeasurementPointID;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMockedMatcherEpipolar::getVisibleLandmarks( const uint64_t p_uFrame,
                                                                                                                    cv::Mat& p_matDisplayLEFT,
                                                                                                                    cv::Mat& p_matDisplayRIGHT,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                    cv::Mat& p_matDisplayTrajectory,
                                                                                                                    const double& p_dMotionScaling )
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Vector3d vecTranslation( p_matTransformationLEFTtoWORLD.translation( ) );
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds current position
    const cv::Point2d ptPositionXY( vecTranslation.x( ), vecTranslation.y( ) );

    //ds get currently visible landmarks
    const std::map< UIDLandmark, CMockedDetection > mapVisibleLandmarks( m_pCameraSTEREO->getDetectedLandmarks( ptPositionXY, matTransformationWORLDtoLEFT, p_matDisplayTrajectory ) );

    //ds new active measurement points after this matching
    std::vector< CDetectionPoint > vecMeasurementPointsActive;

    //ds initial translation
    const CPoint3DWORLD vecTranslationEstimate( p_matTransformationLEFTtoWORLD.translation( ) );

    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePointsLEFT;
    gtools::Vector2dVector vecImagePointsRIGHT;

    //ds active measurements
    for( const CDetectionPoint cMeasurementPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerMeasurementPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerMeasurementPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cMeasurementPoint.vecLandmarks )
        {
            //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( matTransformationWORLDtoLEFT*pLandmarkReference->vecPointXYZOptimized ) ) );

            //ds draw last position
            cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

            try
            {
                //ds look for the detection of this landmark (HACK: set mocked landmark id into keypoint size field to avoid another landmark class)
                const CMockedDetection cDetection( mapVisibleLandmarks.at( static_cast< UIDLandmark >( pLandmarkReference->dKeyPointSize ) ) );

                //ds triangulate point
                const CPoint3DCAMERA vecPointTriangulatedLEFT( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( cDetection.ptUVLEFT,
                                                                                                                                  cDetection.ptUVRIGHT,
                                                                                                                                  m_pCameraLEFT->m_matProjection,
                                                                                                                                  m_pCameraRIGHT->m_matProjection ) );

                //ds update landmark
                pLandmarkReference->bIsCurrentlyVisible        = true;
                pLandmarkReference->matDescriptorLASTLEFT      = CDescriptor( );
                pLandmarkReference->matDescriptorLASTRIGHT     = CDescriptor( );
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addMeasurement( p_uFrame, cDetection.ptUVLEFT, cDetection.ptUVRIGHT, vecPointTriangulatedLEFT, static_cast< CPoint3DWORLD >( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT ), matTransformationWORLDtoLEFT.translation( ), Eigen::Vector3d( 0.0, 0.0, 0.0 ), matProjectionWORLDtoLEFT, CDescriptor( ) );

                //ds register measurement
                vecVisibleLandmarksPerMeasurementPoint.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds store elements for optimization
                vecLandmarksWORLD.push_back( pLandmarkReference->vecPointXYZOptimized );
                vecImagePointsLEFT.push_back( CWrapperOpenCV::fromCVVector( cDetection.ptUVLEFT ) );
                vecImagePointsRIGHT.push_back( CWrapperOpenCV::fromCVVector( cDetection.ptUVRIGHT ) );

                //ds new positions
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                char chBufferMiniInfo[20];
                std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmarkReference->uID, pLandmarkReference->uOptimizationsSuccessful, pLandmarkReference->dCurrentAverageSquaredError );
                cv::putText( p_matDisplayLEFT, chBufferMiniInfo, cv::Point2d( cDetection.ptUVLEFT.x+10, cDetection.ptUVLEFT.y+10 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplayRIGHT, cDetection.ptUVRIGHT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
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
        CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cMeasurementPoint.uID, cMeasurementPoint.vecLandmarks->size( ), vecActiveLandmarksPerMeasurementPoint->size( ), vecVisibleLandmarksPerMeasurementPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerMeasurementPoint->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecMeasurementPointsActive.push_back( CDetectionPoint( cMeasurementPoint.uID, cMeasurementPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerMeasurementPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerMeasurementPoint.begin( ), vecVisibleLandmarksPerMeasurementPoint.end( ) );
        }
        else
        {
            std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point\n" );
        }
    }

    //ds check if we have a sufficient number of points to optimize
    if( m_uMinimumPointsForPoseOptimization < vecLandmarksWORLD.size( ) )
    {
        //ds feed the solver with the 3D points (in camera frame)
        m_cSolverPoseSTEREO.model_points = vecLandmarksWORLD;

        //ds feed the solver with the 2D points
        m_cSolverPoseSTEREO.vecProjectionsUVLEFT  = vecImagePointsLEFT;
        m_cSolverPoseSTEREO.vecProjectionsUVRIGHT = vecImagePointsRIGHT;

        //ds initial guess of the transformation
        m_cSolverPoseSTEREO.T = matTransformationWORLDtoLEFT;

        //ds initialize solver
        m_cSolverPoseSTEREO.init( );

        //ds convergence
        double dErrorPrevious( 0.0 );

        //ds run KLS
        const uint8_t uMaxIterations = 10;
        for( uint8_t uIteration = 0; uIteration < uMaxIterations; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseSTEREO.oneRound( );
            const uint32_t uInliersCurrent       = m_cSolverPoseSTEREO.uNumberOfInliers;

            CLogger::CLogOptimizationOdometry::addEntryIteration( p_uFrame, uIteration, vecLandmarksWORLD.size( ), uInliersCurrent, m_cSolverPoseSTEREO.uNumberOfReprojections, dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( m_dConvergenceDeltaPoseOptimization > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumInliersForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrentIO = m_cSolverPoseSTEREO.oneRoundInliersOnly( );

                    //ds log the error evolution
                    CLogger::CLogOptimizationOdometry::addEntryInliers( p_uFrame, vecLandmarksWORLD.size( ), m_cSolverPoseSTEREO.uNumberOfInliers, m_cSolverPoseSTEREO.uNumberOfReprojections, dErrorSolverPoseCurrentIO );
                }
                else
                {
                    std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        const Eigen::Isometry3d matTransformationLEFTtoWORLDCorrected( m_cSolverPoseSTEREO.T.inverse( ) );

        //ds qualitiy information
        const double dDeltaOptimization      = ( matTransformationLEFTtoWORLDCorrected.translation( )-vecTranslationEstimate ).squaredNorm( );
        const double dOptimizationCovariance = dDeltaOptimization/p_dMotionScaling;

        //ds log resulting trajectory and delta to initial
        CLogger::CLogOptimizationOdometry::addEntryResult( matTransformationLEFTtoWORLDCorrected.translation( ), dDeltaOptimization, dOptimizationCovariance );

        const cv::Point2d ptPositionXY( matTransformationLEFTtoWORLDCorrected.translation( ).x( ), matTransformationLEFTtoWORLDCorrected.translation( ).y( ) );
        cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+ptPositionXY.x*10, 360-ptPositionXY.y*10 ), 2, CColorCodeBGR( 0, 0, 255 ), -1 );
    }
    else
    {
        std::printf( "<CMockedMatcherEpipolar>(getVisibleLandmarksEssential) unable to optimize pose (%lu points)\n", vecLandmarksWORLD.size( ) );
    }

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecMeasurementPointsActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

//ds routine that resets the visibility of all active landmarks
void CMockedMatcherEpipolar::resetVisibilityActiveLandmarks( )
{
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            pLandmark->bIsCurrentlyVisible = false;
        }
    }
}

void CMockedMatcherEpipolar::setKeyFrameToVisibleLandmarks( )
{
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds added in keyframe
            if( pLandmark->bIsCurrentlyVisible ){ ++pLandmark->uNumberOfKeyFramePresences; }
        }
    }
}
