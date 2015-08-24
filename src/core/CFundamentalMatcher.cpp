#include "CFundamentalMatcher.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"
#include "optimization/CBridgeG2O.h"
#include "exceptions/CExceptionEpipolarLine.h"

CFundamentalMatcher::CFundamentalMatcher( const std::shared_ptr< CTriangulator > p_pTriangulator,
                                    const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle,
                                    const double& p_dMinimumDepthMeters,
                                    const double& p_dMaximumDepthMeters,
                                    const double& p_dMatchingDistanceCutoffPoseOptimization,
                                    const double& p_dMatchingDistanceCutoffEpipolar,
                                    const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark ): m_pTriangulator( p_pTriangulator ),
                                                                              m_pCameraLEFT( m_pTriangulator->m_pCameraSTEREO->m_pCameraLEFT ),
                                                                              m_pCameraRIGHT( m_pTriangulator->m_pCameraSTEREO->m_pCameraRIGHT ),
                                                                              m_pCameraSTEREO( m_pTriangulator->m_pCameraSTEREO ),
                                                                              m_pDetector( p_pDetectorSingle ),
                                                                              m_pExtractor( m_pTriangulator->m_pExtractor ),
                                                                              m_pMatcher( m_pTriangulator->m_pMatcher ),
                                                                              m_dMinimumDepthMeters( p_dMinimumDepthMeters ),
                                                                              m_dMaximumDepthMeters( p_dMaximumDepthMeters ),
                                                                              m_dMatchingDistanceCutoffPoseOptimization( p_dMatchingDistanceCutoffPoseOptimization ),
                                                                              m_dMatchingDistanceCutoffTrackingEpipolar( p_dMatchingDistanceCutoffEpipolar ),
                                                                              m_dMatchingDistanceCutoffOriginal( 2*m_dMatchingDistanceCutoffTrackingEpipolar ),
                                                                              m_uAvailableDetectionPointID( 0 ),
                                                                              m_uMaximumFailedSubsequentTrackingsPerLandmark( p_uMaximumFailedSubsequentTrackingsPerLandmark ),
                                                                              m_cSolverPoseSTEREO( m_pCameraSTEREO )
{
    m_vecDetectionPointsActive.clear( );
    m_vecVisibleLandmarks.clear( );

    CLogger::openBox( );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) minimum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) maximum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) matching distance cutoff (pose optimization): %f\n", m_dMatchingDistanceCutoffPoseOptimization );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) matching distance cutoff (fundamental line): %f\n", m_dMatchingDistanceCutoffTrackingEpipolar );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "<CFundamentalMatcher>(CFundamentalMatcher) instance allocated\n" );
    CLogger::closeBox( );
}

CFundamentalMatcher::~CFundamentalMatcher( )
{
    CLogger::CLogDetectionEpipolar::close( );
    CLogger::CLogOptimizationOdometry::close( );
    std::printf( "<CFundamentalMatcher>(~CFundamentalMatcher) instance deallocated\n" );
}

void CFundamentalMatcher::addDetectionPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
{
    m_vecDetectionPointsActive.push_back( CDetectionPoint( m_uAvailableDetectionPointID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );

    ++m_uAvailableDetectionPointID;
}

//ds routine that resets the visibility of all active landmarks
void CFundamentalMatcher::resetVisibilityActiveLandmarks( )
{
    //ds loop over the currently visible landmarks
    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        pLandmark->bIsCurrentlyVisible = false;
    }

    //ds clear reference vector
    m_vecVisibleLandmarks.clear( );
}

void CFundamentalMatcher::setKeyFrameToVisibleLandmarks( )
{
    //ds loop over the currently visible landmarks
    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        ++pLandmark->uNumberOfKeyFramePresences;
    }
}

const std::shared_ptr< const std::vector< CLandmark* > > CFundamentalMatcher::getVisibleOptimizedLandmarks( ) const
{
    //ds return vector
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    for( CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        if( CBridgeG2O::isOptimized( pLandmark ) )
        {
            vecVisibleLandmarks->push_back( pLandmark );
        }
    }

    return vecVisibleLandmarks;
}

const Eigen::Isometry3d CFundamentalMatcher::getPoseOptimizedSTEREO( const UIDFrame p_uFrame,
                                                                  cv::Mat& p_matDisplayLEFT,
                                                                  cv::Mat& p_matDisplayRIGHT,
                                                                  const cv::Mat& p_matImageLEFT,
                                                                  const cv::Mat& p_matImageRIGHT,
                                                                  const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                                  const Eigen::Vector3d& p_vecRotationTotal,
                                                                  const Eigen::Vector3d& p_vecTranslationTotal,
                                                                  const double& p_dMotionScaling )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePointsLEFT;
    gtools::Vector2dVector vecImagePointsRIGHT;

    //ds found landmarks in this frame
    std::vector< CMatchPoseOptimizationSTEREO > vecLandmarkMatches;

    //ds initial translation
    const CPoint3DWORLD vecTranslationEstimate( p_matTransformationEstimateWORLDtoLEFT.inverse( ).translation( ) );

    //ds active measurements
    for( const CDetectionPoint cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds project into camera
            const CPoint3DWORLD vecPointXYZ( pLandmark->vecPointXYZOptimized );
            const CPoint3DCAMERA vecPointXYZCAMERA( p_matTransformationEstimateWORLDtoLEFT*vecPointXYZ );

            //ds compute current reprojection point
            const cv::Point2d ptUVEstimateLEFT( m_pCameraLEFT->getProjection( vecPointXYZCAMERA ) );
            const cv::Point2d ptUVEstimateRIGHT( m_pCameraRIGHT->getProjection( vecPointXYZCAMERA ) );

            //ds check if both are in visible range
            if( m_pCameraSTEREO->m_cVisibleRange.contains( ptUVEstimateLEFT ) && m_pCameraSTEREO->m_cVisibleRange.contains( ptUVEstimateRIGHT ) )
            {
                //ds first try to find the landmarks on LEFT and check epipolar RIGHT
                try
                {
                    //ds compute search ranges
                    const double dScaleSearchULEFT      = m_pCameraLEFT->getPrincipalWeightU( ptUVEstimateLEFT ) + p_dMotionScaling;
                    const double dScaleSearchVLEFT      = m_pCameraLEFT->getPrincipalWeightV( ptUVEstimateLEFT ) + p_dMotionScaling;
                    const double dSearchHalfWidthLEFT   = dScaleSearchULEFT*m_uSearchBlockSizePoseOptimization;
                    const double dSearchHalfHeightLEFT  = dScaleSearchVLEFT*m_uSearchBlockSizePoseOptimization;

                    //ds corners
                    cv::Point2d ptUpperLeftLEFT( std::max( ptUVEstimateLEFT.x-dSearchHalfWidthLEFT, 0.0 ), std::max( ptUVEstimateLEFT.y-dSearchHalfHeightLEFT, 0.0 ) );
                    cv::Point2d ptLowerRightLEFT( std::min( ptUVEstimateLEFT.x+dSearchHalfWidthLEFT, m_pCameraLEFT->m_dWidthPixel ), std::min( ptUVEstimateLEFT.y+dSearchHalfHeightLEFT, m_pCameraLEFT->m_dHeightPixel ) );

                    //ds search rectangle
                    const cv::Rect cSearchROILEFT( ptUpperLeftLEFT, ptLowerRightLEFT );
                    cv::rectangle( p_matDisplayLEFT, cSearchROILEFT, CColorCodeBGR( 255, 0, 0 ) );

                    //ds run detection on current frame
                    std::vector< cv::KeyPoint > vecKeyPointsLEFT;
                    m_pDetector->detect( p_matImageLEFT( cSearchROILEFT ), vecKeyPointsLEFT );

                    //ds if we found some features in the LEFT frame
                    if( 0 < vecKeyPointsLEFT.size( ) )
                    {
                        //ds adjust keypoint offsets
                        std::for_each( vecKeyPointsLEFT.begin( ), vecKeyPointsLEFT.end( ), [ &ptUpperLeftLEFT ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt.x += ptUpperLeftLEFT.x; cKeyPoint.pt.y += ptUpperLeftLEFT.y; } );

                        //ds compute descriptors
                        cv::Mat matDescriptorsLEFT;
                        m_pExtractor->compute( p_matImageLEFT, vecKeyPointsLEFT, matDescriptorsLEFT );

                        //ds check descriptor matches for this landmark
                        std::vector< cv::DMatch > vecMatchesLEFT;
                        m_pMatcher->match( pLandmark->matDescriptorLASTLEFT, matDescriptorsLEFT, vecMatchesLEFT );

                        //ds if we got a match and the matching distance is within the range
                        if( 0 < vecMatchesLEFT.size( ) )
                        {
                            if( m_dMatchingDistanceCutoffPoseOptimization > vecMatchesLEFT[0].distance )
                            {
                                const cv::KeyPoint cKeyPointLEFT( vecKeyPointsLEFT[vecMatchesLEFT[0].trainIdx] );
                                const cv::Point2f ptBestMatchLEFT( cKeyPointLEFT.pt );
                                const CDescriptor matDescriptorLEFT( matDescriptorsLEFT.row(vecMatchesLEFT[0].trainIdx) );

                                //ds triangulate the point
                                const CMatchTriangulation cMatch( m_pTriangulator->getPointTriangulatedCompactInRIGHT( p_matImageRIGHT, cKeyPointLEFT, matDescriptorLEFT ) );
                                const CDescriptor matDescriptorRIGHT( cMatch.matDescriptorCAMERA );

                                //ds check depth
                                const double dDepthMeters = cMatch.vecPointXYZCAMERA.z( );
                                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                {
                                    throw CExceptionNoMatchFound( "invalid depth: " + std::to_string( dDepthMeters ) );
                                }

                                //ds check if the descriptor match is acceptable
                                if( m_dMatchingDistanceCutoffPoseOptimization > cv::norm( pLandmark->matDescriptorLASTRIGHT, matDescriptorRIGHT, cv::NORM_HAMMING ) )
                                {
                                    const cv::Point2f ptBestMatchRIGHT( cMatch.ptUVCAMERA );

                                    //ds store values for optimization
                                    vecLandmarksWORLD.push_back( vecPointXYZ );
                                    vecImagePointsLEFT.push_back( CPoint2DInCameraFrame( ptBestMatchLEFT.x, ptBestMatchLEFT.y ) );
                                    vecImagePointsRIGHT.push_back( CPoint2DInCameraFrame( ptBestMatchRIGHT.x, ptBestMatchRIGHT.y ) );

                                    //ds latter landmark update (cannot be done before pose is optimized)
                                    vecLandmarkMatches.push_back( CMatchPoseOptimizationSTEREO( pLandmark, cMatch.vecPointXYZCAMERA, ptBestMatchLEFT, ptBestMatchRIGHT, matDescriptorLEFT, matDescriptorRIGHT ) );

                                    cv::circle( p_matDisplayLEFT, ptBestMatchLEFT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                                }
                                else
                                {
                                    //ds try the RIGHT frame
                                    throw CExceptionNoMatchFound( "triangulation descriptor mismatch" );
                                }
                            }
                            else
                            {
                                //ds try the RIGHT frame
                                throw CExceptionNoMatchFound( "descriptor mismatch" );
                            }
                        }
                        else
                        {
                            //ds try the RIGHT frame
                            throw CExceptionNoMatchFound( "no matches found" );
                        }
                    }
                    else
                    {
                        //ds try the RIGHT frame
                        throw CExceptionNoMatchFound( "no features detected" );
                    }
                }
                catch( const CExceptionNoMatchFound& p_cException )
                {
                    //ds origin
                    //std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] LEFT failed: %s\n", pLandmark->uID, p_cException.what( ) );

                    //ds try to find the landmarks on RIGHT and check epipolar LEFT
                    try
                    {
                        //ds compute search ranges
                        const double dScaleSearchURIGHT      = m_pCameraRIGHT->getPrincipalWeightU( ptUVEstimateRIGHT ) + p_dMotionScaling;;
                        const double dScaleSearchVRIGHT      = m_pCameraRIGHT->getPrincipalWeightV( ptUVEstimateRIGHT ) + p_dMotionScaling;;
                        const double dSearchHalfWidthRIGHT   = dScaleSearchURIGHT*m_uSearchBlockSizePoseOptimization;
                        const double dSearchHalfHeightRIGHT  = dScaleSearchVRIGHT*m_uSearchBlockSizePoseOptimization;

                        //ds corners
                        cv::Point2d ptUpperLeftRIGHT( std::max( ptUVEstimateRIGHT.x-dSearchHalfWidthRIGHT, 0.0 ), std::max( ptUVEstimateRIGHT.y-dSearchHalfHeightRIGHT, 0.0 ) );
                        cv::Point2d ptLowerRightRIGHT( std::min( ptUVEstimateRIGHT.x+dSearchHalfWidthRIGHT, m_pCameraRIGHT->m_dWidthPixel ), std::min( ptUVEstimateRIGHT.y+dSearchHalfHeightRIGHT, m_pCameraRIGHT->m_dHeightPixel ) );

                        //ds search rectangle
                        const cv::Rect cSearchROIRIGHT( ptUpperLeftRIGHT, ptLowerRightRIGHT );
                        cv::rectangle( p_matDisplayRIGHT, cSearchROIRIGHT, CColorCodeBGR( 255, 0, 0 ) );

                        //ds run detection on current frame
                        std::vector< cv::KeyPoint > vecKeyPointsRIGHT;
                        m_pDetector->detect( p_matImageRIGHT( cSearchROIRIGHT ), vecKeyPointsRIGHT );

                        //ds if we found some features in the RIGHT frame
                        if( 0 < vecKeyPointsRIGHT.size( ) )
                        {
                            //ds adjust keypoint offsets
                            std::for_each( vecKeyPointsRIGHT.begin( ), vecKeyPointsRIGHT.end( ), [ &ptUpperLeftRIGHT ]( cv::KeyPoint& cKeyPoint ){ cKeyPoint.pt.x += ptUpperLeftRIGHT.x; cKeyPoint.pt.y += ptUpperLeftRIGHT.y; } );

                            //ds compute descriptors
                            cv::Mat matDescriptorsRIGHT;
                            m_pExtractor->compute( p_matImageRIGHT, vecKeyPointsRIGHT, matDescriptorsRIGHT );

                            //ds check descriptor matches for this landmark
                            std::vector< cv::DMatch > vecMatchesRIGHT;
                            m_pMatcher->match( pLandmark->matDescriptorLASTRIGHT, matDescriptorsRIGHT, vecMatchesRIGHT );

                            //ds if we got a match and the matching distance is within the range
                            if( 0 < vecMatchesRIGHT.size( ) )
                            {
                                if( m_dMatchingDistanceCutoffPoseOptimization > vecMatchesRIGHT[0].distance )
                                {
                                    const cv::KeyPoint cKeyPointRIGHT( vecKeyPointsRIGHT[vecMatchesRIGHT[0].trainIdx] );
                                    const cv::Point2f ptBestMatchRIGHT( cKeyPointRIGHT.pt );
                                    const CDescriptor matDescriptorRIGHT( matDescriptorsRIGHT.row(vecMatchesRIGHT[0].trainIdx) );

                                    //ds triangulate the point
                                    const CMatchTriangulation cMatchLEFT( m_pTriangulator->getPointTriangulatedCompactInLEFT( p_matImageLEFT, cKeyPointRIGHT, matDescriptorRIGHT ) );
                                    const CDescriptor matDescriptorLEFT( cMatchLEFT.matDescriptorCAMERA );

                                    //ds check depth
                                    const double dDepthMeters = cMatchLEFT.vecPointXYZCAMERA.z( );
                                    if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                                    {
                                        throw CExceptionNoMatchFound( "invalid depth: " + std::to_string( dDepthMeters ) );
                                    }

                                    //ds check if the descriptor match is acceptable
                                    if( m_dMatchingDistanceCutoffPoseOptimization > cv::norm( pLandmark->matDescriptorLASTLEFT, matDescriptorLEFT, cv::NORM_HAMMING ) )
                                    {
                                        const cv::Point2f ptBestMatchLEFT( cMatchLEFT.ptUVCAMERA );

                                        //ds store values for optimization
                                        vecLandmarksWORLD.push_back( vecPointXYZ );
                                        vecImagePointsLEFT.push_back( CPoint2DInCameraFrame( ptBestMatchLEFT.x, ptBestMatchLEFT.y ) );
                                        vecImagePointsRIGHT.push_back( CPoint2DInCameraFrame( ptBestMatchRIGHT.x, ptBestMatchRIGHT.y ) );

                                        //ds latter landmark update (cannot be done before pose is optimized)
                                        vecLandmarkMatches.push_back( CMatchPoseOptimizationSTEREO( pLandmark, cMatchLEFT.vecPointXYZCAMERA, ptBestMatchLEFT, ptBestMatchRIGHT, matDescriptorLEFT, matDescriptorRIGHT ) );

                                        cv::circle( p_matDisplayRIGHT, ptBestMatchRIGHT, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                                    }
                                    /*else
                                    {
                                        std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: triangulation descriptor mismatch\n", pLandmark->uID );
                                    }*/
                                }
                                /*else
                                {
                                    std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: descriptor mismatch\n", pLandmark->uID );
                                }*/
                            }
                            /*else
                            {
                                std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: no matches found\n", pLandmark->uID );
                            }*/
                        }
                        /*else
                        {
                            std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: no features detected\n", pLandmark->uID );
                        }*/
                    }
                    catch( const CExceptionNoMatchFound& p_cException )
                    {
                        //std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: %s\n", pLandmark->uID, p_cException.what( ) );
                    }
                }
            }
            /*else
            {
                std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) landmark [%06lu] out of visible range: LEFT(%6.2f %6.2f) RIGHT(%6.2f %6.2f)\n", pLandmark->uID, ptUVEstimateLEFT.x, ptUVEstimateLEFT.y, ptUVEstimateRIGHT.x, ptUVEstimateRIGHT.y );
            }*/
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
        m_cSolverPoseSTEREO.T = p_matTransformationEstimateWORLDtoLEFT;

        //ds initialize solver
        m_cSolverPoseSTEREO.init( );

        //ds convergence
        double dErrorPrevious( 0.0 );

        //ds run KLS
        for( uint8_t uIteration = 0; uIteration < m_uCapIterationsPoseOptimization; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseSTEREO.oneRound( );
            uint32_t uInliersCurrent             = m_cSolverPoseSTEREO.uNumberOfInliers;

            //ds log the error evolution
            CLogger::CLogOptimizationOdometry::addEntryIteration( p_uFrame, uIteration, vecLandmarksWORLD.size( ), uInliersCurrent, m_cSolverPoseSTEREO.uNumberOfReprojections, dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( m_dConvergenceDeltaPoseOptimization > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumInliersForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrentInliers = m_cSolverPoseSTEREO.oneRoundInliersOnly( );

                    //ds log the error evolution
                    CLogger::CLogOptimizationOdometry::addEntryInliers( p_uFrame, vecLandmarksWORLD.size( ), m_cSolverPoseSTEREO.uNumberOfInliers, m_cSolverPoseSTEREO.uNumberOfReprojections, dErrorSolverPoseCurrentInliers );
                }
                else
                {
                    //std::printf( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        //ds precompute
        const Eigen::Isometry3d matTransformationLEFTtoWORLDOptimized( m_cSolverPoseSTEREO.T.inverse( ) );

        //ds qualitiy information
        const CPoint3DWORLD vecTranslationOptimized( matTransformationLEFTtoWORLDOptimized.translation( )-vecTranslationEstimate );
        const double dDeltaOptimization = vecTranslationOptimized.squaredNorm( );
        const double dOptimizationRISK  = ( vecTranslationOptimized-p_vecTranslationTotal ).squaredNorm( );

        //ds log resulting trajectory and delta to initial
        CLogger::CLogOptimizationOdometry::addEntryResult( matTransformationLEFTtoWORLDOptimized.translation( ), dDeltaOptimization, dOptimizationRISK );

        //ds check if acceptable
        if( m_dMaximumRISK > dOptimizationRISK )
        {
            //ds update all visible landmarks
            const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*m_cSolverPoseSTEREO.T.matrix( ) );
            for( CMatchPoseOptimizationSTEREO pMatchSTEREO: vecLandmarkMatches )
            {
                _addMeasurementToLandmarkSTEREO( p_uFrame, pMatchSTEREO, matTransformationLEFTtoWORLDOptimized, p_vecRotationTotal, matProjectionWORLDtoLEFT );
            }

            //ds update info
            m_uNumberOfDetectionsPoseOptimization = vecLandmarkMatches.size( );

            //ds return optimized pose
            return m_cSolverPoseSTEREO.T;
        }
        else
        {
            throw CExceptionPoseOptimization( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) unable to optimize pose (RISK: " + std::to_string( dOptimizationRISK )+ ")" );
        }
    }
    else
    {
        throw CExceptionPoseOptimization( "<CFundamentalMatcher>(getPoseOptimizedSTEREO) unable to optimize pose (insufficient number of points: " + std::to_string( vecLandmarksWORLD.size( ) ) + ")" );
    }
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CFundamentalMatcher::getMeasurementsDummy( cv::Mat& p_matDisplayLEFT,
                                                                                                                 cv::Mat& p_matDisplayRIGHT,
                                                                                                                 const UIDFrame p_uFrame,
                                                                                                                 const cv::Mat& p_matImageLEFT,
                                                                                                                 const cv::Mat& p_matImageRIGHT,
                                                                                                                 const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD )
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecKeyFramesActive;

    //ds active measurements
    for( const CDetectionPoint cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerDetectionPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerDetectionPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized ) ) );

            cv::circle( p_matDisplayLEFT, ptProjection, 6, CColorCodeBGR( 0, 0, 255 ), 1 );

            vecVisibleLandmarksPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );
            vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
        }

        //ds log
        CLogger::CLogDetectionEpipolar::addEntry(  p_uFrame, cDetectionPoint.uID, cDetectionPoint.vecLandmarks->size( ), vecActiveLandmarksPerDetectionPoint->size( ), vecVisibleLandmarksPerDetectionPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerDetectionPoint->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecKeyFramesActive.push_back( CDetectionPoint( cDetectionPoint.uID, cDetectionPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerDetectionPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerDetectionPoint.begin( ), vecVisibleLandmarksPerDetectionPoint.end( ) );
        }
        else
        {
            std::printf( "<CFundamentalMatcher>(getVisibleLandmarksEssential) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecKeyFramesActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const std::shared_ptr< const std::vector< const CMeasurementLandmark* > > CFundamentalMatcher::getMeasurementsEpipolar( cv::Mat& p_matDisplayLEFT,
                                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                                      const UIDFrame p_uFrame,
                                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                                      const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                      const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                                      const double& p_dMotionScaling )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds measurements to return
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecMeasurements( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecDetectionPointsActive;

    //ds compute initial sampling line
    const double dHalfLineLength = p_dMotionScaling*10;

    //ds total detections counter
    UIDLandmark uDetectionsEpipolar = 0;

    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecMeasurementsPerDetectionPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerDetectionPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds check relative transform
        const Eigen::Isometry3d matTransformationToNow( p_matTransformationWORLDtoLEFT*cDetectionPoint.matTransformationLEFTtoWORLD );

        //ds compute essential matrix for this detection point
        const Eigen::Matrix3d matRotation( matTransformationToNow.linear( ) );
        const Eigen::Vector3d vecTranslation( matTransformationToNow.translation( ) );
        const Eigen::Matrix3d matEssential( matRotation*CMiniVisionToolbox::getSkew( vecTranslation ) );
        const Eigen::Matrix3d matFundamental( m_pCameraLEFT->m_matIntrinsicPInverseTransposed*matEssential*m_pCameraLEFT->m_matIntrinsicPInverse );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds check if we can skip this landmark due to failed optimization
            if( 0 < pLandmark->uOptimizationsFailed )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                pLandmark->bIsCurrentlyVisible = false;
                ++m_uNumberOfInvalidLandmarksTotal;
            }

            //ds check if we can skip this landmark due to invalid optimization
            else if( 0 < pLandmark->uOptimizationsSuccessful && !CBridgeG2O::isOptimized( pLandmark ) )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                pLandmark->bIsCurrentlyVisible = false;
                ++m_uNumberOfInvalidLandmarksTotal;
            }

            //ds process the landmark
            else
            {
                //ds check if already detected (in pose optimization)
                if( pLandmark->bIsCurrentlyVisible )
                {
                    //ds just register the measurement
                    vecMeasurementsPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );
                    vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                }
                else
                {
                    //ds if there was a translation (else the essential matrix is undefined)
                    if( 0.0 < vecTranslation.squaredNorm( ) )
                    {
                        //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
                        const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized ) ) );

                        try
                        {
                            //ds check if the projection is out of the fov
                            if( !m_pCameraLEFT->m_cFOV.contains( ptProjection ) )
                            {
                                throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) projection out of sight" );
                            }

                            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
                            const Eigen::Vector3d vecCoefficients( matFundamental*pLandmark->vecUVReferenceLEFT );

                            //ds line length for this projection based on principal weighting
                            const double dHalfLineLengthU = 15 + m_pCameraLEFT->getPrincipalWeightU( ptProjection )*dHalfLineLength;
                            const double dHalfLineLengthV = 15 + m_pCameraLEFT->getPrincipalWeightV( ptProjection )*dHalfLineLength;

                            assert( 0.0 < dHalfLineLengthU );
                            assert( 0.0 < dHalfLineLengthV );

                            //ds raw values
                            const double dUMinimumRAW     = std::max( ptProjection.x-dHalfLineLengthU, 0.0 );
                            const double dUMaximumRAW     = std::min( ptProjection.x+dHalfLineLengthU, m_pCameraLEFT->m_dWidthPixel );
                            const double dVForUMinimumRAW = _getCurveV( vecCoefficients, dUMinimumRAW );
                            const double dVForUMaximumRAW = _getCurveV( vecCoefficients, dUMaximumRAW );

                            assert( 0.0 <= dUMinimumRAW );
                            assert( m_pCameraLEFT->m_dWidthPixel >= dUMinimumRAW );
                            assert( 0.0 <= dUMaximumRAW );
                            assert( m_pCameraLEFT->m_dWidthPixel >= dUMaximumRAW );
                            assert( dUMinimumRAW     < dUMaximumRAW );
                            assert( dVForUMinimumRAW != dVForUMaximumRAW );

                            //ds check if line is out of scope
                            if( ( 0.0 > dVForUMinimumRAW && 0.0 > dVForUMaximumRAW ) || ( m_pCameraLEFT->m_dHeightPixel < dVForUMinimumRAW && m_pCameraLEFT->m_dHeightPixel < dVForUMaximumRAW ) )
                            {
                                //ds landmark out of sight (not visible in this frame, still active though)
                                throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) vertical out of sight" );
                            }

                            //ds final values set after if tree TODO reduce booleans for performance
                            double dUMinimum     = -1.0;
                            double dUMaximum     = -1.0;
                            double dVForUMinimum = -1.0;
                            double dVForUMaximum = -1.0;

                            //ds compute v border values
                            const double dVLimitMinimum( std::max( ptProjection.y-dHalfLineLengthV, 0.0 ) );
                            const double dVLimitMaximum( std::min( ptProjection.y+dHalfLineLengthV, m_pCameraLEFT->m_dHeightPixel ) );

                            assert( 0.0 <= dVLimitMinimum && m_pCameraLEFT->m_dHeightPixel >= dVLimitMinimum );
                            assert( 0.0 <= dVLimitMaximum && m_pCameraLEFT->m_dHeightPixel >= dVLimitMaximum );

                            //ds regular U
                            dUMinimum = dUMinimumRAW;
                            dUMaximum = dUMaximumRAW;

                            //ds check line configuration
                            if( dVForUMinimumRAW < dVForUMaximumRAW )
                            {
                                //ds check for invalid border values (bad reprojections)
                                if( dVLimitMinimum > dVForUMaximumRAW || dVLimitMaximum < dVForUMinimumRAW )
                                {
                                    throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) caught bad projection negative slope" );
                                }

                                //ds regular case
                                if( dVLimitMinimum > dVForUMinimumRAW )
                                {
                                    dVForUMinimum = dVLimitMinimum;
                                    dUMinimum     = _getCurveU( vecCoefficients, dVForUMinimum );
                                }
                                else
                                {
                                    dVForUMinimum = dVForUMinimumRAW;
                                }
                                if( dVLimitMaximum < dVForUMaximumRAW )
                                {
                                    dVForUMaximum = dVLimitMaximum;
                                    dUMaximum     = _getCurveU( vecCoefficients, dVForUMaximum );
                                }
                                else
                                {
                                    dVForUMaximum = dVForUMaximumRAW;
                                }
                                //std::printf( "sampling case 0: [%6.2f,%6.2f][%6.2f,%6.2f]\n", dUMinimum, dUMaximum, dVForUMinimum, dVForUMaximum );
                            }
                            else
                            {
                                //ds check for invalid border values (bad reprojections)
                                if( dVLimitMinimum > dVForUMinimumRAW || dVLimitMaximum < dVForUMaximumRAW )
                                {
                                    throw CExceptionEpipolarLine( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) caught bad projection positive slope" );
                                }

                                //ds swapped case
                                if( dVLimitMinimum > dVForUMaximumRAW )
                                {
                                    dVForUMinimum = dVLimitMinimum;
                                    dUMaximum     = _getCurveU( vecCoefficients, dVForUMinimum );
                                }
                                else
                                {
                                    dVForUMinimum = dVForUMaximumRAW;
                                }
                                if( dVLimitMaximum < dVForUMinimumRAW )
                                {
                                    dVForUMaximum = dVLimitMaximum;
                                    dUMinimum     = _getCurveU( vecCoefficients, dVForUMaximum );
                                }
                                else
                                {
                                    dVForUMaximum = dVForUMinimumRAW;
                                }
                                //std::printf( "sampling case 1: [%6.2f,%6.2f][%6.2f,%6.2f]\n", dUMinimum, dUMaximum, dVForUMinimum, dVForUMaximum );
                            }

                            assert( 0.0 <= dUMaximum && m_pCameraLEFT->m_dWidthPixel >= dUMaximum );
                            assert( 0.0 <= dUMinimum && m_pCameraLEFT->m_dWidthPixel >= dUMinimum );
                            assert( 0.0 <= dVForUMaximum && m_pCameraLEFT->m_dHeightPixel >= dVForUMaximum );
                            assert( 0.0 <= dVForUMinimum && m_pCameraLEFT->m_dHeightPixel >= dVForUMinimum );
                            assert( 0.0 <= dUMaximum-dUMinimum );
                            assert( 0.0 <= dVForUMaximum-dVForUMinimum );

                            //ds compute pixel ranges to sample
                            const uint32_t uDeltaU = dUMaximum-dUMinimum;
                            const uint32_t uDeltaV = dVForUMaximum-dVForUMinimum;

                            //ds draw last position
                            cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

                            //ds the match to find
                            std::shared_ptr< CMatchTracking > pMatchLEFT = 0;

                            //ds sample the larger range
                            if( uDeltaV < uDeltaU )
                            {
                                //ds get the match over U
                                pMatchLEFT = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, dUMinimum, uDeltaU, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                            }
                            else
                            {
                                //ds get the match over V
                                pMatchLEFT = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, dVForUMinimum, uDeltaV, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                            }

                            //ds add this measurement to the landmark
                            _addMeasurementToLandmarkLEFT( p_uFrame, pLandmark, p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor, p_matTransformationLEFTtoWORLD, p_vecCameraOrientation, matProjectionWORLDtoLEFT );

                            //ds register measurement
                            vecMeasurementsPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );

                            //ds update info
                            ++uDetectionsEpipolar;
                        }
                        catch( const CExceptionEpipolarLine& p_cException )
                        {
                            //std::printf( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) landmark [%06lu] epipolar failure: %s\n", pLandmark->uID, p_cException.what( ) );
                            ++pLandmark->uFailedSubsequentTrackings;
                            pLandmark->bIsCurrentlyVisible = false;
                        }
                        catch( const CExceptionNoMatchFound& p_cException )
                        {
                            //std::printf( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) landmark [%06lu] matching failure: %s\n", pLandmark->uID, p_cException.what( ) );
                            ++pLandmark->uFailedSubsequentTrackings;
                            pLandmark->bIsCurrentlyVisible = false;
                        }

                        //ds check activity
                        if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmark->uFailedSubsequentTrackings )
                        {
                            vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                        }
                    }
                }
            }
        }

        //ds log
        CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cDetectionPoint.uID, cDetectionPoint.vecLandmarks->size( ), vecActiveLandmarksPerDetectionPoint->size( ), vecMeasurementsPerDetectionPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerDetectionPoint->empty( ) )
        {
            //ds register the measurement point and its active landmarks anew
            vecDetectionPointsActive.push_back( CDetectionPoint( cDetectionPoint.uID, cDetectionPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerDetectionPoint ) );

            //ds combine visible landmarks
            vecMeasurements->insert( vecMeasurements->end( ), vecMeasurementsPerDetectionPoint.begin( ), vecMeasurementsPerDetectionPoint.end( ) );
        }
        else
        {
            //std::printf( "<CFundamentalMatcher>(getVisibleLandmarksFundamental) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds update info
    m_uNumberOfDetectionsEpipolar = uDetectionsEpipolar;

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecDetectionPointsActive );

    return vecMeasurements;
}

const cv::Mat CFundamentalMatcher::getMaskVisibleLandmarks( ) const
{
    //ds compute mask to avoid detecting similar features
    cv::Mat matMaskDetection( cv::Mat( m_pCameraSTEREO->m_uPixelHeight, m_pCameraSTEREO->m_uPixelWidth, CV_8UC1, cv::Scalar ( 255 ) ) );

    //ds draw black circles for existing landmark positions into the mask (avoid redetection of landmarks)
    for( const CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        cv::circle( matMaskDetection, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
    }

    return matMaskDetection;
}

const cv::Mat CFundamentalMatcher::getMaskActiveLandmarks( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT, cv::Mat& p_matDisplayLEFT ) const
{
    //ds compute mask to avoid detecting similar features
    cv::Mat matMaskDetection( cv::Mat( m_pCameraSTEREO->m_uPixelHeight, m_pCameraSTEREO->m_uPixelWidth, CV_8UC1, cv::Scalar ( 255 ) ) );

    //ds draw black circles for existing landmark positions into the mask (avoid redetection of landmarks)
    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds if the landmark is visible we dont have to compute the reprojection
            if( pLandmark->bIsCurrentlyVisible )
            {
                cv::circle( matMaskDetection, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
            }
            else
            {
                //ds get into camera frame
                const CPoint3DCAMERA vecPointXYZ( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized );

                cv::circle( matMaskDetection, m_pCameraLEFT->getUV( vecPointXYZ ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
                cv::circle( p_matDisplayLEFT, m_pCameraLEFT->getUV( vecPointXYZ ), m_uFeatureRadiusForMask, cv::Scalar ( 0 ), -1 );
            }
        }
    }

    return matMaskDetection;
}

void CFundamentalMatcher::drawVisibleLandmarks( cv::Mat& p_matDisplayLEFT, cv::Mat& p_matDisplayRIGHT, const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT ) const
{
    //ds for all visible landmarks
    for( const CLandmark* pLandmark: m_vecVisibleLandmarks )
    {
        //ds compute green brightness based on depth (further away -> darker)
        uint8_t uGreenValue = 255-std::sqrt( pLandmark->getLastDepth( ) )*20;
        cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );
        cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );

        //ds get 3d position in current camera frame
        const CPoint3DCAMERA vecXYZLEFT( p_matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized );

        //ds also draw reprojections
        cv::circle( p_matDisplayLEFT, m_pCameraLEFT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
        cv::circle( p_matDisplayRIGHT, m_pCameraRIGHT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                                                                const cv::Mat& p_matImage,
                                                                                                const double& p_dUMinimum,
                                                                                                const uint32_t& p_uDeltaU,
                                                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                                                const double& p_dKeyPointSize,
                                                                                                const uint8_t& p_uRecursionDepth ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_uDeltaU );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( uint32_t u = 0; u < p_uDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const double dU( p_dUMinimum+u );
            const double dV( _getCurveV( p_vecCoefficients, dU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( uint32_t u = 0; u < p_uDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const double dU( p_dUMinimum+u );
            const double dV( _getCurveV( p_vecCoefficients, dU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( dU, dV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
    }
    catch( const CExceptionNoMatchFoundInternal& p_cException )
    {
        //ds escape if the limit is reached
        if( m_uRecursionLimitEpipolarLines < p_uRecursionDepth )
        {
            throw CExceptionNoMatchFound( p_cException.what( ) );
        }
        else
        {
            //ds sample further
            return _getMatchSampleRecursiveU( p_matDisplay, p_matImage, p_dUMinimum, p_uDeltaU, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                                                                const cv::Mat& p_matImage,
                                                                                                const double& p_dVMinimum,
                                                                                                const uint32_t& p_uDeltaV,
                                                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                                                const double& p_dKeyPointSize,
                                                                                                const uint8_t& p_uRecursionDepth ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_uDeltaV );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( uint32_t v = 0; v < p_uDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const double dV( p_dVMinimum+v );
            const double dU( _getCurveU( p_vecCoefficients, dV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( dU, dV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( uint32_t v = 0; v < p_uDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const double dV( p_dVMinimum+v );
            const double dU( _getCurveU( p_vecCoefficients, dV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( dU, dV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( dU, dV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
    }
    catch( const CExceptionNoMatchFoundInternal& p_cException )
    {
        //ds escape if the limit is reached
        if( m_uRecursionLimitEpipolarLines < p_uRecursionDepth )
        {
            throw CExceptionNoMatchFound( p_cException.what( ) );
        }
        else
        {
            //ds sample further
            return _getMatchSampleRecursiveV( p_matDisplay, p_matImage, p_dVMinimum, p_uDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const std::shared_ptr< CMatchTracking > CFundamentalMatcher::_getMatch( const cv::Mat& p_matImage,
                                                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                                                     const CDescriptor& p_matDescriptorReference,
                                                                     const CDescriptor& p_matDescriptorOriginal ) const
{
    //ds descriptor pool
    cv::Mat matPoolDescriptors;

    //ds compute descriptors of current search area
    m_pExtractor->compute( p_matImage, p_vecPoolKeyPoints, matPoolDescriptors );

    //ds escape if we didnt find any descriptors
    if( p_vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (empty KeyPoint pool)" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matDescriptorReference, matPoolDescriptors, vecMatches );

    //ds escape for no matches
    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFoundInternal( "could not find any matches (empty DMatches pool)" );
    }

    //ds buffer first match
    const cv::DMatch& cBestMatch( vecMatches[0] );

    //ds check if we are in the range (works for negative ids as well)
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( cBestMatch.trainIdx ) < p_vecPoolKeyPoints.size( ) );

    //ds bufffer new descriptor
    const CDescriptor matDescriptorNew( matPoolDescriptors.row(cBestMatch.trainIdx) );

    //ds distances
    const double dMatchingDistanceToRelative( cBestMatch.distance );
    const double dMatchingDistanceToOriginal( cv::norm( p_matDescriptorOriginal, matDescriptorNew, cv::NORM_HAMMING ) );

    if( m_dMatchingDistanceCutoffTrackingEpipolar > dMatchingDistanceToRelative )
    {
        if( m_dMatchingDistanceCutoffOriginal > dMatchingDistanceToOriginal )
        {
            //ds return the match
            return std::make_shared< CMatchTracking >( p_vecPoolKeyPoints[cBestMatch.trainIdx], matDescriptorNew );
        }
        else
        {
            throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (ORIGINAL matching distance: "+ std::to_string( dMatchingDistanceToOriginal ) +")" );
        }
    }
    else
    {
        throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (matching distance: "+ std::to_string( cBestMatch.distance ) +")" );
    }
}

void CFundamentalMatcher::_addMeasurementToLandmarkLEFT( const UIDFrame p_uFrame,
                                                  CLandmark* p_pLandmark,
                                                  const cv::Mat& p_matImageRIGHT,
                                                  const cv::KeyPoint& p_cKeyPoint,
                                                  const CDescriptor& p_matDescriptorNew,
                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                  const Eigen::Vector3d& p_vecCameraOrientation,
                                                  const MatrixProjection& p_matProjectionWORLDtoLEFT )
{
    //ds buffer point
    cv::Point2f ptUVLEFT( p_cKeyPoint.pt );

    assert( m_pCameraLEFT->m_cFOV.contains( p_cKeyPoint.pt ) );

    //ds triangulate point
    const CMatchTriangulation cMatch( m_pTriangulator->getPointTriangulatedCompactInRIGHT( p_matImageRIGHT, p_cKeyPoint, p_matDescriptorNew ) );
    const CPoint3DCAMERA vecPointXYZLEFT( cMatch.vecPointXYZCAMERA );
    const cv::Point2f ptUVRIGHT( cMatch.ptUVCAMERA );

    //ds depth
    const double dDepthMeters = vecPointXYZLEFT.z( );

    //ds check depth
    if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
    {
        throw CExceptionNoMatchFound( "<CFundamentalMatcher>(_addMeasurementToLandmark) invalid depth: " + std::to_string( dDepthMeters ) );
    }

    assert( m_pCameraRIGHT->m_cFOV.contains( ptUVRIGHT ) );

    //ds update landmark (NO EXCEPTIONS HERE)
    p_pLandmark->bIsCurrentlyVisible        = true;
    p_pLandmark->matDescriptorLASTLEFT      = p_matDescriptorNew;
    p_pLandmark->matDescriptorLASTRIGHT     = cMatch.matDescriptorCAMERA;
    p_pLandmark->uFailedSubsequentTrackings = 0;
    p_pLandmark->addMeasurement( p_uFrame,
                                 ptUVLEFT,
                                 ptUVRIGHT,
                                 vecPointXYZLEFT,
                                 p_matTransformationLEFTtoWORLD*vecPointXYZLEFT,
                                 p_matTransformationLEFTtoWORLD.translation( ),
                                 p_vecCameraOrientation,
                                 p_matProjectionWORLDtoLEFT,
                                 cMatch.matDescriptorCAMERA );

    //ds add to vector for fast search
    m_vecVisibleLandmarks.push_back( p_pLandmark );
}

void CFundamentalMatcher::_addMeasurementToLandmarkSTEREO( const UIDFrame p_uFrame,
                                                  CMatchPoseOptimizationSTEREO& p_cMatchSTEREO,
                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                  const Eigen::Vector3d& p_vecCameraOrientation,
                                                  const MatrixProjection& p_matProjectionWORLDtoLEFT )
{
    //ds input validation
    assert( m_pCameraLEFT->m_cFOV.contains( p_cMatchSTEREO.ptUVLEFT ) );
    assert( m_pCameraRIGHT->m_cFOV.contains( p_cMatchSTEREO.ptUVRIGHT ) );
    assert( m_dMinimumDepthMeters < p_cMatchSTEREO.vecPointXYZLEFT.z( ) );
    assert( m_dMaximumDepthMeters > p_cMatchSTEREO.vecPointXYZLEFT.z( ) );

    //ds update landmark (NO EXCEPTIONS HERE)
    p_cMatchSTEREO.pLandmark->bIsCurrentlyVisible        = true;
    p_cMatchSTEREO.pLandmark->matDescriptorLASTLEFT      = p_cMatchSTEREO.matDescriptorLEFT;
    p_cMatchSTEREO.pLandmark->matDescriptorLASTRIGHT     = p_cMatchSTEREO.matDescriptorRIGHT;
    p_cMatchSTEREO.pLandmark->uFailedSubsequentTrackings = 0;
    p_cMatchSTEREO.pLandmark->addMeasurement( p_uFrame,
                                              p_cMatchSTEREO.ptUVLEFT,
                                              p_cMatchSTEREO.ptUVRIGHT,
                                              p_cMatchSTEREO.vecPointXYZLEFT,
                                              p_matTransformationLEFTtoWORLD*p_cMatchSTEREO.vecPointXYZLEFT,
                                              p_matTransformationLEFTtoWORLD.translation( ),
                                              p_vecCameraOrientation,
                                              p_matProjectionWORLDtoLEFT,
                                              p_cMatchSTEREO.matDescriptorLEFT );

    //ds add to vector for fast search
    m_vecVisibleLandmarks.push_back( p_cMatchSTEREO.pLandmark );
}

const double CFundamentalMatcher::_getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_dV ) const
{
    return -( p_vecCoefficients(1)*p_dV+p_vecCoefficients(2) )/p_vecCoefficients(0);
}
const double CFundamentalMatcher::_getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_dU ) const
{
    return -( p_vecCoefficients(0)*p_dU+p_vecCoefficients(2) )/p_vecCoefficients(1);
}
