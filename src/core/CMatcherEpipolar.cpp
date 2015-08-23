#include "CMatcherEpipolar.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"
#include "optimization/CBridgeG2O.h"
#include "exceptions/CExceptionEpipolarLine.h"

CMatcherEpipolar::CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                                    const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle,
                                    const double& p_dMinimumDepthMeters,
                                    const double& p_dMaximumDepthMeters,
                                    const double& p_dMatchingDistanceCutoffPoseOptimization,
                                    const double& p_dMatchingDistanceCutoffEssential,
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
                                                                              m_dMatchingDistanceCutoffTrackingEssential( p_dMatchingDistanceCutoffEssential ),
                                                                              m_dMatchingDistanceCutoffOriginal( 2*m_dMatchingDistanceCutoffTrackingEssential ),
                                                                              m_uAvailableDetectionPointID( 0 ),
                                                                              m_iSearchUMin( 5 ),
                                                                              m_iSearchUMax( m_pCameraLEFT->m_uWidthPixel-5 ),
                                                                              m_iSearchVMin( 5 ),
                                                                              m_iSearchVMax( m_pCameraLEFT->m_uHeightPixel-5 ),
                                                                              m_cSearchROI( cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ) ),
                                                                              m_uMaximumFailedSubsequentTrackingsPerLandmark( p_uMaximumFailedSubsequentTrackingsPerLandmark ),
                                                                              m_cSolverPoseProjection( m_pCameraLEFT->m_matProjection ),
                                                                              m_cSolverPoseSTEREO( m_pCameraSTEREO )
{
    CLogger::openBox( );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) minimum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) maximum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) matching distance cutoff (pose optimization): %f\n", m_dMatchingDistanceCutoffPoseOptimization );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) matching distance cutoff (essential line): %f\n", m_dMatchingDistanceCutoffTrackingEssential );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) instance allocated\n" );
    CLogger::closeBox( );
}

CMatcherEpipolar::~CMatcherEpipolar( )
{
    CLogger::CLogDetectionEpipolar::close( );
    CLogger::CLogOptimizationOdometry::close( );
    std::printf( "<CMatcherEpipolar>(~CMatcherEpipolar) instance deallocated\n" );
}

void CMatcherEpipolar::addDetectionPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
{
    m_vecDetectionPointsActive.push_back( CDetectionPoint( m_uAvailableDetectionPointID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );

    ++m_uAvailableDetectionPointID;
}

//ds routine that resets the visibility of all active landmarks
void CMatcherEpipolar::resetVisibilityActiveLandmarks( )
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

void CMatcherEpipolar::setKeyFrameToVisibleLandmarks( )
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

const std::shared_ptr< const std::vector< CLandmark* > > CMatcherEpipolar::getVisibleOptimizedLandmarks( ) const
{
    //ds return vector
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            if( pLandmark->bIsCurrentlyVisible && CBridgeG2O::isOptimized( pLandmark ) )
            {
                vecVisibleLandmarks->push_back( pLandmark );
            }
        }
    }

    return vecVisibleLandmarks;
}

/*const Eigen::Isometry3d CMatcherEpipolar::getPoseOptimizedLEFT( const uint64_t p_uFrame,
                                                            cv::Mat& p_matDisplayLEFT,
                                                            const cv::Mat& p_matImageLEFT,
                                                            const cv::Mat& p_matImageRIGHT,
                                                            const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                            const Eigen::Vector3d& p_vecCameraOrientation )
{
    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePoints;

    //ds found landmarks in this frame
    std::vector< CMatchPoseOptimizationLEFT > vecLandmarkMatches;

    //ds active measurements
    for( const CDetectionPoint& cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cDetectionPoint.vecLandmarks )
        {
            //ds world position
            const CPoint3DWORLD vecPointXYZ( pLandmarkReference->vecPointXYZOptimized );

            //ds compute current reprojection point with the prior
            const cv::Point2d ptUVLEFTEstimate( m_pCameraLEFT->getProjection( static_cast< CPoint3DCAMERA >( p_matTransformationEstimateWORLDtoLEFT*vecPointXYZ ) ) );

            //ds check if in visible range
            if( m_cSearchROI.contains( ptUVLEFTEstimate ) )
            {
                //ds compute search rectangle
                const double dScaleSearchU     = 1.0 + std::fabs( ptUVLEFTEstimate.x - m_pCameraLEFT->m_dCx )/100.0;
                const double dScaleSearchV     = 1.0 + std::fabs( ptUVLEFTEstimate.y - m_pCameraLEFT->m_dCy )/100.0;
                const double dSearchHalfWidth  = dScaleSearchU*m_uSearchBlockSizePoseOptimization;
                const double dSearchHalfHeight = dScaleSearchV*m_uSearchBlockSizePoseOptimization;
                cv::Point2d ptUpperLeft( std::max( ptUVLEFTEstimate.x-dSearchHalfWidth, 0.0 ), std::max( ptUVLEFTEstimate.y-dSearchHalfHeight, 0.0 ) );
                cv::Point2d ptLowerRight( std::min( ptUVLEFTEstimate.x+dSearchHalfWidth, m_pCameraLEFT->m_dWidthPixel ), std::min( ptUVLEFTEstimate.y+dSearchHalfHeight, m_pCameraLEFT->m_dHeightPixel ) );

                const cv::Rect cSearchROI( ptUpperLeft, ptLowerRight );
                cv::rectangle( p_matDisplayLEFT, cSearchROI, CColorCodeBGR( 255, 0, 0 ) );

                //ds detect features in this area
                std::vector< cv::KeyPoint > vecDetections;
                m_pDetector->detect( p_matImageLEFT( cSearchROI ), vecDetections );

                //ds if the landmark has been detected
                if( 0 < vecDetections.size( ) )
                {
                    //ds adjust keypoint offsets
                    for( cv::KeyPoint& cKeyPoint: vecDetections )
                    {
                        cKeyPoint.pt.x += ptUpperLeft.x;
                        cKeyPoint.pt.y += ptUpperLeft.y;
                    }

                    //ds check descriptor matches
                    cv::Mat matDescriptors;
                    m_pExtractor->compute( p_matImageLEFT, vecDetections, matDescriptors );
                    std::vector< cv::DMatch > vecMatches;
                    m_pMatcher->match( pLandmarkReference->matDescriptorLASTLEFT, matDescriptors, vecMatches );

                    //ds if we got a match
                    if( 0 < vecMatches.size( ) )
                    {
                        //ds and the matching distance is within the range
                        if( m_dMatchingDistanceCutoffPoseOptimization > vecMatches[0].distance )
                        {
                            //ds buffer the match
                            const cv::Point2f ptBestMatch( vecDetections[vecMatches[0].trainIdx].pt );

                            //ds store values for optimization
                            vecLandmarksWORLD.push_back( vecPointXYZ );
                            vecImagePoints.push_back( CPoint2DInCameraFrame( ptBestMatch.x, ptBestMatch.y ) );

                            //ds latter landmark update (cannot be done before pose is optimized)
                            vecLandmarkMatches.push_back( CMatchPoseOptimizationLEFT( pLandmarkReference, vecDetections[vecMatches[0].trainIdx], matDescriptors.row(vecMatches[0].trainIdx) ) );

                            cv::circle( p_matDisplayLEFT, ptBestMatch, 4, CColorCodeBGR( 0, 255, 0 ), 1 );
                        }
                        else
                        {
                            cv::circle( p_matDisplayLEFT, vecDetections[vecMatches[0].trainIdx].pt, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
                            //std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) landmark [%06lu] matching failed (matching distance: %f) \n", pLandmarkReference->uID, vecMatches[0].distance );
                        }
                    }
                    else
                    {
                        std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) landmark [%06lu] no matches found\n", pLandmarkReference->uID );
                    }
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) landmark [%06lu] no feature detected\n", pLandmarkReference->uID );
                }
            }
            else
            {
                std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) landmark [%06lu] out of visible range: (%6.2f %6.2f)\n", pLandmarkReference->uID, ptUVLEFTEstimate.x, ptUVLEFTEstimate.y );
            }
        }
    }

    //ds check if we have a sufficient number of points to optimize
    if( m_uMinimumPointsForPoseOptimization < vecLandmarksWORLD.size( ) )
    {
        //ds feed the solver with the 3D points (in camera frame)
        m_cSolverPoseProjection.model_points = vecLandmarksWORLD;

        //ds feed the solver with the 2D points
        m_cSolverPoseProjection.image_points = vecImagePoints;

        //ds initial guess of the transformation
        m_cSolverPoseProjection.T = p_matTransformationEstimateWORLDtoLEFT;

        //ds initialize solver
        m_cSolverPoseProjection.init( );

        //ds convergence
        double dErrorPrevious( 0.0 );

        //ds run KLS
        for( uint8_t uIteration = 0; uIteration < m_uCapIterationsPoseOptimization; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseProjection.oneRound( );
            uint32_t uInliersCurrent             = m_cSolverPoseProjection.num_inliers;

            //ds log the error evolution
            CLogger::CLogOptimizationOdometry::addEntryIteration( p_uFrame, uIteration, vecLandmarksWORLD.size( ), uInliersCurrent,  m_cSolverPoseProjection.num_reprojected_points, dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( m_dConvergenceDeltaPoseOptimization > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumPointsForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrent = m_cSolverPoseProjection.oneRoundInliersOnly( );

                    //ds log the error evolution
                    CLogger::CLogOptimizationOdometry::addEntryInliers( p_uFrame,vecLandmarksWORLD.size( ), m_cSolverPoseProjection.num_inliers, m_cSolverPoseProjection.num_reprojected_points, dErrorSolverPoseCurrent );
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        //ds precompute
        const Eigen::Isometry3d matTransformationLEFTtoWORLD( m_cSolverPoseProjection.T.inverse( ) );
        const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*m_cSolverPoseProjection.T.matrix( ) );

        //ds update all visible landmarks
        for( CMatchPoseOptimizationLEFT pMatch: vecLandmarkMatches )
        {
            try
            {
                _addMeasurementToLandmarkLEFT( p_uFrame, pMatch.pLandmark, p_matImageRIGHT, pMatch.cKeyPoint, pMatch.matDescriptor, matTransformationLEFTtoWORLD, p_vecCameraOrientation, matProjectionWORLDtoLEFT );
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                //std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) landmark [%06lu] caught exception: %s\n", pMatch.pLandmark->uID, p_eException.what( ) );
            }
        }

        //std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) optimized pose with %lu landmarks (updated)\n", vecLandmarksWORLD.size( ) );

        return m_cSolverPoseProjection.T;
    }
    else
    {
        std::printf( "<CMatcherEpipolar>(getPoseOptimizedLEFT) unable to optimize pose (%lu landmarks)\n", vecLandmarksWORLD.size( ) );
        return p_matTransformationEstimateWORLDtoLEFT;
    }
}*/

const Eigen::Isometry3d CMatcherEpipolar::getPoseOptimizedSTEREO( const uint64_t p_uFrame,
                                                                  cv::Mat& p_matDisplayLEFT,
                                                                  cv::Mat& p_matDisplayRIGHT,
                                                                  const cv::Mat& p_matImageLEFT,
                                                                  const cv::Mat& p_matImageRIGHT,
                                                                  const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                                  const Eigen::Vector3d& p_vecCameraOrientation,
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
            if( m_cSearchROI.contains( ptUVEstimateLEFT ) && m_cSearchROI.contains( ptUVEstimateRIGHT ) )
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
                    //std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] LEFT failed: %s\n", pLandmark->uID, p_cException.what( ) );

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
                                        std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: triangulation descriptor mismatch\n", pLandmark->uID );
                                    }*/
                                }
                                /*else
                                {
                                    std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: descriptor mismatch\n", pLandmark->uID );
                                }*/
                            }
                            /*else
                            {
                                std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: no matches found\n", pLandmark->uID );
                            }*/
                        }
                        /*else
                        {
                            std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: no features detected\n", pLandmark->uID );
                        }*/
                    }
                    catch( const CExceptionNoMatchFound& p_cException )
                    {
                        //std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] RIGHT failed: %s\n", pLandmark->uID, p_cException.what( ) );
                    }
                }
            }
            /*else
            {
                std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] out of visible range: LEFT(%6.2f %6.2f) RIGHT(%6.2f %6.2f)\n", pLandmark->uID, ptUVEstimateLEFT.x, ptUVEstimateLEFT.y, ptUVEstimateRIGHT.x, ptUVEstimateRIGHT.y );
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
                    //std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        //ds precompute
        const Eigen::Isometry3d matTransformationLEFTtoWORLD( m_cSolverPoseSTEREO.T.inverse( ) );

        //ds qualitiy information
        const double dDeltaOptimization      = ( matTransformationLEFTtoWORLD.translation( )-vecTranslationEstimate ).squaredNorm( );
        const double dOptimizationCovariance = dDeltaOptimization/p_dMotionScaling;

        //ds log resulting trajectory and delta to initial
        CLogger::CLogOptimizationOdometry::addEntryResult( matTransformationLEFTtoWORLD.translation( ), dDeltaOptimization, p_dMotionScaling, dOptimizationCovariance );

        //ds check if acceptable
        if( 0.5 > dOptimizationCovariance )
        {
            //ds update all visible landmarks
            const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*m_cSolverPoseSTEREO.T.matrix( ) );
            for( CMatchPoseOptimizationSTEREO pMatchSTEREO: vecLandmarkMatches )
            {
                _addMeasurementToLandmarkSTEREO( p_uFrame, pMatchSTEREO, matTransformationLEFTtoWORLD, p_vecCameraOrientation, matProjectionWORLDtoLEFT );
            }

            //ds update info
            m_uNumberOfDetectionsPoseOptimization = vecLandmarkMatches.size( );

            //ds return optimized pose
            return m_cSolverPoseSTEREO.T;
        }
        else
        {
            throw CExceptionPoseOptimization( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) unable to optimize pose (RISK: " + std::to_string( dOptimizationCovariance )+ ")" );
        }
    }
    else
    {
        throw CExceptionPoseOptimization( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) unable to optimize pose (insufficient number of points: " + std::to_string( vecLandmarksWORLD.size( ) ) + ")" );
    }
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssentialOptimized( const uint64_t p_uFrame,
                                                                                                                    cv::Mat& p_matDisplayLEFT,
                                                                                                                    cv::Mat& p_matDisplayRIGHT,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                    const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                                    const int32_t& p_iHalfLineLengthBase,
                                                                                                                    cv::Mat& p_matDisplayTrajectory )
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const Eigen::Matrix3d matKRotation( m_pCameraLEFT->m_matIntrinsic*matTransformationWORLDtoLEFT.linear( ) );
    const Eigen::Vector3d vecCameraPosition( matTransformationWORLDtoLEFT.translation( ) );
    const Eigen::Vector3d vecKTranslation( m_pCameraLEFT->m_matIntrinsic*vecCameraPosition );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecKeyFramesActive;

    //ds rightside keypoints buffer for descriptor computation
    std::vector< cv::KeyPoint > vecKeyPointRIGHT( 1 );

    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePointsLEFT;
    gtools::Vector2dVector vecImagePointsRIGHT;

    //ds active measurements
    for( const CDetectionPoint cKeyFrame: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerKeyFrame;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerKeyFrame( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds compute essential matrix for this detection point
        const Eigen::Isometry3d matTransformationToNow( matTransformationWORLDtoLEFT*cKeyFrame.matTransformationLEFTtoWORLD );
        const Eigen::Matrix3d matEssential( matTransformationToNow.linear( )*CMiniVisionToolbox::getSkew( matTransformationToNow.translation( ) ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cKeyFrame.vecLandmarks )
        {
            //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( matTransformationWORLDtoLEFT*pLandmarkReference->vecPointXYZOptimized ) ) );

            //ds compute maximum and minimum points (from top to bottom line)
            const int32_t iULastDetection( ptProjection.x );
            const int32_t iVLastDetection( ptProjection.y );

            //ds compute distance to principal point
            const double dPrincipalDistance( pLandmarkReference->vecUVLEFTReference.head( 2 ).norm( ) );

            //ds compute sampling line
            const int32_t iHalfLineLength( ( 1 + 0.1*pLandmarkReference->uFailedSubsequentTrackings + dPrincipalDistance )*p_iHalfLineLengthBase );

            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*pLandmarkReference->vecUVLEFTReference );

            //ds get back to pixel coordinates
            int32_t iUMinimum( std::max( iULastDetection-iHalfLineLength, m_iSearchUMin ) );
            int32_t iUMaximum( std::min( iULastDetection+iHalfLineLength, m_iSearchUMax ) );
            int32_t iVForUMinimum( _getCurveEssentialV( vecCoefficients, iUMinimum ) );
            int32_t iVForUMaximum( _getCurveEssentialV( vecCoefficients, iUMaximum ) );

            //std::printf( "U [%i, %i]\n", iUMinimum, iUMaximum );
            //std::printf( "V [%i, %i]\n", iVForUMinimum, iVForUMaximum );

            //ds check if the line is completely out of the visible region
            if( ( iVForUMinimum < m_iSearchVMin && iVForUMaximum < m_iSearchVMin ) ||
                ( iVForUMinimum > m_iSearchVMax && iVForUMaximum > m_iSearchVMax ) )
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] out of sight (epipolar line drifted)\n", pLandmarkReference->uID );
                continue;
            }

            //ds compute v border values
            const int32_t iVLimitMinimum( std::max( iVLastDetection-iHalfLineLength, m_iSearchVMin ) );
            const int32_t iVLimitMaximum( std::min( iVLastDetection+iHalfLineLength, m_iSearchVMax ) );

            //ds negative slope (max v is also at max u)
            if( iVForUMaximum > iVForUMinimum )
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMinimum > iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMinimum;
                    iUMinimum     = _getCurveEssentialU( vecCoefficients, iVLimitMinimum );
                }
                if( iVLimitMaximum < iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMaximum;
                    iUMaximum     = _getCurveEssentialU( vecCoefficients, iVLimitMaximum );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMaximum;
                    iUMinimum     = _getCurveEssentialU( vecCoefficients, iVLimitMaximum );
                }
                if( iVLimitMinimum > iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMinimum;
                    iUMaximum     = _getCurveEssentialU( vecCoefficients, iVLimitMinimum );
                }

                //ds swap required for uniform looping
                std::swap( iVForUMinimum, iVForUMaximum );
            }

            //ds escape for invalid points (caused by invalid projections)
            if( iUMinimum > iUMaximum || iVForUMinimum > iVForUMaximum )
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] out of sight (U[%i,%i] V[%i,%i]) dropped\n", pLandmarkReference->uID, iUMinimum, iUMaximum, iVForUMinimum, iVForUMaximum );
                continue;
            }

            assert( 0 <= iUMinimum );
            assert( m_pCameraLEFT->m_iWidthPixel >= iUMaximum );
            assert( 0 <= iVForUMinimum );
            assert( m_pCameraLEFT->m_iHeightPixel >= iVForUMaximum );
            assert( 0 <= iUMaximum-iUMinimum );
            assert( 0 <= iVForUMaximum-iVForUMinimum );

            //ds compute pixel ranges to sample
            const uint32_t uDeltaU( iUMaximum-iUMinimum );
            const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

            //ds draw last position
            cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

            //ds the match to find
            const CMatchTracking* pMatchLEFT( 0 );

            try
            {
                //ds sample the larger range
                if( uDeltaU > uDeltaV )
                {
                    //ds get the match over U
                    pMatchLEFT = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorLASTLEFT, pLandmarkReference->matDescriptorReferenceLEFT, pLandmarkReference->dKeyPointSize, 0 );
                }
                else
                {
                    //ds get the match over V
                    pMatchLEFT = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorLASTLEFT, pLandmarkReference->matDescriptorReferenceLEFT, pLandmarkReference->dKeyPointSize, 0 );
                }

                const cv::Point2d& ptUVLEFT( pMatchLEFT->cKeyPoint.pt );

                assert( m_cSearchROI.contains( ptUVLEFT ) );

                //ds triangulate point
                const CPoint3DCAMERA vecPointTriangulatedLEFT( m_pTriangulator->getPointTriangulatedLimited( p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor ) );
                const CPoint3DWORLD vecPointXYZ( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT );

                //ds depth
                const double& dDepthMeters = vecPointTriangulatedLEFT.z( );

                //ds check depth
                if( m_dMinimumDepthMeters > dDepthMeters || m_dMaximumDepthMeters < dDepthMeters )
                {
                    throw CExceptionNoMatchFound( "<CMatcherEpipolar>(getVisibleLandmarksEssential) invalid depth: " + std::to_string( dDepthMeters ) );
                }

                //ds get projection
                cv::Point2d ptUVRIGHT( m_pCameraRIGHT->getProjection( vecPointTriangulatedLEFT ) );

                assert( m_cSearchROI.contains( ptUVRIGHT ) );

                //ds enforce epipolar constraint TODO integrate epipolar error
                const double dEpipolarError( ptUVRIGHT.y-ptUVLEFT.y );
                if( 0.1 < dEpipolarError )
                {
                    std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] epipolar error: %f\n", pLandmarkReference->uID, dEpipolarError );
                }
                ptUVRIGHT.y = ptUVLEFT.y;

                //ds compute reference descriptor on right side as well
                vecKeyPointRIGHT[0] = cv::KeyPoint( ptUVRIGHT, pMatchLEFT->cKeyPoint.size );
                CDescriptor matDescriptorRIGHT;
                m_pExtractor->compute( p_matImageRIGHT, vecKeyPointRIGHT, matDescriptorRIGHT );

                //ds update landmark
                pLandmarkReference->matDescriptorLASTLEFT      = pMatchLEFT->matDescriptor; //( pMatchLEFT->matDescriptor + pLandmarkReference->matDescriptorLastLEFT )/2.0;
                pLandmarkReference->matDescriptorLASTRIGHT     = matDescriptorRIGHT;
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addMeasurement( p_uFrame, ptUVLEFT, ptUVRIGHT, vecPointTriangulatedLEFT, vecPointXYZ, vecCameraPosition, p_vecCameraOrientation, matProjectionWORLDtoLEFT, pMatchLEFT->matDescriptor );

                //ds register measurement
                vecVisibleLandmarksPerKeyFrame.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds store elements for optimization
                vecLandmarksWORLD.push_back( pLandmarkReference->vecPointXYZOptimized );
                vecImagePointsLEFT.push_back( CWrapperOpenCV::fromCVVector( ptUVLEFT ) );
                vecImagePointsRIGHT.push_back( CWrapperOpenCV::fromCVVector( ptUVRIGHT ) );

                //ds new positions
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                char chBufferMiniInfo[20];
                std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmarkReference->uID, pLandmarkReference->uOptimizationsSuccessful, pLandmarkReference->dCurrentAverageSquaredError );
                cv::putText( p_matDisplayLEFT, chBufferMiniInfo, cv::Point2d( pMatchLEFT->cKeyPoint.pt.x+pLandmarkReference->dKeyPointSize, pMatchLEFT->cKeyPoint.pt.y+pLandmarkReference->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplayRIGHT, ptUVRIGHT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                //ds free handle
                delete pMatchLEFT;
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                ++pLandmarkReference->uFailedSubsequentTrackings;
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] caught exception: %s\n", pLandmarkReference->uID, p_eException.what( ) );
            }

            //ds draw projection
            cv::circle( p_matDisplayLEFT, ptProjection, pLandmarkReference->dKeyPointSize, CColorCodeBGR( 0, 0, 255 ), 1 );

            //ds check activity
            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings )
            {
                vecActiveLandmarksPerKeyFrame->push_back( pLandmarkReference );
            }
            else
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] dropped\n", pLandmarkReference->uID );
            }
        }

        //ds log
        CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cKeyFrame.uID, cKeyFrame.vecLandmarks->size( ), vecActiveLandmarksPerKeyFrame->size( ), vecVisibleLandmarksPerKeyFrame.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerKeyFrame->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecKeyFramesActive.push_back( CDetectionPoint( cKeyFrame.uID, cKeyFrame.matTransformationLEFTtoWORLD, vecActiveLandmarksPerKeyFrame ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerKeyFrame.begin( ), vecVisibleLandmarksPerKeyFrame.end( ) );
        }
        else
        {
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point\n" );
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

            //ds log the error evolution
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
                    std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        const Eigen::Isometry3d matTransformationLEFTtoWORLDCorrected( m_cSolverPoseSTEREO.T.inverse( ) );
        const cv::Point2d ptPositionXY( matTransformationLEFTtoWORLDCorrected.translation( ).x( ), matTransformationLEFTtoWORLDCorrected.translation( ).y( ) );
        cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+ptPositionXY.x*10, 360-ptPositionXY.y*10 ), 5, CColorCodeBGR( 0, 0, 255 ), 1 );
    }
    else
    {
        std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) unable to optimize pose (%lu points)\n", vecLandmarksWORLD.size( ) );
    }

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecKeyFramesActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( cv::Mat& p_matDisplayLEFT,
                                                                                                                    cv::Mat& p_matDisplayRIGHT,
                                                                                                                    const uint64_t p_uFrame,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                    const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                                    const int32_t& p_iHalfLineLengthBase )
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecDetectionPointsActive;

    //ds active measurements
    for( const CDetectionPoint cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerDetectionPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerDetectionPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds check relative transform
        const Eigen::Isometry3d matTransformationToNow( matTransformationWORLDtoLEFT*cDetectionPoint.matTransformationLEFTtoWORLD );
        const CPoint3DWORLD vecTranslationToNow( matTransformationToNow.translation( ) );

        //ds compute essential matrix for this detection point
        const Eigen::Matrix3d matEssential( matTransformationToNow.linear( )*CMiniVisionToolbox::getSkew( vecTranslationToNow ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmark: *cDetectionPoint.vecLandmarks )
        {
            //ds check if we can skip this landmark due to failed optimization
            if( 0 < pLandmark->uOptimizationsFailed )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] dropped (optimization failed, successful: %u, error: %f)\n", pLandmark->uID, pLandmark->uOptimizationsSuccessful, pLandmark->dCurrentAverageSquaredError );
                ++m_uNumberOfInvalidLandmarksTotal;
            }

            //ds check if we can skip this landmark due to invalid optimization
            else if( 0 < pLandmark->uOptimizationsSuccessful && !CBridgeG2O::isOptimized( pLandmark ) )
            {
                cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 4, CColorCodeBGR( 0, 0, 255 ), -1 );
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] dropped (invalid optimization, successful: %u, error: %f)\n", pLandmark->uID, pLandmark->uOptimizationsSuccessful, pLandmark->dCurrentAverageSquaredError );
                ++m_uNumberOfInvalidLandmarksTotal;
            }

            //ds process the landmark
            else
            {
                //ds check if already detected (in pose optimization)
                if( pLandmark->bIsCurrentlyVisible )
                {
                    //ds just register the measurement
                    vecVisibleLandmarksPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );
                    vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                }
                else
                {
                    //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
                    const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized ) ) );

                    //ds compute maximum and minimum points (from top to bottom line)
                    const int32_t iULastDetection( ptProjection.x );
                    const int32_t iVLastDetection( ptProjection.y );

                    //ds compute distance to principal point
                    const double dPrincipalDistance( pLandmark->vecUVLEFTReference.head( 2 ).norm( ) );

                    //ds compute sampling line
                    const int32_t iHalfLineLength( ( 1 + 0.1*pLandmark->uFailedSubsequentTrackings + dPrincipalDistance )*p_iHalfLineLengthBase );

                    //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
                    const Eigen::Vector3d vecCoefficients( matEssential*pLandmark->vecUVLEFTReference );

                    //ds get back to pixel coordinates
                    int32_t iUMinimum( std::max( iULastDetection-iHalfLineLength, m_iSearchUMin ) );
                    int32_t iUMaximum( std::min( iULastDetection+iHalfLineLength, m_iSearchUMax ) );
                    int32_t iVForUMinimum( _getCurveEssentialV( vecCoefficients, iUMinimum ) );
                    int32_t iVForUMaximum( _getCurveEssentialV( vecCoefficients, iUMaximum ) );

                    //std::printf( "U [%i, %i]\n", iUMinimum, iUMaximum );
                    //std::printf( "V [%i, %i]\n", iVForUMinimum, iVForUMaximum );

                    //ds check if the line is completely out of the visible region
                    if( ( iVForUMinimum < m_iSearchVMin && iVForUMaximum < m_iSearchVMin ) ||
                        ( iVForUMinimum > m_iSearchVMax && iVForUMaximum > m_iSearchVMax ) )
                    {
                        //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] out of sight (epipolar line drifted)\n", pLandmark->uID );
                        continue;
                    }

                    //ds compute v border values
                    const int32_t iVLimitMinimum( std::max( iVLastDetection-iHalfLineLength, m_iSearchVMin ) );
                    const int32_t iVLimitMaximum( std::min( iVLastDetection+iHalfLineLength, m_iSearchVMax ) );

                    //ds negative slope (max v is also at max u)
                    if( iVForUMaximum > iVForUMinimum )
                    {
                        //ds adjust ROI (recompute U)
                        if( iVLimitMinimum > iVForUMinimum )
                        {
                            iVForUMinimum = std::max( iVLimitMinimum, m_iSearchVMin );
                            iUMinimum     = std::max( _getCurveEssentialU( vecCoefficients, iVLimitMinimum ), m_iSearchUMin );
                        }
                        if( iVLimitMaximum < iVForUMaximum )
                        {
                            iVForUMaximum = std::min( iVLimitMaximum, m_iSearchVMax );
                            iUMaximum     = std::min( _getCurveEssentialU( vecCoefficients, iVLimitMaximum ), m_iSearchUMax );
                        }
                    }

                    //ds positive slope (max v is at min u)
                    else
                    {
                        //ds adjust ROI (recompute U)
                        if( iVLimitMaximum < iVForUMinimum )
                        {
                            iVForUMinimum = std::min( iVLimitMaximum, m_iSearchVMax );
                            iUMinimum     = std::max( _getCurveEssentialU( vecCoefficients, iVLimitMaximum ), m_iSearchUMin );
                        }
                        if( iVLimitMinimum > iVForUMaximum )
                        {
                            iVForUMaximum = std::max( iVLimitMinimum, m_iSearchVMin );
                            iUMaximum     = std::min( _getCurveEssentialU( vecCoefficients, iVLimitMinimum ), m_iSearchUMax );
                        }

                        //ds for positive looping
                        std::swap( iVForUMinimum, iVForUMaximum );
                    }

                    //ds check if there was an overlap TODO FIXIT
                    if( iVForUMinimum > iVForUMaximum ){ continue; }
                    if( iUMinimum > iUMaximum ){ continue; }
                    if( m_pCameraLEFT->m_iWidthPixel < iUMaximum ){ continue; }
                    if( m_pCameraLEFT->m_iHeightPixel < iVForUMaximum ){ continue; }

                    assert( 0 <= iUMinimum );
                    assert( m_pCameraLEFT->m_iWidthPixel >= iUMaximum );
                    assert( 0 <= iVForUMinimum );
                    assert( m_pCameraLEFT->m_iHeightPixel >= iVForUMaximum );
                    assert( 0 <= iUMaximum-iUMinimum );
                    assert( 0 <= iVForUMaximum-iVForUMinimum );

                    //ds compute pixel ranges to sample
                    const uint32_t uDeltaU( iUMaximum-iUMinimum );
                    const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

                    //ds draw last position
                    cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

                    //ds the match to find
                    const CMatchTracking* pMatchLEFT( 0 );

                    try
                    {
                        //ds sample the larger range
                        if( uDeltaU > uDeltaV )
                        {
                            //ds get the match over U
                            pMatchLEFT = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, iUMinimum, uDeltaU, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                        }
                        else
                        {
                            //ds get the match over V
                            pMatchLEFT = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, iVForUMinimum, uDeltaV, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                        }

                        //ds add this measurement to the landmark
                        _addMeasurementToLandmarkLEFT( p_uFrame, pLandmark, p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor, p_matTransformationLEFTtoWORLD, p_vecCameraOrientation, matProjectionWORLDtoLEFT );

                        //ds register measurement
                        vecVisibleLandmarksPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );

                        //ds new positions
                        //cv::circle( p_matDisplayLEFT, pLandmark->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                        //ds draw reprojection of triangulation
                        //cv::circle( p_matDisplayRIGHT, pLandmark->getLastDetectionRIGHT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                        //ds free handle
                        delete pMatchLEFT;
                    }
                    catch( const CExceptionNoMatchFound& p_eException )
                    {
                        ++pLandmark->uFailedSubsequentTrackings;
                        //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] caught exception: %s\n", pLandmark->uID, p_eException.what( ) );
                    }

                    //ds draw projection
                    //cv::circle( p_matDisplayLEFT, ptProjection, pLandmark->dKeyPointSize, CColorCodeBGR( 0, 0, 255 ), 1 );
                    //char chBufferMiniInfo[20];
                    //std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmark->uID, pLandmark->uOptimizationsSuccessful, pLandmark->dCurrentAverageSquaredError );
                    //cv::putText( p_matDisplayLEFT, chBufferMiniInfo, cv::Point2d( ptProjection.x+pLandmark->dKeyPointSize, ptProjection.y+pLandmark->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

                    //ds check activity
                    if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmark->uFailedSubsequentTrackings )
                    {
                        vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                    }
                    else
                    {
                        //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] dropped (out of sight)\n", pLandmark->uID );
                    }
                }
            }
        }

        //ds log
        CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cDetectionPoint.uID, cDetectionPoint.vecLandmarks->size( ), vecActiveLandmarksPerDetectionPoint->size( ), vecVisibleLandmarksPerDetectionPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerDetectionPoint->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecDetectionPointsActive.push_back( CDetectionPoint( cDetectionPoint.uID, cDetectionPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerDetectionPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerDetectionPoint.begin( ), vecVisibleLandmarksPerDetectionPoint.end( ) );
        }
        else
        {
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecDetectionPointsActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksMocked( cv::Mat& p_matDisplayLEFT,
                                                                                                                 cv::Mat& p_matDisplayRIGHT,
                                                                                                                 const uint64_t p_uFrame,
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
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecKeyFramesActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksFundamental( cv::Mat& p_matDisplayLEFT,
                                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                                      const uint64_t p_uFrame,
                                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                      const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                                      const double& p_dMotionScaling )
{
    assert( 1.0 <= p_dMotionScaling );

    //ds detected landmarks at this position
    std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< const CMeasurementLandmark* > >( ) );

    //ds precompute inverse once
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds new active measurement points
    std::vector< CDetectionPoint > vecDetectionPointsActive;

    //ds compute initial sampling line
    const double dHalfLineLength = p_dMotionScaling*10;

    //ds total detections counter
    UIDLandmark uDetectionsEpipolar = 0;

    //ds active measurements
    for( const CDetectionPoint cDetectionPoint: m_vecDetectionPointsActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerDetectionPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerDetectionPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds check relative transform
        const Eigen::Isometry3d matTransformationToNow( matTransformationWORLDtoLEFT*cDetectionPoint.matTransformationLEFTtoWORLD );

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
                    vecVisibleLandmarksPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );
                    vecActiveLandmarksPerDetectionPoint->push_back( pLandmark );
                }
                else
                {
                    //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
                    const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DWORLD >( matTransformationWORLDtoLEFT*pLandmark->vecPointXYZOptimized ) ) );

                    try
                    {
                        //ds check if the projection is out of the fov
                        if( !m_pCameraLEFT->m_cFOV.contains( ptProjection ) )
                        {
                            throw CExceptionEpipolarLine( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) projection out of sight" );
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
                        const double dVForUMinimumRAW = _getCurveFundamentalV( vecCoefficients, dUMinimumRAW );
                        const double dVForUMaximumRAW = _getCurveFundamentalV( vecCoefficients, dUMaximumRAW );

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
                            throw CExceptionEpipolarLine( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) vertical out of sight" );
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
                                throw CExceptionEpipolarLine( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) caught bad projection negative slope" );
                            }

                            //ds regular case
                            if( dVLimitMinimum > dVForUMinimumRAW )
                            {
                                dVForUMinimum = dVLimitMinimum;
                                dUMinimum     = _getCurveFundamentalU( vecCoefficients, dVForUMinimum );
                            }
                            else
                            {
                                dVForUMinimum = dVForUMinimumRAW;
                            }
                            if( dVLimitMaximum < dVForUMaximumRAW )
                            {
                                dVForUMaximum = dVLimitMaximum;
                                dUMaximum     = _getCurveFundamentalU( vecCoefficients, dVForUMaximum );
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
                                throw CExceptionEpipolarLine( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) caught bad projection positive slope" );
                            }

                            //ds swapped case
                            if( dVLimitMinimum > dVForUMaximumRAW )
                            {
                                dVForUMinimum = dVLimitMinimum;
                                dUMaximum     = _getCurveFundamentalU( vecCoefficients, dVForUMinimum );
                            }
                            else
                            {
                                dVForUMinimum = dVForUMaximumRAW;
                            }
                            if( dVLimitMaximum < dVForUMinimumRAW )
                            {
                                dVForUMaximum = dVLimitMaximum;
                                dUMinimum     = _getCurveFundamentalU( vecCoefficients, dVForUMaximum );
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
                            pMatchLEFT = _getMatchSampleRecursiveFundamentalU( p_matDisplayLEFT, p_matImageLEFT, dUMinimum, uDeltaU, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                        }
                        else
                        {
                            //ds get the match over V
                            pMatchLEFT = _getMatchSampleRecursiveFundamentalV( p_matDisplayLEFT, p_matImageLEFT, dVForUMinimum, uDeltaV, vecCoefficients, pLandmark->matDescriptorLASTLEFT, pLandmark->matDescriptorReferenceLEFT, pLandmark->dKeyPointSize, 0 );
                        }

                        //ds add this measurement to the landmark
                        _addMeasurementToLandmarkLEFT( p_uFrame, pLandmark, p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor, p_matTransformationLEFTtoWORLD, p_vecCameraOrientation, matProjectionWORLDtoLEFT );

                        //ds register measurement
                        vecVisibleLandmarksPerDetectionPoint.push_back( pLandmark->getLastMeasurement( ) );

                        //ds update info
                        ++uDetectionsEpipolar;
                    }
                    catch( const CExceptionEpipolarLine& p_cException )
                    {
                        //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) landmark [%06lu] epipolar failure: %s\n", pLandmark->uID, p_cException.what( ) );
                        ++pLandmark->uFailedSubsequentTrackings;
                        pLandmark->bIsCurrentlyVisible = false;
                    }
                    catch( const CExceptionNoMatchFound& p_cException )
                    {
                        //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) landmark [%06lu] matching failure: %s\n", pLandmark->uID, p_cException.what( ) );
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

        //ds log
        CLogger::CLogDetectionEpipolar::addEntry( p_uFrame, cDetectionPoint.uID, cDetectionPoint.vecLandmarks->size( ), vecActiveLandmarksPerDetectionPoint->size( ), vecVisibleLandmarksPerDetectionPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerDetectionPoint->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecDetectionPointsActive.push_back( CDetectionPoint( cDetectionPoint.uID, cDetectionPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerDetectionPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerDetectionPoint.begin( ), vecVisibleLandmarksPerDetectionPoint.end( ) );
        }
        else
        {
            //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksFundamental) erased detection point [%06lu]\n", cDetectionPoint.uID );
        }
    }

    //ds update info
    m_uNumberOfDetectionsEpipolar = uDetectionsEpipolar;

    //ds update active measurement points
    m_vecDetectionPointsActive.swap( vecDetectionPointsActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const CMatchTracking* CMatcherEpipolar::_getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                                     const cv::Mat& p_matImage,
                                                                     const int32_t& p_iUMinimum,
                                                                     const int32_t& p_iDeltaU,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const CDescriptor& p_matReferenceDescriptor,
                                                                     const CDescriptor& p_matOriginalDescriptor,
                                                                     const double& p_dKeyPointSize,
                                                                     const uint8_t& p_uRecursionDepth ) const
{
    //ds compute sampling range
    //const int32_t iDeltaU( 0.95*p_iDeltaU );
    //const int32_t iUMinimum( p_iUMinimum+0.025*p_iDeltaU );

    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaU );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( int32_t i = 0; i < p_iDeltaU; ++i )
        {
            //ds compute corresponding V coordinate
            const int32_t iU( p_iUMinimum+i );
            const int32_t iV( _getCurveEssentialV( p_vecCoefficients, iU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[i] = cv::KeyPoint( iU, iV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( iU, iV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( int32_t i = 0; i < p_iDeltaU; ++i )
        {
            //ds compute corresponding V coordinate
            const int32_t iU( p_iUMinimum+i );
            const int32_t iV( _getCurveEssentialV( p_vecCoefficients, iU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[i] = cv::KeyPoint( iU, iV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( iU, iV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatchCStyle( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
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
            return _getMatchSampleRecursiveU( p_matDisplay, p_matImage, p_iUMinimum, p_iDeltaU, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
        }
    }
}

const CMatchTracking* CMatcherEpipolar::_getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                                     const cv::Mat& p_matImage,
                                                                     const int32_t& p_iVMinimum,
                                                                     const int32_t& p_iDeltaV,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const CDescriptor& p_matReferenceDescriptor,
                                                                     const CDescriptor& p_matOriginalDescriptor,
                                                                     const double& p_dKeyPointSize,
                                                                     const uint8_t& p_uRecursionDepth ) const
{
    //ds compute sampling range
    //const int32_t iDeltaV( 0.95*p_iDeltaV );
    //const int32_t iVMinimum( p_iVMinimum+0.025*p_iDeltaV );

    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaV );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( int32_t v = 0; v < p_iDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const int32_t uV( p_iVMinimum+v );
            const int32_t uU( _getCurveEssentialU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( int32_t v = 0; v < p_iDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const int32_t uV( p_iVMinimum+v );
            const int32_t uU( _getCurveEssentialU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatchCStyle( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
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
            return _getMatchSampleRecursiveV( p_matDisplay, p_matImage, p_iVMinimum, p_iDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
        }
    }
}

const std::shared_ptr< CMatchTracking > CMatcherEpipolar::_getMatchSampleRecursiveFundamentalU( cv::Mat& p_matDisplay,
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
            const double dV( _getCurveFundamentalV( p_vecCoefficients, dU ) + iSamplingOffset );

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
            const double dV( _getCurveFundamentalV( p_vecCoefficients, dU ) + iSamplingOffset );

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
            return _getMatchSampleRecursiveFundamentalU( p_matDisplay, p_matImage, p_dUMinimum, p_uDeltaU, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const std::shared_ptr< CMatchTracking > CMatcherEpipolar::_getMatchSampleRecursiveFundamentalV( cv::Mat& p_matDisplay,
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
            const double dU( _getCurveFundamentalU( p_vecCoefficients, dV ) + iSamplingOffset );

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
            const double dU( _getCurveFundamentalU( p_vecCoefficients, dV ) + iSamplingOffset );

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
            return _getMatchSampleRecursiveFundamentalV( p_matDisplay, p_matImage, p_dVMinimum, p_uDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+m_uRecursionStepSize );
        }
    }
}

const CMatchTracking* CMatcherEpipolar::_getMatchSampleRecursiveU( const cv::Mat& p_matImage,
                                                                     const int32_t& p_iUMinimum,
                                                                     const int32_t& p_iDeltaU,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const CDescriptor& p_matReferenceDescriptor,
                                                                     const CDescriptor& p_matOriginalDescriptor,
                                                                     const double& p_dKeyPointSize,
                                                                     const uint8_t& p_uRecursionDepth ) const
{
    //ds compute sampling range
    //const int32_t iDeltaU( 0.95*p_iDeltaU );
    //const int32_t iUMinimum( p_iUMinimum+0.025*p_iDeltaU );

    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaU );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( int32_t u = 0; u < p_iDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const int32_t uU( p_iUMinimum+u );
            const int32_t uV( _getCurveEssentialV( p_vecCoefficients, uU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( int32_t u = 0; u < p_iDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const int32_t uU( p_iUMinimum+u );
            const int32_t uV( _getCurveEssentialV( p_vecCoefficients, uU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatchCStyle( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
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
            return _getMatchSampleRecursiveU( p_matImage, p_iUMinimum, p_iDeltaU, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
        }
    }
}

const CMatchTracking* CMatcherEpipolar::_getMatchSampleRecursiveV( const cv::Mat& p_matImage,
                                                                     const int32_t& p_iVMinimum,
                                                                     const int32_t& p_iDeltaV,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const CDescriptor& p_matReferenceDescriptor,
                                                                     const CDescriptor& p_matOriginalDescriptor,
                                                                     const double& p_dKeyPointSize,
                                                                     const uint8_t& p_uRecursionDepth ) const
{
    //ds compute sampling range
    //const int32_t iDeltaV( 0.95*p_iDeltaV );
    //const int32_t iVMinimum( p_iVMinimum+0.025*p_iDeltaV );

    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaV );

    //ds determine sampling direction - if even we loop positively
    if( 0 == p_uRecursionDepth%2 )
    {
        const int8_t iSamplingOffset( p_uRecursionDepth );

        //ds sample over U
        for( int32_t v = 0; v < p_iDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const int32_t uV( p_iVMinimum+v );
            const int32_t uU( _getCurveEssentialU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
        }
    }
    else
    {
        const int8_t iSamplingOffset( -p_uRecursionDepth );

        //ds sample over U
        for( int32_t v = 0; v < p_iDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const int32_t uV( p_iVMinimum+v );
            const int32_t uU( _getCurveEssentialU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
        }
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatchCStyle( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matOriginalDescriptor );
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
            return _getMatchSampleRecursiveV( p_matImage, p_iVMinimum, p_iDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
        }
    }
}

const CMatchTracking* CMatcherEpipolar::_getMatchCStyle( const cv::Mat& p_matImage,
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
    //std::vector< std::vector< cv::DMatch > > vecMatches;
    m_pMatcher->match( p_matDescriptorReference, matPoolDescriptors, vecMatches );
    //m_pMatcher->radiusMatch( p_matDescriptorReference, matPoolDescriptors, vecMatches, m_fMatchingDistanceCutoff );

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
    
    //ds check relative matching to original descriptor
    //std::vector< cv::DMatch > vecMatchOriginal;
    //m_pMatcher->match( p_matDescriptorOriginal, matDescriptorNew, vecMatchOriginal );

    //ds distances
    const double& dMatchingDistanceToRelative( cBestMatch.distance );
    //const double& dMatchingDistanceToOriginal( vecMatchOriginal[0].distance );

    /*ds direct match
    if( m_dMatchingDistanceCutoff > dMatchingDistanceToOriginal )
    {
        //std::printf( "DIRECT MATCH - relative/original: %f/%f\n", dMatchingDistanceToRelative, dMatchingDistanceToOriginal );

        //ds return the match
        return CMatchTracking( p_vecPoolKeyPoints[cBestMatch.trainIdx], matDescriptorNew );
    }*/

    if( m_dMatchingDistanceCutoffTrackingEssential > dMatchingDistanceToRelative )
    {
        //if( m_dMatchingDistanceCutoffOriginal > dMatchingDistanceToOriginal )
        //{
            //ds return the match
            return new CMatchTracking( p_vecPoolKeyPoints[cBestMatch.trainIdx], matDescriptorNew );
        //}
        //else
        //{
        //    throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (ORIGINAL matching distance: "+ std::to_string( vecMatchOriginal[0].distance ) +")" );
        //}
    }
    else
    {
        throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (matching distance: "+ std::to_string( cBestMatch.distance ) +")" );
    }
}

const std::shared_ptr< CMatchTracking > CMatcherEpipolar::_getMatch( const cv::Mat& p_matImage,
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

    if( m_dMatchingDistanceCutoffTrackingEssential > dMatchingDistanceToRelative )
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

void CMatcherEpipolar::_addMeasurementToLandmarkLEFT( const uint64_t p_uFrame,
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
        throw CExceptionNoMatchFound( "<CMatcherEpipolar>(_addMeasurementToLandmark) invalid depth: " + std::to_string( dDepthMeters ) );
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
}

void CMatcherEpipolar::_addMeasurementToLandmarkSTEREO( const uint64_t p_uFrame,
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
}

const double CMatcherEpipolar::_getCurveEssentialX( const Eigen::Vector3d& p_vecCoefficients, const double& p_dY ) const
{
    return -( p_vecCoefficients(2)+p_vecCoefficients(1)*p_dY )/p_vecCoefficients(0);
}
const double CMatcherEpipolar::_getCurveEssentialY( const Eigen::Vector3d& p_vecCoefficients, const double& p_dX ) const
{
    return -( p_vecCoefficients(2)+p_vecCoefficients(0)*p_dX )/p_vecCoefficients(1);
}
const int32_t CMatcherEpipolar::_getCurveEssentialU( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uV ) const
{
    return m_pCameraLEFT->getU( _getCurveEssentialX( p_vecCoefficients, m_pCameraLEFT->getNormalizedY( p_uV ) ) );
}
const int32_t CMatcherEpipolar::_getCurveEssentialV( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uU ) const
{
    return m_pCameraLEFT->getV( _getCurveEssentialY( p_vecCoefficients, m_pCameraLEFT->getNormalizedX( p_uU ) ) );
}
const int32_t CMatcherEpipolar::_getCurveEssentialU( const Eigen::Vector3d& p_vecCoefficients, const double& p_uV ) const
{
    return m_pCameraLEFT->getU( _getCurveEssentialX( p_vecCoefficients, m_pCameraLEFT->getNormalizedY( p_uV ) ) );
}
const int32_t CMatcherEpipolar::_getCurveEssentialV( const Eigen::Vector3d& p_vecCoefficients, const double& p_uU ) const
{
    return m_pCameraLEFT->getV( _getCurveEssentialY( p_vecCoefficients, m_pCameraLEFT->getNormalizedX( p_uU ) ) );
}

const double CMatcherEpipolar::_getCurveFundamentalU( const Eigen::Vector3d& p_vecCoefficients, const double& p_dV ) const
{
    return -( p_vecCoefficients(1)*p_dV+p_vecCoefficients(2) )/p_vecCoefficients(0);
}
const double CMatcherEpipolar::_getCurveFundamentalV( const Eigen::Vector3d& p_vecCoefficients, const double& p_dU ) const
{
    return -( p_vecCoefficients(0)*p_dU+p_vecCoefficients(2) )/p_vecCoefficients(1);
}
