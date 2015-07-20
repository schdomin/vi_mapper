#include "CMatcherEpipolar.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CMatcherEpipolar::CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                                    const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle,
                                    const double& p_dMinimumDepthMeters,
                                    const double& p_dMaximumDepthMeters,
                                    const float& p_fMatchingDistanceCutoff,
                                    const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark ): m_pTriangulator( p_pTriangulator ),
                                                                              m_pCameraLEFT( m_pTriangulator->m_pCameraSTEREO->m_pCameraLEFT ),
                                                                              m_pCameraRIGHT( m_pTriangulator->m_pCameraSTEREO->m_pCameraRIGHT ),
                                                                              m_pCameraSTEREO( m_pTriangulator->m_pCameraSTEREO ),
                                                                              m_pDetector( p_pDetectorSingle ),
                                                                              m_pExtractor( m_pTriangulator->m_pExtractor ),
                                                                              m_pMatcher( m_pTriangulator->m_pMatcher ),
                                                                              m_dMinimumDepthMeters( p_dMinimumDepthMeters ),
                                                                              m_dMaximumDepthMeters( p_dMaximumDepthMeters ),
                                                                              m_dMatchingDistanceCutoff( p_fMatchingDistanceCutoff ),
                                                                              m_dMatchingDistanceCutoffOriginal( 2*p_fMatchingDistanceCutoff ),
                                                                              m_uAvailableKeyFrameID( 0 ),
                                                                              m_iSearchUMin( 5 ),
                                                                              m_iSearchUMax( m_pCameraLEFT->m_uWidthPixel-5 ),
                                                                              m_iSearchVMin( 5 ),
                                                                              m_iSearchVMax( m_pCameraLEFT->m_uHeightPixel-5 ),
                                                                              m_cSearchROI( cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ) ),
                                                                              m_cSearchROIPoseOptimization( cv::Point2i( 20, 20 ), cv::Point2i(  m_pCameraLEFT->m_uWidthPixel-20, m_pCameraLEFT->m_uHeightPixel-20 ) ),
                                                                              m_uMaximumFailedSubsequentTrackingsPerLandmark( p_uMaximumFailedSubsequentTrackingsPerLandmark ),
                                                                              m_uRecursionLimitEpipolarLines( 13 ),
                                                                              m_uMinimumPointsForPoseOptimization( 4 ),
                                                                              m_cSolverPoseProjection( m_pCameraLEFT->m_matProjection ),
                                                                              m_cSolverPoseSTEREO( m_pCameraSTEREO ),
                                                                              m_pFileOdometryError( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/error_odometry.txt", "w" ) ),
                                                                              m_pFileEpipolarDetection( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/epipolar_detection.txt", "w" ) )
{
    //ds dump file format
    std::fprintf( m_pFileOdometryError, "ID_FRAME | ITERATION | TOTAL_POINTS INLIERS REPROJECTIONS |   ERROR\n" );
    std::fprintf( m_pFileEpipolarDetection, "FRAME MEASUREMENT_POINT LANDMARKS_TOTAL LANDMARKS_ACTIVE LANDMARKS_VISIBLE\n" );

    CLogger::openBox( );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) minimum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) maximum depth cutoff: %f\n", m_dMinimumDepthMeters );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) matching distance cutoff: %f\n", m_dMatchingDistanceCutoff );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) maximum number of non-detections before dropping landmark: %u\n", m_uMaximumFailedSubsequentTrackingsPerLandmark );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) instance allocated\n" );
    CLogger::closeBox( );
}

CMatcherEpipolar::~CMatcherEpipolar( )
{
    std::fclose( m_pFileOdometryError );
    std::fclose( m_pFileEpipolarDetection );
    std::printf( "<CMatcherEpipolar>(~CMatcherEpipolar) instance deallocated\n" );
}

const Eigen::Isometry3d CMatcherEpipolar::getPoseOptimizedLEFT( const uint64_t p_uFrame,
                                                            cv::Mat& p_matDisplayLEFT,
                                                            const cv::Mat& p_matImageLEFT,
                                                            const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT )
{
    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePoints;

    const double& dSearchHalfHeight( m_cSearchROIPoseOptimization.x );
    const double& dSearchHalfWidth( m_cSearchROIPoseOptimization.y );

    //ds active measurements
    for( const CKeyFrame cKeyFrame: m_vecKeyFramesActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cKeyFrame.vecLandmarks )
        {
            //ds world position
            const CPoint3DInWorldFrame vecPointXYZ( pLandmarkReference->vecPointXYZCalibrated );

            //ds compute current reprojection point
            const cv::Point2d ptUVLEFTEstimate( m_pCameraLEFT->getProjection( static_cast< CPoint3DInCameraFrame >( p_matTransformationEstimateWORLDtoLEFT*vecPointXYZ ) ) );

            //ds check if in visible range
            if( m_cSearchROIPoseOptimization.contains( ptUVLEFTEstimate ) )
            {
                //ds search rectangle
                const cv::Point2d ptUpperLeft( ptUVLEFTEstimate.x-dSearchHalfWidth, ptUVLEFTEstimate.y-dSearchHalfHeight );
                const cv::Point2d ptLowerRight( ptUVLEFTEstimate.x+dSearchHalfWidth, ptUVLEFTEstimate.y+dSearchHalfHeight );
                const cv::Rect cSearchROI( ptUpperLeft, ptLowerRight );
                cv::rectangle( p_matDisplayLEFT, cSearchROI, CColorCodeBGR( 255, 0, 0 ) );

                //ds detect features in this area
                std::vector< cv::KeyPoint > vecKeyPoints;
                m_pDetector->detect( p_matImageLEFT( cSearchROI ), vecKeyPoints );

                //ds if the landmark has been detected
                if( 0 < vecKeyPoints.size( ) )
                {
                    //ds adjust keypoint offsets
                    for( cv::KeyPoint& cKeyPoint: vecKeyPoints )
                    {
                        cKeyPoint.pt.x += ptUpperLeft.x;
                        cKeyPoint.pt.y += ptUpperLeft.y;
                    }

                    //ds check descriptor matches
                    cv::Mat matDescriptors;
                    m_pExtractor->compute( p_matImageLEFT, vecKeyPoints, matDescriptors );
                    std::vector< cv::DMatch > vecMatches;
                    m_pMatcher->match( pLandmarkReference->matDescriptorLastLEFT, matDescriptors, vecMatches );

                    //ds if we got a match and the matching distance is within the range
                    if( 0 < vecMatches.size( ) )
                    {
                        if( m_dMatchingDistanceCutoffOriginal > vecMatches[0].distance )
                        {
                            const cv::Point2f ptBestMatch( vecKeyPoints[vecMatches[0].trainIdx].pt );
                            cv::circle( p_matDisplayLEFT, ptBestMatch, 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                            //ds store values for optimization
                            vecLandmarksWORLD.push_back( vecPointXYZ );
                            vecImagePoints.push_back( CPoint2DInCameraFrame( ptBestMatch.x, ptBestMatch.y ) );
                        }
                        else
                        {
                            std::printf( "<CMatcherEpipolar>(getPoseOptimized) landmark [%06lu] matching failed (matching distance: %f) \n", pLandmarkReference->uID, vecMatches[0].distance );
                        }
                    }
                    else
                    {
                        std::printf( "<CMatcherEpipolar>(getPoseOptimized) landmark [%06lu] matching failed (no match found) \n", pLandmarkReference->uID );
                    }
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimized) landmark [%06lu] no feature detected\n", pLandmarkReference->uID );
                }
            }
            else
            {
                std::printf( "<CMatcherEpipolar>(getPoseOptimized) landmark [%06lu] out of visible range: (%6.2f %6.2f)\n", pLandmarkReference->uID, ptUVLEFTEstimate.x, ptUVLEFTEstimate.y );
            }
        }
    }

    //ds check if we have a sufficient number of points to optimize
    if( m_uMinimumPointsForPoseOptimization < vecLandmarksWORLD.size( ) )
    {
        std::printf( "<CMatcherEpipolar>(getPoseOptimized) optimizing pose with %lu points\n", vecLandmarksWORLD.size( ) );

        //ds feed the solver with the 3D points (in camera frame)
        m_cSolverPoseProjection.model_points = vecLandmarksWORLD;

        //ds feed the solver with the 2D points
        m_cSolverPoseProjection.image_points = vecImagePoints;

        //ds initial guess of the transformation
        m_cSolverPoseProjection.T = p_matTransformationEstimateWORLDtoLEFT;

        //ds initialize solver
        m_cSolverPoseProjection.init( );

        //ds convergence
        double dConvergenceDelta( 1e-5 );
        double dErrorPrevious( 0.0 );

        //ds run KLS
        const uint8_t uIterations = 10;
        for( uint8_t uIteration = 0; uIteration < uIterations; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseProjection.oneRound( );
            uint32_t uInliersCurrent             = m_cSolverPoseProjection.num_inliers;
            uint32_t uGoodReprojections          = m_cSolverPoseProjection.num_reprojected_points;

            //ds log the error evolution
            std::fprintf( m_pFileOdometryError, "    %04lu |         %01u |          %03lu     %03u           %03u | %7.2f\n",
                                                 p_uFrame,
                                                 uIteration,
                                                 vecLandmarksWORLD.size( ),
                                                 uInliersCurrent,
                                                 uGoodReprojections,
                                                 dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( dConvergenceDelta > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumPointsForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrent = m_cSolverPoseProjection.oneRoundInliersOnly( );
                    uint32_t uInliersCurrent             = m_cSolverPoseProjection.num_inliers;
                    uint32_t uGoodReprojections          = m_cSolverPoseProjection.num_reprojected_points;

                    //ds log the error evolution
                    std::fprintf( m_pFileOdometryError, "    %04lu |    INLIER |          %03lu     %03u           %03u | %7.2f\n",
                                                         p_uFrame,
                                                         vecLandmarksWORLD.size( ),
                                                         uInliersCurrent,
                                                         uGoodReprojections,
                                                         dErrorSolverPoseCurrent );
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimized) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        return m_cSolverPoseProjection.T;
    }
    else
    {
        std::printf( "<CMatcherEpipolar>(getPoseOptimized) unable to optimize pose (%lu points)\n", vecLandmarksWORLD.size( ) );
        return p_matTransformationEstimateWORLDtoLEFT;
    }
}

const Eigen::Isometry3d CMatcherEpipolar::getPoseOptimizedSTEREO( const uint64_t p_uFrame,
                                                                  cv::Mat& p_matDisplayLEFT,
                                                                  cv::Mat& p_matDisplayRIGHT,
                                                                  const cv::Mat& p_matImageLEFT,
                                                                  const cv::Mat& p_matImageRIGHT,
                                                                  const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT )
{
    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePointsLEFT;
    gtools::Vector2dVector vecImagePointsRIGHT;

    const double& dSearchHalfHeight( m_cSearchROIPoseOptimization.x );
    const double& dSearchHalfWidth( m_cSearchROIPoseOptimization.y );

    //ds active measurements
    for( const CKeyFrame cKeyFrame: m_vecKeyFramesActive )
    {
        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cKeyFrame.vecLandmarks )
        {
            //ds world position
            const CPoint3DInWorldFrame vecPointXYZ( pLandmarkReference->vecPointXYZCalibrated );
            const CPoint3DInCameraFrame vecPointCAMERA( p_matTransformationEstimateWORLDtoLEFT*vecPointXYZ );

            //ds compute current reprojection point
            const cv::Point2d ptUVLEFTEstimate( m_pCameraLEFT->getProjection( vecPointCAMERA ) );
            const cv::Point2d ptUVRIGHTEstimate( m_pCameraRIGHT->getProjection( vecPointCAMERA ) );

            //ds check if in visible range
            if( m_cSearchROIPoseOptimization.contains( ptUVLEFTEstimate ) && m_cSearchROIPoseOptimization.contains( ptUVRIGHTEstimate ) )
            {
                //ds search rectangles
                const cv::Point2d ptUpperLeftLEFT( ptUVLEFTEstimate.x-dSearchHalfWidth, ptUVLEFTEstimate.y-dSearchHalfHeight );
                const cv::Point2d ptLowerRightLEFT( ptUVLEFTEstimate.x+dSearchHalfWidth, ptUVLEFTEstimate.y+dSearchHalfHeight );
                const cv::Rect cSearchROILEFT( ptUpperLeftLEFT, ptLowerRightLEFT );
                const cv::Point2d ptUpperLeftRIGHT( ptUVRIGHTEstimate.x-dSearchHalfWidth, ptUVRIGHTEstimate.y-dSearchHalfHeight );
                const cv::Point2d ptLowerRightRIGHT( ptUVRIGHTEstimate.x+dSearchHalfWidth, ptUVRIGHTEstimate.y+dSearchHalfHeight );
                const cv::Rect cSearchROIRIGHT( ptUpperLeftRIGHT, ptLowerRightRIGHT );


                cv::rectangle( p_matDisplayLEFT, cSearchROILEFT, CColorCodeBGR( 255, 0, 0 ) );
                cv::rectangle( p_matDisplayRIGHT, cSearchROIRIGHT, CColorCodeBGR( 255, 0, 0 ) );

                //ds detect features in this area
                std::vector< cv::KeyPoint > vecKeyPointsLEFT;
                m_pDetector->detect( p_matImageLEFT( cSearchROILEFT ), vecKeyPointsLEFT );
                std::vector< cv::KeyPoint > vecKeyPointsRIGHT;
                m_pDetector->detect( p_matImageRIGHT( cSearchROIRIGHT ), vecKeyPointsRIGHT );

                //ds if the landmark has been detected
                if( 0 < vecKeyPointsLEFT.size( ) && 0 < vecKeyPointsRIGHT.size( ) )
                {
                    //ds adjust keypoint offsets
                    for( cv::KeyPoint& cKeyPoint: vecKeyPointsLEFT )
                    {
                        cKeyPoint.pt.x += ptUpperLeftLEFT.x;
                        cKeyPoint.pt.y += ptUpperLeftLEFT.y;
                    }
                    for( cv::KeyPoint& cKeyPoint: vecKeyPointsRIGHT )
                    {
                        cKeyPoint.pt.x += ptUpperLeftRIGHT.x;
                        cKeyPoint.pt.y += ptUpperLeftRIGHT.y;
                    }

                    //ds check descriptor matches
                    cv::Mat matDescriptorsLEFT;
                    m_pExtractor->compute( p_matImageLEFT, vecKeyPointsLEFT, matDescriptorsLEFT );
                    std::vector< cv::DMatch > vecMatchesLEFT;
                    m_pMatcher->match( pLandmarkReference->matDescriptorLastLEFT, matDescriptorsLEFT, vecMatchesLEFT );
                    cv::Mat matDescriptorsRIGHT;
                    m_pExtractor->compute( p_matImageRIGHT, vecKeyPointsRIGHT, matDescriptorsRIGHT );
                    std::vector< cv::DMatch > vecMatchesRIGHT;
                    m_pMatcher->match( pLandmarkReference->matDescriptorLastRIGHT, matDescriptorsRIGHT, vecMatchesRIGHT );

                    //ds if we got a match and the matching distance is within the range
                    if( 0 < vecMatchesLEFT.size( ) && 0 < vecMatchesRIGHT.size( ) )
                    {
                        if( m_dMatchingDistanceCutoffOriginal > vecMatchesLEFT[0].distance && m_dMatchingDistanceCutoffOriginal > vecMatchesRIGHT[0].distance )
                        {
                            const cv::Point2f ptBestMatchLEFT( vecKeyPointsLEFT[vecMatchesLEFT[0].trainIdx].pt );
                            cv::circle( p_matDisplayLEFT, ptBestMatchLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                            const cv::Point2f ptBestMatchRIGHT( vecKeyPointsRIGHT[vecMatchesRIGHT[0].trainIdx].pt );
                            cv::circle( p_matDisplayRIGHT, ptBestMatchRIGHT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                            //ds store values for optimization
                            vecLandmarksWORLD.push_back( vecPointXYZ );
                            vecImagePointsLEFT.push_back( CPoint2DInCameraFrame( ptBestMatchLEFT.x, ptBestMatchLEFT.y ) );
                            vecImagePointsRIGHT.push_back( CPoint2DInCameraFrame( ptBestMatchRIGHT.x, ptBestMatchRIGHT.y ) );
                        }
                        else
                        {
                            std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] matching failed (matching distance LEFT: %6.2f RIGHT: %6.2f) \n", pLandmarkReference->uID, vecMatchesLEFT[0].distance, vecMatchesRIGHT[0].distance );
                        }
                    }
                    else
                    {
                        std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] matching failed (no match found) \n", pLandmarkReference->uID );
                    }
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] no feature detected\n", pLandmarkReference->uID );
                }
            }
            else
            {
                std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) landmark [%06lu] out of visible range: LEFT(%6.2f %6.2f) RIGHT(%6.2f %6.2f)\n", pLandmarkReference->uID, ptUVLEFTEstimate.x, ptUVLEFTEstimate.y, ptUVRIGHTEstimate.x, ptUVRIGHTEstimate.y );
            }
        }
    }

    //ds check if we have a sufficient number of points to optimize
    if( m_uMinimumPointsForPoseOptimization < vecLandmarksWORLD.size( ) )
    {
        std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) optimizing pose with %lu points\n", vecLandmarksWORLD.size( ) );

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
        double dConvergenceDelta( 1e-5 );
        double dErrorPrevious( 0.0 );

        //ds run KLS
        const uint8_t uIterations = 10;
        for( uint8_t uIteration = 0; uIteration < uIterations; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseSTEREO.oneRound( );
            uint32_t uInliersCurrent             = m_cSolverPoseSTEREO.uNumberOfInliers;
            uint32_t uGoodReprojections          = m_cSolverPoseSTEREO.uNumberOfReprojections;

            //ds log the error evolution
            std::fprintf( m_pFileOdometryError, "    %04lu |         %01u |          %03lu     %03u           %03u | %7.2f\n",
                                                 p_uFrame,
                                                 uIteration,
                                                 vecLandmarksWORLD.size( ),
                                                 uInliersCurrent,
                                                 uGoodReprojections,
                                                 dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( dConvergenceDelta > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumPointsForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrent = m_cSolverPoseSTEREO.oneRoundInliersOnly( );
                    uint32_t uInliersCurrent             = m_cSolverPoseSTEREO.uNumberOfInliers;
                    uint32_t uGoodReprojections          = m_cSolverPoseSTEREO.uNumberOfReprojections;

                    //ds log the error evolution
                    std::fprintf( m_pFileOdometryError, "    %04lu |    INLIER |          %03lu     %03u           %03u | %7.2f\n",
                                                         p_uFrame,
                                                         vecLandmarksWORLD.size( ),
                                                         uInliersCurrent,
                                                         uGoodReprojections,
                                                         dErrorSolverPoseCurrent );
                }
                else
                {
                    std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) unable to run loop on inliers only (%u not sufficient)\n", uInliersCurrent );
                }
                break;
            }

            //ds save error
            dErrorPrevious = dErrorSolverPoseCurrent;
        }

        return m_cSolverPoseSTEREO.T;
    }
    else
    {
        std::printf( "<CMatcherEpipolar>(getPoseOptimizedSTEREO) unable to optimize pose (%lu points)\n", vecLandmarksWORLD.size( ) );
        return p_matTransformationEstimateWORLDtoLEFT;
    }
}

/*const std::shared_ptr< std::vector< CLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                                               const Eigen::Isometry3d& p_matCurrentTransformation,
                                                                                                               cv::Mat& p_matImage,
                                                                                                               const int32_t& p_iHalfLineLengthBase,
                                                                                                               std::shared_ptr< std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > > >& p_vecDetectionPoints ) const
{
    //ds detected landmarks at this position - VISIBLE (detected in the image) != ACTIVE (viable landmark)
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarksTotal( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds active landmarks after detection
    std::shared_ptr< std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > > > vecActiveLandmarksNew( std::make_shared< std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > > >( ) );

    //ds precompute inverse;
    const Eigen::Isometry3d matCurrentTransformationInverse( p_matCurrentTransformation.inverse( ) );

    //ds check all scan points we have
    for( std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > cDetectionPoint: *p_vecDetectionPoints )
    {
        //ds landmarks redetected in the current detection selection
        std::vector< CLandmark* > vecActiveLandmarksNewPerDetection;

        //ds compute essential matrix
        const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssentialPrecomputed( cDetectionPoint.first, matCurrentTransformationInverse ) );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: cDetectionPoint.second )
        {
            //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
            const Eigen::Vector3d vecCoefficients( matEssential*pLandmarkReference->vecPositionUVReference );

            //ds compute maximum and minimum points (from top to bottom line)
            const int32_t iULastDetection( pLandmarkReference->ptPositionUVLast.x );

            assert( 0 <= iULastDetection );

            //ds compute distance to principal point
            const double dPrincipalDistance( 0 ); // ( m_pCamera->m_vecPrincipalPointNormalized-pLandmarkReference->vecPositionUVLast ).norm( ) );

            //ds compute sampling line
            const int32_t iHalfLineLength( ( 1+0.25*pLandmarkReference->uFailedSubsequentTrackings )*dPrincipalDistance*p_iHalfLineLengthBase );

            assert( 0 <= iULastDetection+iHalfLineLength );

            //ds get back to pixel coordinates
            int32_t iUMinimum( std::max( iULastDetection-iHalfLineLength, m_iSearchUMin ) );
            int32_t iUMaximum( std::min( iULastDetection+iHalfLineLength, m_iSearchUMax ) );
            const int32_t iVCenter( _getCurveV( vecCoefficients, static_cast< double >( iUMaximum+iUMinimum )/2 ) );

            //ds check if the line is completely out of the visible region
            if( ( m_iSearchVMin > iVCenter || m_iSearchVMax < iVCenter ) )
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (epipolar line drifted)\n" );
                continue;
            }

            //ds compute v border values
            int32_t iVForUMinimum( _getCurveV( vecCoefficients, iUMinimum ) );
            int32_t iVForUMaximum( _getCurveV( vecCoefficients, iUMaximum ) );
            const int32_t iVLimitMaximum( std::min( iVCenter+iHalfLineLength, m_iSearchVMax ) );
            const int32_t iVLimitMinimum( std::max( iVCenter-iHalfLineLength, m_iSearchVMin ) );

            //ds negative slope (max v is also at max u)
            if( iVForUMaximum > iVForUMinimum )
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMinimum > iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMinimum;
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }
                if( iVLimitMaximum < iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMaximum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMaximum;
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
                if( iVLimitMinimum > iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMinimum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }

                //ds search space below sampling resoluting (set manually)
                if( iUMaximum < iUMinimum )
                {
                    iUMinimum = iUMaximum-1;
                }

                //ds swap required for uniform looping
                std::swap( iVForUMinimum, iVForUMaximum );
            }

            assert( 0 <= iUMinimum && m_pCamera->m_iWidthPixel >= iUMaximum );
            assert( 0 <= iVForUMinimum && m_pCamera->m_iHeightPixel >= iVForUMaximum );
            assert( 0 <= iUMaximum-iUMinimum );
            assert( 0 <= iVForUMaximum-iVForUMinimum );

            //ds compute pixel ranges to sample
            const uint32_t uDeltaU( iUMaximum-iUMinimum );
            const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

            //ds escape for single points
            if( 0 == uDeltaU && 0 == uDeltaV )
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (zero length)\n" );
                continue;
            }

            //ds sample the larger range
            if( uDeltaU > uDeltaV )
            {
                try
                {
                    //ds get the match over U
                    const cv::Point2f ptMatch( _getMatchSampleU( p_matDisplay, p_matImage, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorReference, pLandmarkReference->cKeyPoint.size ).first );

                    //ds add to references
                    //pLandmarkReference->vecPositionUVLast = m_pCamera->getHomogenized( ptMatch );
                    pLandmarkReference->ptPositionUVLast  = ptMatch;
                    vecActiveLandmarksNewPerDetection.push_back( pLandmarkReference );
                    vecVisibleLandmarksTotal->push_back( pLandmarkReference );
                    pLandmarkReference->uFailedSubsequentTrackings = 0;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet (not DETECTED now but still ACTIVE)
                    if( pLandmarkReference->uFailedSubsequentTrackings < m_uMaximumFailedSubsequentTrackingsPerLandmark )
                    {
                        //ds reset reference point and mark non-match
                        ++pLandmarkReference->uFailedSubsequentTrackings;
                        vecActiveLandmarksNewPerDetection.push_back( pLandmarkReference );
                    }
                    else
                    {
                        std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
                    }
                }
            }
            else
            {
                try
                {
                    //ds get the match over V
                    const cv::Point2f ptMatch( _getMatchSampleV( p_matDisplay, p_matImage, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorReference, pLandmarkReference->cKeyPoint.size ).first );

                    //ds add to references
                    //pLandmarkReference->vecPositionUVLast = m_pCamera->getHomogenized( ptMatch );
                    pLandmarkReference->ptPositionUVLast  = ptMatch;
                    vecActiveLandmarksNewPerDetection.push_back( pLandmarkReference );
                    vecVisibleLandmarksTotal->push_back( pLandmarkReference );
                    pLandmarkReference->uFailedSubsequentTrackings = 0;
                }
                catch( const CExceptionNoMatchFound& p_eException )
                {
                    //ds check if we dont have to drop the landmark yet (not DETECTED now but still ACTIVE)
                    if( pLandmarkReference->uFailedSubsequentTrackings < m_uMaximumFailedSubsequentTrackingsPerLandmark )
                    {
                        //ds reset reference point and mark non-match
                        ++pLandmarkReference->uFailedSubsequentTrackings;
                        vecActiveLandmarksNewPerDetection.push_back( pLandmarkReference );
                    }
                    else
                    {
                        std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
                    }
                }
            }
        }

        //ds check if we still have active landmarks
        if( !vecActiveLandmarksNewPerDetection.empty( ) )
        {
            //ds update scan point vector
            vecActiveLandmarksNew->push_back( std::pair< Eigen::Isometry3d, std::vector< CLandmark* > >( cDetectionPoint.first, vecActiveLandmarksNewPerDetection ) );
        }
        else
        {
            //ds erase the scan point
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point\n" );
        }
    }

    //ds update scan points handle
    p_vecDetectionPoints.reset( );
    p_vecDetectionPoints = vecActiveLandmarksNew;

    //ds return active landmarks
    return vecVisibleLandmarksTotal;
}

const std::shared_ptr< std::vector< CLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                                   const Eigen::Isometry3d& p_matCurrentTransformation,
                                                                                                   cv::Mat& p_matImage,
                                                                                                   const int32_t& p_iHalfLineLengthBase,
                                                                                                   const Eigen::Isometry3d p_matTransformationOnDetection,
                                                                                                   const std::vector< CLandmark* >& p_vecLandmarks ) const
{
    //ds detected landmarks at this position - VISIBLE (detected in the image) != ACTIVE (viable landmark)
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds precompute inverse;
    const Eigen::Isometry3d matCurrentTransformationInverse( p_matCurrentTransformation.inverse( ) );

    //ds compute essential matrix
    const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssentialPrecomputed( p_matTransformationOnDetection, matCurrentTransformationInverse ) );

    //ds loop over the points for the current scan
    for( CLandmark* pLandmarkReference: p_vecLandmarks )
    {
        //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
        const Eigen::Vector3d vecCoefficients( matEssential*pLandmarkReference->vecPositionUVReference );

        //ds compute maximum and minimum points (from top to bottom line)
        const int32_t iULastDetection( pLandmarkReference->ptPositionUVLast.x );

        assert( 0 <= iULastDetection );

        //ds compute distance to principal point
        const double dPrincipalDistance( 0 ); //( ( m_pCamera->m_vecPrincipalPointNormalized-pLandmarkReference->vecPositionUVLast ).norm( ) );

        //ds compute sampling line
        const int32_t iHalfLineLength( ( 1+0.25*pLandmarkReference->uFailedSubsequentTrackings )*dPrincipalDistance*p_iHalfLineLengthBase );

        assert( 0 <= iULastDetection+iHalfLineLength );

        //ds get back to pixel coordinates
        int32_t iUMinimum( std::max( iULastDetection-iHalfLineLength, m_iSearchUMin ) );
        int32_t iUMaximum( std::min( iULastDetection+iHalfLineLength, m_iSearchUMax ) );
        const int32_t iVCenter( _getCurveV( vecCoefficients, static_cast< double >( iUMaximum+iUMinimum )/2 ) );

        //ds check if the line is completely out of the visible region
        if( ( m_iSearchVMin > iVCenter || m_iSearchVMax < iVCenter ) )
        {
            //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (epipolar line drifted)\n" );
            continue;
        }

        //ds compute v border values
        int32_t iVForUMinimum( _getCurveV( vecCoefficients, iUMinimum ) );
        int32_t iVForUMaximum( _getCurveV( vecCoefficients, iUMaximum ) );
        const int32_t iVLimitMaximum( std::min( iVCenter+iHalfLineLength, m_iSearchVMax ) );
        const int32_t iVLimitMinimum( std::max( iVCenter-iHalfLineLength, m_iSearchVMin ) );

        //ds negative slope (max v is also at max u)
        if( iVForUMaximum > iVForUMinimum )
        {
            //ds adjust ROI (recompute U)
            if( iVLimitMinimum > iVForUMinimum )
            {
                iVForUMinimum = iVLimitMinimum;
                iUMinimum     = _getCurveU( vecCoefficients, iVLimitMinimum );
            }
            if( iVLimitMaximum < iVForUMaximum )
            {
                iVForUMaximum = iVLimitMaximum;
                iUMaximum     = _getCurveU( vecCoefficients, iVLimitMaximum );
            }
        }

        //ds positive slope (max v is at min u)
        else
        {
            //ds adjust ROI (recompute U)
            if( iVLimitMaximum < iVForUMinimum )
            {
                iVForUMinimum = iVLimitMaximum;
                iUMinimum     = _getCurveU( vecCoefficients, iVLimitMaximum );
            }
            if( iVLimitMinimum > iVForUMaximum )
            {
                iVForUMaximum = iVLimitMinimum;
                iUMaximum     = _getCurveU( vecCoefficients, iVLimitMinimum );
            }

            //ds search space below sampling resoluting (set manually)
            if( iUMaximum < iUMinimum )
            {
                iUMinimum = iUMaximum-1;
            }

            //ds swap required for uniform looping
            std::swap( iVForUMinimum, iVForUMaximum );
        }

        assert( 0 <= iUMinimum && m_pCamera->m_iWidthPixel >= iUMaximum );
        assert( 0 <= iVForUMinimum && m_pCamera->m_iHeightPixel >= iVForUMaximum );
        assert( 0 <= iUMaximum-iUMinimum );
        assert( 0 <= iVForUMaximum-iVForUMinimum );

        //ds compute pixel ranges to sample
        const uint32_t uDeltaU( iUMaximum-iUMinimum );
        const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

        //ds escape for single points
        if( 0 == uDeltaU && 0 == uDeltaV )
        {
            //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (zero length)\n" );
            continue;
        }

        //ds sample the larger range
        if( uDeltaU > uDeltaV )
        {
            try
            {
                //ds get the match over U
                const cv::Point2f ptMatch( _getMatchSampleU( p_matDisplay, p_matImage, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorReference, pLandmarkReference->cKeyPoint.size ).first );

                //ds add to references
                //pLandmarkReference->vecPositionUVLast = m_pCamera->getHomogenized( ptMatch );
                pLandmarkReference->ptPositionUVLast  = ptMatch;
                vecVisibleLandmarks->push_back( pLandmarkReference );
                pLandmarkReference->uFailedSubsequentTrackings = 0;
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
            }
        }
        else
        {
            try
            {
                //ds get the match over V
                const cv::Point2f ptMatch( _getMatchSampleV( p_matDisplay, p_matImage, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorReference, pLandmarkReference->cKeyPoint.size ).first );

                //ds add to references
                //pLandmarkReference->vecPositionUVLast = m_pCamera->getHomogenized( ptMatch );
                pLandmarkReference->ptPositionUVLast  = ptMatch;
                vecVisibleLandmarks->push_back( pLandmarkReference );
                pLandmarkReference->uFailedSubsequentTrackings = 0;
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
            }
        }
    }

    //ds return active landmarks
    return vecVisibleLandmarks;
}*/
/*
const std::shared_ptr< std::vector< CLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                                   const cv::Mat& p_matImage,
                                                                                                   const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                                                                                                   const Eigen::Matrix3d& p_matEssential,
                                                                                                   const int32_t& p_iHalfLineLengthBase ) const
{
    //ds detected landmarks at this position
    std::shared_ptr< std::vector< CLandmark* > > vecVisibleLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds loop over the points for the current scan
    for( CLandmark* pLandmarkReference: *p_vecLandmarks )
    {
        //ds compute the projection of the point (line) in the current frame (working in normalized coordinates)
        const Eigen::Vector3d vecCoefficients( p_matEssential*pLandmarkReference->vecPositionUVReference );

        //ds compute maximum and minimum points (from top to bottom line)
        const int32_t iULastDetection( pLandmarkReference->ptPositionUVLast.x );

        assert( 0 <= iULastDetection );

        //ds compute distance to principal point
        const double dPrincipalDistance( pLandmarkReference->vecPositionUVReference.head( 2 ).norm( ) );

        //ds compute sampling line
        const int32_t iHalfLineLength( ( 1+0.1*pLandmarkReference->uFailedSubsequentTrackings )*dPrincipalDistance*p_iHalfLineLengthBase );

        assert( 0 <= iULastDetection+iHalfLineLength );

        //ds get back to pixel coordinates
        int32_t iUMinimum( std::max( iULastDetection-iHalfLineLength, m_iSearchUMin ) );
        int32_t iUMaximum( std::min( iULastDetection+iHalfLineLength, m_iSearchUMax ) );
        const int32_t iVCenter( _getCurveV( vecCoefficients, static_cast< double >( iUMaximum+iUMinimum )/2 ) );

        //ds check if the line is completely out of the visible region
        if( ( m_iSearchVMin > iVCenter || m_iSearchVMax < iVCenter ) )
        {
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (epipolar line drifted)\n" );
            continue;
        }

        //ds compute v border values
        int32_t iVForUMinimum( _getCurveV( vecCoefficients, iUMinimum ) );
        int32_t iVForUMaximum( _getCurveV( vecCoefficients, iUMaximum ) );
        const int32_t iVLimitMaximum( std::min( iVCenter+iHalfLineLength, m_iSearchVMax ) );
        const int32_t iVLimitMinimum( std::max( iVCenter-iHalfLineLength, m_iSearchVMin ) );

        //ds negative slope (max v is also at max u)
        if( iVForUMaximum > iVForUMinimum )
        {
            //ds adjust ROI (recompute U)
            if( iVLimitMinimum > iVForUMinimum )
            {
                iVForUMinimum = iVLimitMinimum;
                iUMinimum     = _getCurveU( vecCoefficients, iVLimitMinimum );
            }
            if( iVLimitMaximum < iVForUMaximum )
            {
                iVForUMaximum = iVLimitMaximum;
                iUMaximum     = _getCurveU( vecCoefficients, iVLimitMaximum );
            }
        }

        //ds positive slope (max v is at min u)
        else
        {
            //ds adjust ROI (recompute U)
            if( iVLimitMaximum < iVForUMinimum )
            {
                iVForUMinimum = iVLimitMaximum;
                iUMinimum     = _getCurveU( vecCoefficients, iVLimitMaximum );
            }
            if( iVLimitMinimum > iVForUMaximum )
            {
                iVForUMaximum = iVLimitMinimum;
                iUMaximum     = _getCurveU( vecCoefficients, iVLimitMinimum );
            }

            //ds search space below sampling resolution (set manually)
            if( iUMaximum < iUMinimum )
            {
                iUMinimum = iUMaximum-1;
            }

            //ds swap required for uniform looping
            std::swap( iVForUMinimum, iVForUMaximum );
        }

        assert( 0 <= iUMinimum && m_pCamera->m_iWidthPixel >= iUMaximum );
        assert( 0 <= iVForUMinimum && m_pCamera->m_iHeightPixel >= iVForUMaximum );
        assert( 0 <= iUMaximum-iUMinimum );
        assert( 0 <= iVForUMaximum-iVForUMinimum );

        //ds compute pixel ranges to sample
        const uint32_t uDeltaU( iUMaximum-iUMinimum );
        const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

        //ds escape for single points
        if( 0 == uDeltaU && 0 == uDeltaV )
        {
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark out of sight (zero length)\n" );
            continue;
        }

        //ds sample the larger range
        if( uDeltaU > uDeltaV )
        {
            try
            {*/
                /*ds get the match over U
                const std::pair< cv::Point2f, CDescriptor > cMatch( _getMatchSampleRecursiveU( p_matDisplay, p_matImage, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorLast, pLandmarkReference->cKeyPoint.size, 0 ) );

                cv::circle( p_matDisplay, pLandmarkReference->ptPositionUVLast, 2, CColorCodeBGR( 255, 0, 0), -1 );

                //ds update landmark
                pLandmarkReference->ptPositionUVLast  = cMatch.first;
                pLandmarkReference->matDescriptorLast = cMatch.second;
                pLandmarkReference->uFailedSubsequentTrackings = 0;*//*
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                ++pLandmarkReference->uFailedSubsequentTrackings;
            }

            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings )
            {
                vecVisibleLandmarks->push_back( pLandmarkReference );
            }
            else
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
            }
        }
        else
        {
            try
            {*/
                /*ds get the match over V
                const std::pair< cv::Point2f, CDescriptor > cMatch( _getMatchSampleRecursiveV( p_matDisplay, p_matImage, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorLast, pLandmarkReference->cKeyPoint.size, 0 ) );

                cv::circle( p_matDisplay, pLandmarkReference->ptPositionUVLast, 2, CColorCodeBGR( 255, 0, 0), -1 );

                //ds update landmark
                pLandmarkReference->ptPositionUVLast  = cMatch.first;
                pLandmarkReference->matDescriptorLast = cMatch.second;
                pLandmarkReference->uFailedSubsequentTrackings = 0;*//*
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                ++pLandmarkReference->uFailedSubsequentTrackings;
            }

            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings )
            {
                vecVisibleLandmarks->push_back( pLandmarkReference );
            }
            else
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
            }
        }
    }

    //ds return active landmarks
    return vecVisibleLandmarks;
}*/

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( const uint64_t p_uFrame,
                                                                                                                    cv::Mat& p_matDisplayLEFT,
                                                                                                                    cv::Mat& p_matDisplayRIGHT,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
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
    std::vector< CKeyFrame > vecKeyFramesActive;

    //ds rightside keypoints buffer for descriptor computation
    std::vector< cv::KeyPoint > vecKeyPointRIGHT( 1 );

    //ds vectors for pose solver
    gtools::Vector3dVector vecLandmarksWORLD;
    gtools::Vector2dVector vecImagePointsLEFT;
    gtools::Vector2dVector vecImagePointsRIGHT;

    //ds active measurements
    for( const CKeyFrame cKeyFrame: m_vecKeyFramesActive )
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
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DInWorldFrame >( matTransformationWORLDtoLEFT*pLandmarkReference->vecPointXYZCalibrated ) ) );

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
            int32_t iVForUMinimum( _getCurveV( vecCoefficients, iUMinimum ) );
            int32_t iVForUMaximum( _getCurveV( vecCoefficients, iUMaximum ) );

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
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }
                if( iVLimitMaximum < iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMaximum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMaximum;
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
                if( iVLimitMinimum > iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMinimum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }

                //ds swap required for uniform looping
                std::swap( iVForUMinimum, iVForUMaximum );
            }

            //ds escape for invalid points (caused by invalid projections)
            if( iUMinimum > iUMaximum || iVForUMinimum > iVForUMaximum )
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] out of sight (U[%i,%i] V[%i,%i])\n", pLandmarkReference->uID, iUMinimum, iUMaximum, iVForUMinimum, iVForUMaximum );
                continue;
            }

            assert( 0 <= iUMinimum );
            assert( m_pCameraLEFT->m_iWidthPixel >= iUMaximum );
            assert( 0 <= iVForUMinimum );
            assert( m_pCameraLEFT->m_iHeightPixel >= iVForUMaximum );
            //assert( 0 <= iUMaximum-iUMinimum );
            //assert( 0 <= iVForUMaximum-iVForUMinimum );

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
                    pMatchLEFT = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorLastLEFT, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }
                else
                {
                    //ds get the match over V
                    pMatchLEFT = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorLastLEFT, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }

                const cv::Point2d& ptUVLEFT( pMatchLEFT->cKeyPoint.pt );

                assert( m_cSearchROI.contains( ptUVLEFT ) );

                //ds triangulate point
                const CPoint3DInCameraFrame vecPointTriangulatedLEFT( m_pTriangulator->getPointTriangulatedLimited( p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor ) );
                //const CPoint3DInCameraFrame vecPointTriangulatedRIGHT( m_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointTriangulatedLEFT );
                const CPoint3DInWorldFrame vecPointXYZ( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT );

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
                pLandmarkReference->matDescriptorLastLEFT      = pMatchLEFT->matDescriptor; //( pMatchLEFT->matDescriptor + pLandmarkReference->matDescriptorLastLEFT )/2.0;
                pLandmarkReference->matDescriptorLastRIGHT     = matDescriptorRIGHT;
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addPosition( p_uFrame, ptUVLEFT, ptUVRIGHT, vecPointTriangulatedLEFT, vecPointXYZ, vecCameraPosition, matKRotation, vecKTranslation, matProjectionWORLDtoLEFT );

                //ds register measurement
                vecVisibleLandmarksPerKeyFrame.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds store elements for optimization
                vecLandmarksWORLD.push_back( pLandmarkReference->vecPointXYZCalibrated );
                vecImagePointsLEFT.push_back( CWrapperOpenCV::fromCVVector( ptUVLEFT ) );
                vecImagePointsRIGHT.push_back( CWrapperOpenCV::fromCVVector( ptUVRIGHT )  );

                //ds new positions
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                char chBufferMiniInfo[20];
                std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmarkReference->uID, pLandmarkReference->uCalibrations, pLandmarkReference->dCurrentAverageSquaredError );
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
        std::fprintf( m_pFileEpipolarDetection, "%04lu %03lu %02lu %02lu %02lu\n", p_uFrame, cKeyFrame.uID, cKeyFrame.vecLandmarks->size( ), vecActiveLandmarksPerKeyFrame->size( ), vecVisibleLandmarksPerKeyFrame.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerKeyFrame->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecKeyFramesActive.push_back( CKeyFrame( cKeyFrame.uID, cKeyFrame.matTransformationLEFTtoWORLD, vecActiveLandmarksPerKeyFrame ) );

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
        double dConvergenceDelta( 1e-5 );
        double dErrorPrevious( 0.0 );

        //ds run KLS
        const uint8_t uMaxIterations = 10;
        for( uint8_t uIteration = 0; uIteration < uMaxIterations; ++uIteration )
        {
            //ds run optimization
            const double dErrorSolverPoseCurrent = m_cSolverPoseSTEREO.oneRound( );
            const uint32_t uInliersCurrent       = m_cSolverPoseSTEREO.uNumberOfInliers;

            //ds log the error evolution
            std::fprintf( m_pFileOdometryError, "    %04lu |         %01u |          %03lu     %03u           %03u | %7.2f\n",
                                                 p_uFrame,
                                                 uIteration,
                                                 vecLandmarksWORLD.size( ),
                                                 uInliersCurrent,
                                                 m_cSolverPoseSTEREO.uNumberOfReprojections,
                                                 dErrorSolverPoseCurrent );

            //ds check convergence (triggers another last loop)
            if( dConvergenceDelta > std::fabs( dErrorPrevious-dErrorSolverPoseCurrent ) )
            {
                //ds if we have a sufficient number of inliers
                if( m_uMinimumPointsForPoseOptimization < uInliersCurrent )
                {
                    //ds the last round is run with only inliers
                    const double dErrorSolverPoseCurrentIO = m_cSolverPoseSTEREO.oneRoundInliersOnly( );

                    //ds log the error evolution
                    std::fprintf( m_pFileOdometryError, "    %04lu |    INLIER |          %03lu     %03u           %03u | %7.2f\n",
                                                         p_uFrame,
                                                         vecLandmarksWORLD.size( ),
                                                         m_cSolverPoseSTEREO.uNumberOfInliers,
                                                         m_cSolverPoseSTEREO.uNumberOfReprojections,
                                                         dErrorSolverPoseCurrentIO );
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
    m_vecKeyFramesActive.swap( vecKeyFramesActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

const std::shared_ptr< std::vector< const CMeasurementLandmark* > > CMatcherEpipolar::getVisibleLandmarksEssential( const uint64_t p_uFrame,
                                                                                                                    const cv::Mat& p_matImageLEFT,
                                                                                                                    const cv::Mat& p_matImageRIGHT,
                                                                                                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                                    const int32_t& p_iHalfLineLengthBase )
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
    std::vector< CKeyFrame > vecMeasurementPointsActive;

    //ds rightside keypoints buffer for descriptor computation
    std::vector< cv::KeyPoint > vecKeyPointRIGHT( 1 );

    //ds active measurements
    for( const CKeyFrame cMeasurementPoint: m_vecKeyFramesActive )
    {
        //ds visible (=in this image detected) active (=not detected in this image but failed detections below threshold)
        std::vector< const CMeasurementLandmark* > vecVisibleLandmarksPerMeasurementPoint;
        std::shared_ptr< std::vector< CLandmark* > >vecActiveLandmarksPerMeasurementPoint( std::make_shared< std::vector< CLandmark* > >( ) );

        //ds compute essential matrix for this detection point
        const Eigen::Isometry3d matTransformationToNow( matTransformationWORLDtoLEFT*cMeasurementPoint.matTransformationLEFTtoWORLD );
        const Eigen::Matrix3d matSkewTranslation( CMiniVisionToolbox::getSkew( matTransformationToNow.translation( ) ) );
        const Eigen::Matrix3d matEssential( matTransformationToNow.linear( )*matSkewTranslation );

        //ds loop over the points for the current scan
        for( CLandmark* pLandmarkReference: *cMeasurementPoint.vecLandmarks )
        {
            //ds projection from triangulation to estimate epipolar line drawing TODO: remove cast
            const cv::Point2d ptProjection( m_pCameraLEFT->getProjection( static_cast< CPoint3DInWorldFrame >( matTransformationWORLDtoLEFT*pLandmarkReference->vecPointXYZCalibrated ) ) );

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
            int32_t iVForUMinimum( _getCurveV( vecCoefficients, iUMinimum ) );
            int32_t iVForUMaximum( _getCurveV( vecCoefficients, iUMaximum ) );

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
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }
                if( iVLimitMaximum < iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMaximum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
            }

            //ds positive slope (max v is at min u)
            else
            {
                //ds adjust ROI (recompute U)
                if( iVLimitMaximum < iVForUMinimum )
                {
                    iVForUMinimum = iVLimitMaximum;
                    iUMinimum     = _getCurveU( vecCoefficients, iVLimitMaximum );
                }
                if( iVLimitMinimum > iVForUMaximum )
                {
                    iVForUMaximum = iVLimitMinimum;
                    iUMaximum     = _getCurveU( vecCoefficients, iVLimitMinimum );
                }

                //ds swap required for uniform looping
                std::swap( iVForUMinimum, iVForUMaximum );
            }

            //ds escape for invalid points (caused by invalid projections)
            if( iUMinimum > iUMaximum || iVForUMinimum > iVForUMaximum )
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] out of sight (U[%i,%i] V[%i,%i])\n", pLandmarkReference->uID, iUMinimum, iUMaximum, iVForUMinimum, iVForUMaximum );
                continue;
            }

            assert( 0 <= iUMinimum );
            assert( m_pCameraLEFT->m_iWidthPixel >= iUMaximum );
            assert( 0 <= iVForUMinimum );
            assert( m_pCameraLEFT->m_iHeightPixel >= iVForUMaximum );
            //assert( 0 <= iUMaximum-iUMinimum );
            //assert( 0 <= iVForUMaximum-iVForUMinimum );

            //ds compute pixel ranges to sample
            const uint32_t uDeltaU( iUMaximum-iUMinimum );
            const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

            //ds the match to find
            const CMatchTracking* pMatchLEFT( 0 );

            try
            {
                //ds sample the larger range
                if( uDeltaU > uDeltaV )
                {
                    //ds get the match over U
                    pMatchLEFT = _getMatchSampleRecursiveU( p_matImageLEFT, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorLastLEFT, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }
                else
                {
                    //ds get the match over V
                    pMatchLEFT = _getMatchSampleRecursiveV( p_matImageLEFT, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorLastLEFT, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }

                assert( m_cSearchROI.contains( pMatchLEFT->cKeyPoint.pt ) );

                //ds triangulate point
                const CPoint3DInCameraFrame vecPointTriangulatedLEFT( m_pTriangulator->getPointTriangulatedLimited( p_matImageRIGHT, pMatchLEFT->cKeyPoint, pMatchLEFT->matDescriptor ) );
                //const CPoint3DInCameraFrame vecPointTriangulatedRIGHT( m_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointTriangulatedLEFT );

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
                const double dEpipolarError( ptUVRIGHT.y-pMatchLEFT->cKeyPoint.pt.y );
                if( 0.1 < dEpipolarError )
                {
                    std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] epipolar error: %f\n", pLandmarkReference->uID, dEpipolarError );
                }
                ptUVRIGHT.y = pMatchLEFT->cKeyPoint.pt.y;

                //ds compute reference descriptor on right side as well
                vecKeyPointRIGHT[0] = cv::KeyPoint( ptUVRIGHT, pMatchLEFT->cKeyPoint.size );
                CDescriptor matDescriptorRIGHT;
                m_pExtractor->compute( p_matImageRIGHT, vecKeyPointRIGHT, matDescriptorRIGHT );

                //ds update landmark
                pLandmarkReference->matDescriptorLastLEFT      = pMatchLEFT->matDescriptor;
                pLandmarkReference->matDescriptorLastRIGHT     = matDescriptorRIGHT;
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addPosition( p_uFrame, pMatchLEFT->cKeyPoint.pt, ptUVRIGHT, vecPointTriangulatedLEFT, static_cast< CPoint3DInWorldFrame >( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT ), vecCameraPosition, matKRotation, vecKTranslation, matProjectionWORLDtoLEFT );

                //ds register measurement
                vecVisibleLandmarksPerMeasurementPoint.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds free handle
                delete pMatchLEFT;
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                //ds draw last position
                //cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
                ++pLandmarkReference->uFailedSubsequentTrackings;
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] caught exception: %s\n", pLandmarkReference->uID, p_eException.what( ) );
            }

            //ds check activity
            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings )
            {
                vecActiveLandmarksPerMeasurementPoint->push_back( pLandmarkReference );
            }
            else
            {
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%06lu] dropped\n", pLandmarkReference->uID );
            }
        }

        //ds log
        std::fprintf( m_pFileEpipolarDetection, "%04lu %03lu %02lu %02lu %02lu\n", p_uFrame, cMeasurementPoint.uID, cMeasurementPoint.vecLandmarks->size( ), vecActiveLandmarksPerMeasurementPoint->size( ), vecVisibleLandmarksPerMeasurementPoint.size( ) );

        //ds check if we can keep the measurement point
        if( !vecActiveLandmarksPerMeasurementPoint->empty( ) )
        {
            //ds register the measurement point and its visible landmarks anew
            vecMeasurementPointsActive.push_back( CKeyFrame( cMeasurementPoint.uID, cMeasurementPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerMeasurementPoint ) );

            //ds combine visible landmarks
            vecVisibleLandmarks->insert( vecVisibleLandmarks->end( ), vecVisibleLandmarksPerMeasurementPoint.begin( ), vecVisibleLandmarksPerMeasurementPoint.end( ) );
        }
        else
        {
            std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) erased detection point\n" );
        }
    }

    //ds info
    //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) visible landmarks: %lu (active measurement points: %lu)\n", vecVisibleLandmarks->size( ), vecMeasurementPointsActive.size( ) );

    //ds update active measurement points
    m_vecKeyFramesActive.swap( vecMeasurementPointsActive );

    //ds return active landmarks
    return vecVisibleLandmarks;
}

/*const std::pair< cv::Point2f, CDescriptor >  CMatcherEpipolar::_getMatchSampleU( cv::Mat& p_matDisplay,
                                                                     const cv::Mat& p_matImage,
                                                                     const int32_t& p_iUMinimum,
                                                                     const int32_t& p_iDeltaU,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const cv::Mat& p_matReferenceDescriptor,
                                                                     const float& p_fKeyPointSize ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaU );

    //ds sample over U
    for( int32_t u = 0; u < p_iDeltaU; ++u )
    {
        //ds compute corresponding V coordinate
        const int32_t uU( p_iUMinimum+u );
        const int32_t uV( _getCurveV( p_vecCoefficients, uU ) );

        //ds add keypoint
        vecPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
        cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
    }
    catch( const CExceptionNoMatchFound& p_cException )
    {
        //ds loop over parallel lines (elliptic)
        const int32_t iEllipticDeltaU( p_iDeltaU/2 );

        //ds return if not possible
        if( 0 == iEllipticDeltaU )
        {
            throw p_cException;
        }

        const int32_t iEllipticUMinimum( p_iUMinimum+p_iDeltaU/4 );
        std::vector< cv::KeyPoint > vecEllipticPoolKeyPoints( iEllipticDeltaU );

        //ds sample shifted points - UPPER ELLIPTIC LINE
        for( int32_t u = 0; u < iEllipticDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const int32_t uU( iEllipticUMinimum+u );
            const int32_t uV( _getCurveV( p_vecCoefficients, uU ) + 1 );

            //ds add keypoint
            vecEllipticPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
        }

        try
        {
            //ds return if we find a match on the UPPER elliptic epipolar line
            return _getMatch( p_matImage, vecEllipticPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            //ds sample shifted points - LOWER ELLIPTIC LINE
            for( int32_t u = 0; u < iEllipticDeltaU; ++u )
            {
                //ds compute corresponding V coordinate
                const int32_t uU( iEllipticUMinimum+u );
                const int32_t uV( _getCurveV( p_vecCoefficients, uU ) - 1 );

                //ds add keypoint
                vecEllipticPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
                cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
            }

            //ds return if we find a match on the LOWER elliptic epipolar line
            return _getMatch( p_matImage, vecEllipticPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
        }
    }
}
const std::pair< cv::Point2f, CDescriptor >  CMatcherEpipolar::_getMatchSampleV( cv::Mat& p_matDisplay,
                                                                     const cv::Mat& p_matImage,
                                                                     const int32_t& p_iVMinimum,
                                                                     const int32_t& p_iDeltaV,
                                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                                     const cv::Mat& p_matReferenceDescriptor,
                                                                     const float& p_fKeyPointSize ) const
{
    //ds allocate keypoint pool
    std::vector< cv::KeyPoint > vecPoolKeyPoints( p_iDeltaV );

    //ds sample over U
    for( int32_t v = 0; v < p_iDeltaV; ++v )
    {
        //ds compute corresponding U coordinate
        const int32_t uV( p_iVMinimum+v );
        const int32_t uU( _getCurveU( p_vecCoefficients, uV ) );

        //ds add keypoint
        vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
        cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
    }

    try
    {
        //ds return if we find a match on the epipolar line
        return _getMatch( p_matImage, vecPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
    }
    catch( const CExceptionNoMatchFound& p_cException )
    {
        //ds loop over parallel lines (elliptic)
        const int32_t iEllipticDeltaV( p_iDeltaV/2 );

        //ds return if not possible
        if( 0 == iEllipticDeltaV )
        {
            throw p_cException;
        }

        const int32_t iEllipticVMinimum( p_iVMinimum+p_iDeltaV/4 );
        std::vector< cv::KeyPoint > vecEllipticPoolKeyPoints( iEllipticDeltaV );

        //ds sample shifted points - UPPER ELLIPTIC LINE
        for( int32_t v = 0; v < iEllipticDeltaV; ++v )
        {
            //ds compute corresponding U coordinate
            const int32_t uV( iEllipticVMinimum+v );
            const int32_t uU( _getCurveU( p_vecCoefficients, uV ) + 1 );

            //ds add keypoint
            vecEllipticPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
        }

        try
        {
            //ds return if we find a match on the UPPER elliptic epipolar line
            return _getMatch( p_matImage, vecEllipticPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            //ds sample shifted points - LOWER ELLIPTIC LINE
            for( int32_t v = 0; v < iEllipticDeltaV; ++v )
            {
                //ds compute corresponding U coordinate
                const int32_t uV( iEllipticVMinimum+v );
                const int32_t uU( _getCurveU( p_vecCoefficients, uV ) - 1 );

                //ds add keypoint
                vecEllipticPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_fKeyPointSize );
                cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
            }

            //ds return if we find a match on the LOWER elliptic epipolar line
            return _getMatch( p_matImage, vecEllipticPoolKeyPoints, p_matReferenceDescriptor, p_matReferenceDescriptor );
        }
    }
}*/

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
            const int32_t iV( _getCurveV( p_vecCoefficients, iU ) + iSamplingOffset );

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
            const int32_t iV( _getCurveV( p_vecCoefficients, iU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[i] = cv::KeyPoint( iU, iV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( iU, iV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
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
            const int32_t uU( _getCurveU( p_vecCoefficients, uV ) + iSamplingOffset );

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
            const int32_t uU( _getCurveU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 219, 112, 147 ), -1 );
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
            return _getMatchSampleRecursiveV( p_matDisplay, p_matImage, p_iVMinimum, p_iDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
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
            const int32_t uV( _getCurveV( p_vecCoefficients, uU ) + iSamplingOffset );

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
            const int32_t uV( _getCurveV( p_vecCoefficients, uU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
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
            const int32_t uU( _getCurveU( p_vecCoefficients, uV ) + iSamplingOffset );

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
            const int32_t uU( _getCurveU( p_vecCoefficients, uV ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[v] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
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
            return _getMatchSampleRecursiveV( p_matImage, p_iVMinimum, p_iDeltaV, p_vecCoefficients, p_matReferenceDescriptor, p_matOriginalDescriptor, p_dKeyPointSize, p_uRecursionDepth+1 );
        }
    }
}

const CMatchTracking* CMatcherEpipolar::_getMatch( const cv::Mat& p_matImage,
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

    if( m_dMatchingDistanceCutoff > dMatchingDistanceToRelative )
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

inline const double CMatcherEpipolar::_getCurveX( const Eigen::Vector3d& p_vecCoefficients, const double& p_dY ) const
{
    return -( p_vecCoefficients(2)+p_vecCoefficients(1)*p_dY )/p_vecCoefficients(0);
}
inline const double CMatcherEpipolar::_getCurveY( const Eigen::Vector3d& p_vecCoefficients, const double& p_dX ) const
{
    return -( p_vecCoefficients(2)+p_vecCoefficients(0)*p_dX )/p_vecCoefficients(1);
}
inline const int32_t CMatcherEpipolar::_getCurveU( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uV ) const
{
    return m_pCameraLEFT->getU( _getCurveX( p_vecCoefficients, m_pCameraLEFT->getNormalizedY( p_uV ) ) );
}
inline const int32_t CMatcherEpipolar::_getCurveV( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uU ) const
{
    return m_pCameraLEFT->getV( _getCurveY( p_vecCoefficients, m_pCameraLEFT->getNormalizedX( p_uU ) ) );
}
inline const int32_t CMatcherEpipolar::_getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_uV ) const
{
    return m_pCameraLEFT->getU( _getCurveX( p_vecCoefficients, m_pCameraLEFT->getNormalizedY( p_uV ) ) );
}
inline const int32_t CMatcherEpipolar::_getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_uU ) const
{
    return m_pCameraLEFT->getV( _getCurveY( p_vecCoefficients, m_pCameraLEFT->getNormalizedX( p_uU ) ) );
}
