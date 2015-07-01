#include "CMatcherEpipolar.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "exceptions/CExceptionNoMatchFoundInternal.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CMatcherEpipolar::CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                                    const float& p_fMatchingDistanceCutoff,
                                    const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark ): m_pTriangulator( p_pTriangulator ),
                                                                              m_pCameraLEFT( m_pTriangulator->m_pCameraSTEREO->m_pCameraLEFT ),
                                                                              m_pCameraRIGHT( m_pTriangulator->m_pCameraSTEREO->m_pCameraRIGHT ),
                                                                              m_pCameraSTEREO( m_pTriangulator->m_pCameraSTEREO ),
                                                                              m_pExtractor( m_pTriangulator->m_pExtractor ),
                                                                              m_pMatcher( m_pTriangulator->m_pMatcher ),
                                                                              m_dMatchingDistanceCutoff( p_fMatchingDistanceCutoff ),
                                                                              m_dMatchingDistanceCutoffOriginal( 2*p_fMatchingDistanceCutoff ),
                                                                              m_uAvailableMeasurementPointID( 0 ),
                                                                              m_iSearchUMin( 5 ),
                                                                              m_iSearchUMax( m_pCameraLEFT->m_uWidthPixel-5 ),
                                                                              m_iSearchVMin( 5 ),
                                                                              m_iSearchVMax( m_pCameraLEFT->m_uHeightPixel-5 ),
                                                                              m_cSearchROI( cv::Point2i( m_iSearchUMin, m_iSearchVMin ), cv::Point2i( m_iSearchUMax, m_iSearchVMax ) ),
                                                                              m_uMaximumFailedSubsequentTrackingsPerLandmark( p_uMaximumFailedSubsequentTrackingsPerLandmark ),
                                                                              m_uRecursionLimitEpipolarLines( 25 ),
                                                                              m_pFileOdometryError( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/error_odometry.txt", "w" ) ),
                                                                              m_pFileEpipolarDetection( std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/epipolar_detection.txt", "w" ) )
{
    //ds dump file format
    std::fprintf( m_pFileOdometryError, "FRAME MEASUREMENT_POINT ITERATION TOTAL_POINTS INLIERS ERROR\n" );
    std::fprintf( m_pFileEpipolarDetection, "FRAME MEASUREMENT_POINT LANDMARKS_TOTAL LANDMARKS_ACTIVE LANDMARKS_VISIBLE\n" );

    CLogger::openBox( );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CMatcherEpipolar>(CMatcherEpipolar) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
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

    //ds new active measurement points
    std::vector< CMeasurementPoint > vecMeasurementPointsActive;

    //ds active measurements
    for( const CMeasurementPoint cMeasurementPoint: m_vecMeasurementPointsActive )
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
                std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] out of sight (U[%i,%i] V[%i,%i])\n", pLandmarkReference->uID, iUMinimum, iUMaximum, iVForUMinimum, iVForUMaximum );
                continue;
            }

            assert( 0 <= iUMinimum && m_pCameraLEFT->m_iWidthPixel >= iUMaximum );
            assert( 0 <= iVForUMinimum && m_pCameraLEFT->m_iHeightPixel >= iVForUMaximum );
            //assert( 0 <= iUMaximum-iUMinimum );
            //assert( 0 <= iVForUMaximum-iVForUMinimum );

            //ds compute pixel ranges to sample
            const uint32_t uDeltaU( iUMaximum-iUMinimum );
            const uint32_t uDeltaV( iVForUMaximum-iVForUMinimum );

            //ds draw last position
            cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

            //ds the match to find
            const CMatchTracking* pMatch( 0 );

            try
            {
                //ds sample the larger range
                if( uDeltaU > uDeltaV )
                {
                    //ds get the match over U
                    pMatch = _getMatchSampleRecursiveU( p_matDisplayLEFT, p_matImageLEFT, iUMinimum, uDeltaU, vecCoefficients, pLandmarkReference->matDescriptorLast, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }
                else
                {
                    //ds get the match over V
                    pMatch = _getMatchSampleRecursiveV( p_matDisplayLEFT, p_matImageLEFT, iVForUMinimum, uDeltaV, vecCoefficients, pLandmarkReference->matDescriptorLast, pLandmarkReference->matDescriptorReference, pLandmarkReference->dKeyPointSize, 0 );
                }

                //ds triangulate point
                const CPoint3DInCameraFrame vecPointTriangulatedLEFT( m_pTriangulator->getPointTriangulatedLimited( p_matImageRIGHT, pMatch->cKeyPoint, pMatch->matDescriptor ) );
                //const CPoint3DInCameraFrame vecPointTriangulatedRIGHT( m_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointTriangulatedLEFT );

                //ds get projection
                const cv::Point2d ptUVRIGHT( m_pCameraRIGHT->getProjection( vecPointTriangulatedLEFT ) );

                //ds update landmark
                pLandmarkReference->matDescriptorLast          = pMatch->matDescriptor;
                pLandmarkReference->uFailedSubsequentTrackings = 0;
                pLandmarkReference->addPosition( p_uFrame, pMatch->cKeyPoint.pt, ptUVRIGHT, vecPointTriangulatedLEFT, static_cast< CPoint3DInWorldFrame >( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT ), vecCameraPosition, matKRotation, vecKTranslation );

                vecVisibleLandmarksPerMeasurementPoint.push_back( pLandmarkReference->getLastMeasurement( ) );

                //ds new positions
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 0, 255, 0 ), -1 );

                char chBufferMiniInfo[20];
                std::snprintf( chBufferMiniInfo, 20, "%lu(%u|%5.2f)", pLandmarkReference->uID, pLandmarkReference->uCalibrations, pLandmarkReference->dCurrentAverageSquaredError );
                cv::putText( p_matDisplayLEFT, chBufferMiniInfo, cv::Point2d( pMatch->cKeyPoint.pt.x+pLandmarkReference->dKeyPointSize, pMatch->cKeyPoint.pt.y+pLandmarkReference->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplayRIGHT, ptUVRIGHT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                cv::circle( p_matDisplayLEFT, ptProjection, pLandmarkReference->dKeyPointSize, CColorCodeBGR( 0, 0, 255 ), 1 );

                //ds free handle
                delete pMatch;
            }
            catch( const CExceptionNoMatchFound& p_eException )
            {
                //ds draw last position
                cv::circle( p_matDisplayLEFT, pLandmarkReference->getLastDetectionLEFT( ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
                ++pLandmarkReference->uFailedSubsequentTrackings;
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark [%lu] caught exception: %s\n", pLandmarkReference->uID, p_eException.what( ) );
            }

            //ds check activity
            if( m_uMaximumFailedSubsequentTrackingsPerLandmark > pLandmarkReference->uFailedSubsequentTrackings && ptProjection.inside( m_cSearchROI ) )
            {
                vecActiveLandmarksPerMeasurementPoint->push_back( pLandmarkReference );
            }
            else
            {
                //std::printf( "<CMatcherEpipolar>(getVisibleLandmarksEssential) landmark dropped\n" );
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

            double dErrorSolverPosePrevious( 0.0 );
            const double dDeltaConvergence( 1e-5 );

            m_cSolverPose.init();

            //ds run LS - SNIPPET
            const uint8_t num_iterations = 10;
            for( uint8_t uIteration = 0; uIteration < num_iterations; ++uIteration )
            {
                //ds UGLY in/out inliers count

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

              /*char filename[1024];
              std::sprintf (filename, "/home/dominik/debug/frame-%04lu_measurement-%02u_iteration-%05d.dat", p_uFrame, uMeasurements, iteration);
              gtools::Vector2dVector currentPoints;
              m_cSolverPose.project(currentPoints);
              std::ofstream os(filename);
              for (size_t j = 0; j<vecImagePoints.size(); j++){
                Eigen::Vector2d ip = vecImagePoints[j];
                Eigen::Vector2d cp = currentPoints[j];
                os << ip.x() << " " << ip.y() << std::endl;
                os << cp.x() << " " << cp.y() << std::endl;
                os << std::endl;
              }*/
            }

            const Eigen::Isometry3d matTransformationLEFTtoWORLDCorrected( m_cSolverPose.T.inverse( ) );
            const Eigen::Vector3d vecTranslationCurrent( matTransformationLEFTtoWORLDCorrected.translation( ) );

            //ds draw position on trajectory mat
            cv::circle( p_matDisplayTrajectory, cv::Point2d( 180+vecTranslationCurrent( 0 )*10, 360-vecTranslationCurrent( 1 )*10 ), 2, CColorCodeBGR( 0, 0, 255 ), -1 );

            //ds register the measurement point and its visible landmarks anew
            vecMeasurementPointsActive.push_back( CMeasurementPoint( cMeasurementPoint.uID, cMeasurementPoint.matTransformationLEFTtoWORLD, vecActiveLandmarksPerMeasurementPoint ) );
            //vecMeasurementPointsActive.push_back( CMeasurementPoint( matTransformationLEFTtoWORLDCorrected, vecActiveLandmarksPerMeasurementPoint ) );

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
    m_vecMeasurementPointsActive.swap( vecMeasurementPointsActive );

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
        for( int32_t u = 0; u < p_iDeltaU; ++u )
        {
            //ds compute corresponding V coordinate
            const int32_t uU( p_iUMinimum+u );
            const int32_t uV( _getCurveV( p_vecCoefficients, uU ) + iSamplingOffset );

            //ds add keypoint
            vecPoolKeyPoints[u] = cv::KeyPoint( uU, uV, p_dKeyPointSize );
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
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
            cv::circle( p_matDisplay, cv::Point2i( uU, uV ), 1, CColorCodeBGR( 0, 165, 255 ), -1 );
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
    std::vector< cv::DMatch > vecMatchOriginal;
    m_pMatcher->match( p_matDescriptorOriginal, matDescriptorNew, vecMatchOriginal );

    //ds distances
    const double& dMatchingDistanceToRelative( cBestMatch.distance );
    const double& dMatchingDistanceToOriginal( vecMatchOriginal[0].distance );

    /*ds direct match
    if( m_dMatchingDistanceCutoff > dMatchingDistanceToOriginal )
    {
        //std::printf( "DIRECT MATCH - relative/original: %f/%f\n", dMatchingDistanceToRelative, dMatchingDistanceToOriginal );

        //ds return the match
        return CMatchTracking( p_vecPoolKeyPoints[cBestMatch.trainIdx], matDescriptorNew );
    }*/

    if( m_dMatchingDistanceCutoff > dMatchingDistanceToRelative )
    {
        if( m_dMatchingDistanceCutoffOriginal > dMatchingDistanceToOriginal )
        {
            //ds return the match
            return new CMatchTracking( p_vecPoolKeyPoints[cBestMatch.trainIdx], matDescriptorNew );
        }
        else
        {
            throw CExceptionNoMatchFoundInternal( "could not find a matching descriptor (ORIGINAL matching distance: "+ std::to_string( vecMatchOriginal[0].distance ) +")" );
        }
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
