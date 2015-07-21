#include "CTriangulator.h"

#include "exceptions/CExceptionNoMatchFound.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CTriangulator::CTriangulator( const std::shared_ptr< CStereoCamera > p_pStereoCamera,
                              const std::shared_ptr< cv::DescriptorExtractor > p_pExtractor,
                              const std::shared_ptr< cv::DescriptorMatcher > p_pMatcher,
                              const float& p_fMatchingDistanceCutoff ): m_pCameraSTEREO( p_pStereoCamera ),
                                                                                            m_pExtractor( p_pExtractor ),
                                                                                            m_pMatcher( p_pMatcher ),
                                                                                            m_fMatchingDistanceCutoff( p_fMatchingDistanceCutoff ),
                                                                                            m_uLimitedSearchRangeToLeft( 50 ),
                                                                                            m_uLimitedSearchRangeToRight( 10 ),
                                                                                            m_uLimitedSearchRange( m_uLimitedSearchRangeToLeft+m_uLimitedSearchRangeToRight ),
                                                                                            m_uAdaptiveSteps( 10 )
{
    CLogger::openBox( );
    std::printf( "<CTriangulator>(CTriangulator) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CTriangulator>(CTriangulator) matching distance cutoff: %f\n", m_fMatchingDistanceCutoff );
    std::printf( "<CTriangulator>(CTriangulator) instance allocated\n" );
    CLogger::closeBox( );
}

CTriangulator::~CTriangulator( )
{
    std::printf( "<CTriangulator>(~CTriangulator) instance deallocated\n" );
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedFull( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds buffer keypoint size
    const float& fKeyPointSize( p_cKeyPointLEFT.size );

    //ds get keypoint to eigen space
    const CPoint2DInCameraFrame vecReference( CWrapperOpenCV::fromCVVector( p_cKeyPointLEFT.pt ) );

    //ds search limit
    const uint32_t uLimit( m_pCameraSTEREO->m_uPixelWidth );

    //ds right keypoint vector (check the full range)
    std::vector< cv::KeyPoint > vecPoolKeyPoints( uLimit );

    //ds set the keypoints
    for( uint32_t uU = 0; uU < uLimit; ++uU )
    {
        vecPoolKeyPoints[uU] = cv::KeyPoint( uU, vecReference(1), fKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds get the matching keypoint
        const CPoint2DInCameraFrame vecMatch( CWrapperOpenCV::fromCVVector( vecPoolKeyPoints[vecMatches[0].trainIdx].pt ) );

        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( vecReference, vecMatch, m_pCameraSTEREO->m_pCameraLEFT->m_matProjection, m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedLimited( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t iVReference( p_cKeyPointLEFT.pt.y );
    const double dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( iUReference-m_uLimitedSearchRangeToLeft );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
    }
    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( p_cKeyPointLEFT.pt,
                                                                           vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                           m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                           m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedAdaptive( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds buffer keypoint size
    const float& fKeyPointSize( p_cKeyPointLEFT.size );

    //ds get keypoint to eigen space
    const CPoint2DInCameraFrame vecReference( CWrapperOpenCV::fromCVVector( p_cKeyPointLEFT.pt ) );
    const int32_t iUReference( vecReference(0) );

    //ds keypoint buffer
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uAdaptiveSteps );

    //ds start point
    int32_t iU( iUReference+m_uAdaptiveSteps );

    assert( 0 < iUReference );

    //ds scan to the left
    while( 0 < iU )
    {
        for( uint32_t u = 0; u < m_uAdaptiveSteps; ++u )
        {
            vecPoolKeyPoints[u] = cv::KeyPoint( iU, vecReference(1), fKeyPointSize );
            --iU;
        }

        //ds compute descriptors
        cv::Mat matPoolDescriptors;
        m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

        //ds if we managed to compute descriptors
        if( !vecPoolKeyPoints.empty( ) )
        {
            //ds match the descriptors
            std::vector< cv::DMatch > vecMatches;
            m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

            //ds if we found matches
            if( !vecMatches.empty( ) )
            {
                //ds make sure the matcher returned a valid ID
                assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < m_uAdaptiveSteps );

                //ds check match quality
                if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
                {
                    //ds get the matching keypoint
                    const CPoint2DInCameraFrame vecMatch( CWrapperOpenCV::fromCVVector( vecPoolKeyPoints[vecMatches[0].trainIdx].pt ) );

                    //ds triangulated 3d point
                    return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( vecReference, vecMatch, m_pCameraSTEREO->m_pCameraLEFT->m_matProjection, m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
                }
            }
        }
    }

    //ds no match found if still here
    throw CExceptionNoMatchFound( "no match found" );
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedLimitedSVDLS( cv::Mat& p_matDisplayRIGHT, const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( std::max( iUReference-m_uLimitedSearchRangeToLeft, static_cast< uint32_t >( 0 ) ) );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
        cv::circle( p_matDisplayRIGHT, vecPoolKeyPoints[u].pt, 1, CColorCodeBGR( 125, 125, 125 ), -1 );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( p_cKeyPointLEFT.pt,
                                                                           vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                           m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                           m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedLimitedQRLS( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( iUReference-m_uLimitedSearchRangeToLeft );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds precheck for nans
        const CPoint3DInCameraFrame vecTriangulatedPoint( CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( p_cKeyPointLEFT.pt,
                                                                                                                     vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                                                                     m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                                                                     m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection ) );

        //ds check if valid
        if( 0 != vecTriangulatedPoint.squaredNorm( ) )
        {
            return vecTriangulatedPoint;
        }
        else
        {
            throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) QRLS failed" );
        }
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}

const CPoint3DInCameraFrame CTriangulator::getPointTriangulatedLimitedSVDDLT( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const
{
    //ds left references
    const int32_t& iUReference( p_cKeyPointLEFT.pt.x );
    const int32_t& iVReference( p_cKeyPointLEFT.pt.y );
    const double& dKeyPointSize( p_cKeyPointLEFT.size );

    assert( 0 < iUReference );

    //ds compute loop range (dont care about overflows to keep performance here, the matcher can handle negative coordinates)
    const uint32_t uBegin( iUReference-m_uLimitedSearchRangeToLeft );

    //ds right keypoint vector
    std::vector< cv::KeyPoint > vecPoolKeyPoints( m_uLimitedSearchRange );

    //ds set the keypoints
    for( uint32_t u = 0; u < m_uLimitedSearchRange; ++u )
    {
        vecPoolKeyPoints[u] = cv::KeyPoint( uBegin+u, iVReference, dKeyPointSize );
    }

    //ds compute descriptors
    cv::Mat matPoolDescriptors;
    m_pExtractor->compute( p_matImageRIGHT, vecPoolKeyPoints, matPoolDescriptors );

    //ds check if we failed to compute descriptors
    if( vecPoolKeyPoints.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) could not compute descriptors" );
    }

    //ds match the descriptors
    std::vector< cv::DMatch > vecMatches;
    m_pMatcher->match( p_matReferenceDescriptorLEFT, matPoolDescriptors, vecMatches );

    if( vecMatches.empty( ) )
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) no match found" );
    }

    //ds make sure the matcher returned a valid ID
    assert( static_cast< std::vector< cv::KeyPoint >::size_type >( vecMatches[0].trainIdx ) < vecPoolKeyPoints.size( ) );

    //ds check match quality
    if( m_fMatchingDistanceCutoff > vecMatches[0].distance )
    {
        //ds triangulate 3d point
        return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDDLT( p_cKeyPointLEFT.pt,
                                                                            vecPoolKeyPoints[vecMatches[0].trainIdx].pt,
                                                                            m_pCameraSTEREO->m_pCameraLEFT->m_matProjection,
                                                                            m_pCameraSTEREO->m_pCameraRIGHT->m_matProjection );
    }
    else
    {
        throw CExceptionNoMatchFound( "<CTriangulator>(getPointTriangulatedLimited) matching distance: " + std::to_string( vecMatches[0].distance ) );
    }
}
