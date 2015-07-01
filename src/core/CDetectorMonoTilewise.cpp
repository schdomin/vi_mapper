#include <core/CDetectorMonoTilewise.h>
#include "exceptions/CExceptionNoMatchFound.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CLogger.h"

CDetectorMonoTilewise::CDetectorMonoTilewise( const std::shared_ptr< CPinholeCamera > p_pCamera,
                              const std::shared_ptr< cv::FeatureDetector > p_pDetector,
                              const uint8_t& p_uTileNumberQuadraticBase ): m_pCamera( p_pCamera ),
                                                                           m_pDetectorTilewise( p_pDetector ),
                                                                           m_uTileNumberQuadraticBase( p_uTileNumberQuadraticBase ),
                                                                           m_dTileWidth( static_cast< double >( m_pCamera->m_uWidthPixel )/m_uTileNumberQuadraticBase ),
                                                                           m_dTileHeight( static_cast< double >( m_pCamera->m_uHeightPixel )/m_uTileNumberQuadraticBase )
{
    CLogger::openBox( );
    std::printf( "<CDetectorMonoTilewise>(CDetectorMonoTilewise) feature detector: %s\n", m_pDetectorTilewise->name( ).c_str( ) );
    std::printf( "<CDetectorMonoTilewise>(CDetectorMonoTilewise) tiles (quadratic): %u x %u\n", m_uTileNumberQuadraticBase, m_uTileNumberQuadraticBase );
    std::printf( "<CDetectorMonoTilewise>(CDetectorMonoTilewise) tile width: %f\n", m_dTileWidth );
    std::printf( "<CDetectorMonoTilewise>(CDetectorMonoTilewise) tile height: %f\n", m_dTileHeight );
    std::printf( "<CDetectorMonoTilewise>(CDetectorMonoTilewise) instance allocated\n" );
    CLogger::closeBox( );
}

CDetectorMonoTilewise::~CDetectorMonoTilewise( )
{
    std::printf( "<CDetectorMonoTilewise>(~CDetectorMonoTilewise) instance deallocated\n" );
}

const std::shared_ptr< std::vector< cv::KeyPoint > > CDetectorMonoTilewise::detectKeyPointsTilewise( const cv::Mat& p_matImage ) const
{
    //ds found points
    std::shared_ptr< std::vector< cv::KeyPoint > > vecFeatures( std::make_shared< std::vector< cv::KeyPoint > >( ) );

    //ds loop over tiles
    for( uint32_t u = 0; u < m_uTileNumberQuadraticBase; ++u )
    {
        for( uint32_t v = 0; v < m_uTileNumberQuadraticBase; ++v )
        {
            //ds compute rectangle points
            const double dULU( u*m_dTileWidth );
            const double dULV( v*m_dTileHeight );
            const double dLRU( dULU+m_dTileWidth );
            const double dLRV( dULV+m_dTileHeight );
            const cv::Rect cTileROI( cv::Point2d( dULU, dULV ), cv::Point2d( dLRU, dLRV ) );

            //ds current keypoints
            std::vector< cv::KeyPoint > vecFeaturesPerTile;

            //ds detect
            m_pDetectorTilewise->detect( p_matImage( cTileROI ), vecFeaturesPerTile );

            //ds fix positions (lambda magic)
            std::for_each( vecFeaturesPerTile.begin( ), vecFeaturesPerTile.end( ), [ &dULU, &dULV ]( cv::KeyPoint& p_cKeyPoint ) { p_cKeyPoint.pt.x += dULU; p_cKeyPoint.pt.y += dULV; } );

            //ds add the keypoints
            vecFeatures->insert( vecFeatures->end( ), vecFeaturesPerTile.begin( ), vecFeaturesPerTile.end( ) );
        }
    }

    return vecFeatures;
}
const std::shared_ptr< std::vector< cv::KeyPoint > > CDetectorMonoTilewise::detectKeyPointsTilewise( const cv::Mat& p_matImage, const cv::Mat& p_matMask ) const
{
    //ds found points
    std::shared_ptr< std::vector< cv::KeyPoint > > vecFeatures( std::make_shared< std::vector< cv::KeyPoint > >( ) );

    //ds loop over tiles
    for( uint32_t u = 0; u < m_uTileNumberQuadraticBase; ++u )
    {
        for( uint32_t v = 0; v < m_uTileNumberQuadraticBase; ++v )
        {
            //ds compute rectangle points
            const double dULU( u*m_dTileWidth );
            const double dULV( v*m_dTileHeight );
            const double dLRU( dULU+m_dTileWidth );
            const double dLRV( dULV+m_dTileHeight );
            const cv::Rect cTileROI( cv::Point2d( dULU, dULV ), cv::Point2d( dLRU, dLRV ) );

            //ds current keypoints
            std::vector< cv::KeyPoint > vecFeaturesPerTile;

            //ds detect
            m_pDetectorTilewise->detect( p_matImage( cTileROI ), vecFeaturesPerTile, p_matMask( cTileROI ) );

            //ds fix positions (lambda magic)
            std::for_each( vecFeaturesPerTile.begin( ), vecFeaturesPerTile.end( ), [ &dULU, &dULV ]( cv::KeyPoint& p_cKeyPoint ) { p_cKeyPoint.pt.x += dULU; p_cKeyPoint.pt.y += dULV; } );

            //ds add the keypoints
            vecFeatures->insert( vecFeatures->end( ), vecFeaturesPerTile.begin( ), vecFeaturesPerTile.end( ) );
        }
    }

    return vecFeatures;
}
