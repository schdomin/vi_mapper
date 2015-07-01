#ifndef CDETECTORMONOTILEWISE_H
#define CDETECTORMONOTILEWISE_H

#include <memory>

#include "vision/CPinholeCamera.h"

class CDetectorMonoTilewise
{

//ds ctor/dtor
public:

    CDetectorMonoTilewise( const std::shared_ptr< CPinholeCamera > p_pCamera,
                       const std::shared_ptr< cv::FeatureDetector > p_pDetector,
                       const uint8_t& p_uTileNumberQuadraticBase = 3 );
    ~CDetectorMonoTilewise( );

//ds members
private:

    //ds cameras
    const std::shared_ptr< CPinholeCamera > m_pCamera;

    //ds detection
    const std::shared_ptr< cv::FeatureDetector > m_pDetectorTilewise;

    //ds specifics
    const uint8_t m_uTileNumberQuadraticBase;
    const double m_dTileWidth;
    const double m_dTileHeight;

//ds api
public:

    const std::shared_ptr< std::vector< cv::KeyPoint > > detectKeyPointsTilewise( const cv::Mat& p_matImage ) const;
    const std::shared_ptr< std::vector< cv::KeyPoint > > detectKeyPointsTilewise( const cv::Mat& p_matImage, const cv::Mat& p_matMask ) const;

};

#endif //#define CDETECTORMONOTILEWISE_H
