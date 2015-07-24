#ifndef CTRIANGULATOR_H
#define CTRIANGULATOR_H

#include "vision/CStereoCamera.h"

class CTriangulator
{

//ds ctor/dtor
public:

    CTriangulator( const std::shared_ptr< CStereoCamera > p_pStereoCamera,
                   const std::shared_ptr< cv::DescriptorExtractor > p_pExtractor,
                   const std::shared_ptr< cv::DescriptorMatcher > p_pMatcher,
                   const float& p_fMatchingDistanceCutoff );
    ~CTriangulator( );

//ds members
private:

    //ds cameras
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds matching
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const float m_fMatchingDistanceCutoff;

    //ds triangulation
    const uint32_t m_uLimitedSearchRangeToLEFT;
    const uint32_t m_uLimitedSearchRangeToRIGHT;
    const uint32_t m_uLimitedSearchRange;
    const uint32_t m_uAdaptiveSteps;

//ds api
public:

    //ds triangulation methods
    const CPoint3DInCameraFrame getPointTriangulatedFull( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CPoint3DInCameraFrame getPointTriangulatedLimited( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CPoint3DInCameraFrame getPointTriangulatedAdaptive( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;

    const CMatchTriangulation getPointTriangulatedCompactInLEFT( const cv::Mat& p_matImageLEFT, const cv::KeyPoint& p_cKeyPointRIGHT, const CDescriptor& p_matReferenceDescriptorRIGHT ) const;
    const CMatchTriangulation getPointTriangulatedCompactInRIGHT( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;

    //ds testing
    const CPoint3DInCameraFrame getPointTriangulatedLimitedSVDLS( cv::Mat& p_matDisplayRIGHT, const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CPoint3DInCameraFrame getPointTriangulatedLimitedQRLS( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;
    const CPoint3DInCameraFrame getPointTriangulatedLimitedSVDDLT( const cv::Mat& p_matImageRIGHT, const cv::KeyPoint& p_cKeyPointLEFT, const CDescriptor& p_matReferenceDescriptorLEFT ) const;

private:

    friend class CMatcherEpipolar;
};

#endif //#define CTRIANGULATOR_H
