#ifndef CSTEREOCAMERA_H
#define CSTEREOCAMERA_H

#include <cassert>
#include <memory>

#include "CPinholeCamera.h"
#include "CWrapperOpenCV.h"

class CStereoCamera
{

public:

    CStereoCamera( const std::shared_ptr< CPinholeCamera >& p_pCameraLEFT, const std::shared_ptr< CPinholeCamera >& p_pCameraRIGHT ): m_pCameraLEFT( p_pCameraLEFT ),
                                                                                                                                      m_pCameraRIGHT( p_pCameraRIGHT ),
                                                                                                                                      m_uPixelWidth( p_pCameraLEFT->m_uWidthPixel ),
                                                                                                                                      m_uPixelHeight( p_pCameraLEFT->m_uHeightPixel )
    {
        //ds check stereo setup
        assert( p_pCameraLEFT->m_uWidthPixel == p_pCameraRIGHT->m_uWidthPixel );
        assert( p_pCameraLEFT->m_uHeightPixel == p_pCameraRIGHT->m_uHeightPixel );

        //ds setup extrinsic transformations
        m_matTransformLEFTtoIMU.linear()        = p_pCameraLEFT->m_vecRotationToIMU.matrix( );
        m_matTransformLEFTtoIMU.translation( )  = p_pCameraLEFT->m_vecTranslationToIMU;
        m_matTransformRIGHTtoIMU.linear( )      = p_pCameraRIGHT->m_vecRotationToIMU.matrix( );
        m_matTransformRIGHTtoIMU.translation( ) = p_pCameraRIGHT->m_vecTranslationToIMU;
        m_matTransformLEFTtoRIGHT               = m_matTransformRIGHTtoIMU.inverse( )*m_matTransformLEFTtoIMU;

        //ds compute undistorted and rectified mappings
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_pCameraLEFT->m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraLEFT->m_matProjection ),
                                     cv::Size( m_pCameraLEFT->m_uWidthPixel, m_pCameraLEFT->m_uHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsLEFT[0],
                                     m_arrUndistortRectifyMapsLEFT[1] );
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_pCameraRIGHT->m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_pCameraRIGHT->m_matProjection ),
                                     cv::Size( m_pCameraRIGHT->m_uWidthPixel, m_pCameraRIGHT->m_uHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsRIGHT[0],
                                     m_arrUndistortRectifyMapsRIGHT[1] );
    }

    //ds no manual dynamic allocation
    ~CStereoCamera( ){ }


//ds fields
public:

    //ds stereo cameras
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;

    //ds common dimensions
    const uint32_t m_uPixelWidth;
    const uint32_t m_uPixelHeight;

    //ds extrinsics
    Eigen::Isometry3d m_matTransformLEFTtoIMU;
    Eigen::Isometry3d m_matTransformRIGHTtoIMU;
    Eigen::Isometry3d m_matTransformLEFTtoRIGHT;

    //ds undistortion/rectification
    cv::Mat m_arrUndistortRectifyMapsLEFT[2];
    cv::Mat m_arrUndistortRectifyMapsRIGHT[2];

//ds accessors
public:

    //ds undistortion/rectification (TODO remove UGLY in/out)
    void undistortAndrectify( cv::Mat& p_matImageLEFT, cv::Mat& p_matImageRIGHT ) const
    {
        //ds remap images
        cv::remap( p_matImageLEFT, p_matImageLEFT, m_arrUndistortRectifyMapsLEFT[0], m_arrUndistortRectifyMapsLEFT[1], cv::INTER_LINEAR );
        cv::remap( p_matImageRIGHT, p_matImageRIGHT, m_arrUndistortRectifyMapsRIGHT[0], m_arrUndistortRectifyMapsRIGHT[1], cv::INTER_LINEAR );
    }

};

#endif //CSTEREOCAMERA_H
