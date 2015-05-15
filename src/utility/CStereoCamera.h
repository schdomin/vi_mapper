#ifndef CSTEREOCAMERA_H
#define CSTEREOCAMERA_H

#include <opencv/cv.h>

#include "CPinholeCamera.h"
#include "CWrapperOpenCV.h"

class CStereoCamera
{

public:

    CStereoCamera( const CPinholeCamera& p_cCameraLEFT, const CPinholeCamera& p_cCameraRIGHT ): m_cCameraLEFT( p_cCameraLEFT ), m_cCameraRIGHT( p_cCameraRIGHT )
    {
        //ds setup extrinsic transformations
        m_matTransformLEFTtoIMU.linear()        = p_cCameraLEFT.m_vecRotationToIMU.matrix( );
        m_matTransformLEFTtoIMU.translation( )  = p_cCameraLEFT.m_vecTranslationToIMU;
        m_matTransformRIGHTtoIMU.linear( )      = p_cCameraRIGHT.m_vecRotationToIMU.matrix( );
        m_matTransformRIGHTtoIMU.translation( ) = p_cCameraRIGHT.m_vecTranslationToIMU;
        m_matTransformLEFTtoRIGHT               = m_matTransformRIGHTtoIMU.inverse( )*m_matTransformLEFTtoIMU;

        //ds compute undistorted and rectified mappings
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_cCameraLEFT.m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_cCameraLEFT.m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_cCameraLEFT.m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_cCameraLEFT.m_matProjection ),
                                     cv::Size( m_cCameraLEFT.m_iWidthPixel, m_cCameraLEFT.m_iHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsLEFT[0],
                                     m_arrUndistortRectifyMapsLEFT[1] );
        cv::initUndistortRectifyMap( CWrapperOpenCV::toCVMatrix( p_cCameraRIGHT.m_matIntrinsic ),
                                     CWrapperOpenCV::toCVVector( p_cCameraRIGHT.m_vecDistortionCoefficients ),
                                     CWrapperOpenCV::toCVMatrix( p_cCameraRIGHT.m_matRectification ),
                                     CWrapperOpenCV::toCVMatrix( p_cCameraRIGHT.m_matProjection ),
                                     cv::Size( m_cCameraRIGHT.m_iWidthPixel, m_cCameraRIGHT.m_iHeightPixel ),
                                     CV_16SC2,
                                     m_arrUndistortRectifyMapsRIGHT[0],
                                     m_arrUndistortRectifyMapsRIGHT[1] );
    }

    //ds no manual dynamic allocation
    ~CStereoCamera( ){ }


//ds fields
private:

    //ds stereo cameras
    const CPinholeCamera m_cCameraLEFT;
    const CPinholeCamera m_cCameraRIGHT;

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
    void undistortAndrectify( cv::Mat& p_matImageLEFT, cv::Mat& p_matImageRight ) const
    {
        //ds remap images
        cv::remap( p_matImageLEFT, p_matImageLEFT, m_arrUndistortRectifyMapsLEFT[0], m_arrUndistortRectifyMapsLEFT[1], cv::INTER_LINEAR );
        cv::remap( p_matImageRight, p_matImageRight, m_arrUndistortRectifyMapsRIGHT[0], m_arrUndistortRectifyMapsRIGHT[1], cv::INTER_LINEAR );
    }

};

#endif //CSTEREOCAMERA_H
