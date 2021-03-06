#ifndef CSTEREOCAMERA_H
#define CSTEREOCAMERA_H

#include "CPinholeCamera.h"
#include "utility/CWrapperOpenCV.h"

class CStereoCamera
{

public:

    CStereoCamera( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT, const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT ): m_pCameraLEFT( p_pCameraLEFT ),
                                                                                                                                    m_pCameraRIGHT( p_pCameraRIGHT ),
                                                                                                                                    m_uPixelWidth( p_pCameraLEFT->m_uWidthPixel ),
                                                                                                                                    m_uPixelHeight( p_pCameraLEFT->m_uHeightPixel ),
                                                                                                                                    m_cVisibleRange( 0, 0, m_uPixelWidth, m_uPixelHeight )
    {
        //ds check stereo setup
        assert( p_pCameraLEFT->m_uWidthPixel == p_pCameraRIGHT->m_uWidthPixel );
        assert( p_pCameraLEFT->m_uHeightPixel == p_pCameraRIGHT->m_uHeightPixel );

        //ds setup extrinsic transformations
        m_matTransformLEFTtoIMU   = p_pCameraLEFT->m_matTransformationCAMERAtoIMU;
        m_matTransformRIGHTtoIMU  = p_pCameraRIGHT->m_matTransformationCAMERAtoIMU;
        m_matTransformLEFTtoRIGHT = m_matTransformRIGHTtoIMU.inverse( )*m_matTransformLEFTtoIMU;

        m_dBaselineMeters = m_matTransformLEFTtoRIGHT.translation( ).norm( );

        //ds log complete configuration
        CLogger::openBox( );
        std::cout << "Configuration stereo camera: " << m_pCameraLEFT->m_strCameraLabel << "-" << m_pCameraRIGHT->m_strCameraLabel << "\n"
                  << "\nTransformation matrix (LEFT to RIGHT):\n\n" << m_matTransformLEFTtoRIGHT.matrix( ) << "\n"
                  << "\nBaseline: " << m_dBaselineMeters << "m" << std::endl;
        CLogger::closeBox( );

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

    //ds wrapping constructors
    CStereoCamera( const CPinholeCamera& p_cCameraLEFT, const CPinholeCamera& p_cCameraRIGHT ): CStereoCamera( std::make_shared< CPinholeCamera >( p_cCameraLEFT ), std::make_shared< CPinholeCamera >( p_cCameraRIGHT ) )
    {
        //ds nothing to do
    }

    CStereoCamera( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                   const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT,
                   const Eigen::Vector3d& p_vecTranslationToRIGHT ): CStereoCamera( p_pCameraLEFT, p_pCameraRIGHT )
    {
        //ds adjust transformation
        m_matTransformLEFTtoRIGHT = Eigen::Matrix4d::Identity( );
        m_matTransformLEFTtoRIGHT.translation( ) = p_vecTranslationToRIGHT;

        m_dBaselineMeters = m_matTransformLEFTtoRIGHT.translation( ).norm( );

        CLogger::openBox( );
        std::printf( "<CStereoCamera>(CStereoCamera) manually set transformation LEFT to RIGHT: \n" );
        std::cout << m_matTransformLEFTtoRIGHT.matrix( ) << "\n" << std::endl;
        std::printf( "<CStereoCamera>(CStereoCamera) new baseline: %f\n", m_dBaselineMeters );
        CLogger::closeBox( );
    }

    //ds no manual dynamic allocation
    ~CStereoCamera( ){ }


//ds fields
public:

    //ds stereo cameras
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;

    //ds intrinsics
    double m_dBaselineMeters;

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

    //ds visible range
    const cv::Rect m_cVisibleRange;

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
