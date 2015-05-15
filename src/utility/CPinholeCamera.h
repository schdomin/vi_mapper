#ifndef CPINHOLECAMERA_H
#define CPINHOLECAMERA_H

#include <iostream>
#include <Eigen/Core>

#include "CStereoCamera.h"

class CPinholeCamera
{

public:

    CPinholeCamera( const std::string& p_strCameraLabel,
                    const Eigen::Matrix3d& p_matIntrinsic,
                    const Eigen::Vector4d& p_vecDistortionCoefficients,
                    const Eigen::Matrix3d& p_matRectification,
                    const Eigen::Matrix< double, 3, 4 >& p_matProjection,
                    const uint32_t p_uWidthPixel,
                    const uint32_t p_uHeightPixel ): m_strCameraLabel( p_strCameraLabel ),
                                                     m_matIntrinsic( p_matIntrinsic ),
                                                     m_dFx( p_matIntrinsic(0,0) ),
                                                     m_dFy( p_matIntrinsic(1,1) ),
                                                     m_dCx( p_matIntrinsic(0,2) ),
                                                     m_dCy( p_matIntrinsic(1,2) ),
                                                     m_vecDistortionCoefficients( p_vecDistortionCoefficients ),
                                                     m_matRectification( p_matRectification ),
                                                     m_matProjection( p_matProjection ),
                                                     m_vecPrincipalPoint( Eigen::Vector2d( p_matIntrinsic(0,2), p_matIntrinsic(1,2) ) ),
                                                     m_iWidthPixel( p_uWidthPixel ),
                                                     m_iHeightPixel( p_uHeightPixel ),
                                                     m_prRangeWidthNormalized( std::pair< double, double >( getNormalizedX( 0 ), getNormalizedX( p_uWidthPixel ) ) ),
                                                     m_prRangeHeightNormalized( std::pair< double, double >( getNormalizedY( 0 ), getNormalizedY( p_uHeightPixel ) ) )
    {
        //ds log complete configuration
        _logConfiguration( );
    }

    CPinholeCamera( const std::string& p_strCameraLabel,
                    const double p_matIntrinsic[9],
                    const double p_vecDistortionCoefficients[4],
                    const double p_matRectification[9],
                    const double p_matProjection[12],
                    const double p_matQuaternionToIMU[4],
                    const double p_vecTranslationToIMU[3],
                    const uint32_t& p_uWidthPixel,
                    const uint32_t& p_uHeightPixel ): m_strCameraLabel( p_strCameraLabel ),
                                                      m_matIntrinsic( Eigen::Matrix3d( p_matIntrinsic ).transpose( ) ),
                                                      m_dFx( m_matIntrinsic(0,0) ),
                                                      m_dFy( m_matIntrinsic(1,1) ),
                                                      m_dCx( m_matIntrinsic(0,2) ),
                                                      m_dCy( m_matIntrinsic(1,2) ),
                                                      m_vecDistortionCoefficients( p_vecDistortionCoefficients ),
                                                      m_matRectification( Eigen::Matrix3d( p_matRectification ).transpose( ) ),
                                                      m_matProjection( Eigen::Matrix< double, 4, 3 >( p_matProjection ).transpose( ) ),
                                                      m_vecPrincipalPoint( Eigen::Vector2d( m_dCx, m_dCy ) ),
                                                      m_vecRotationToIMU( p_matQuaternionToIMU ),
                                                      m_vecTranslationToIMU( p_vecTranslationToIMU ),
                                                      m_iWidthPixel( p_uWidthPixel ),
                                                      m_iHeightPixel( p_uHeightPixel ),
                                                      m_prRangeWidthNormalized( std::pair< double, double >( getNormalizedX( 0 ), getNormalizedX( p_uWidthPixel ) ) ),
                                                      m_prRangeHeightNormalized( std::pair< double, double >( getNormalizedY( 0 ), getNormalizedY( p_uHeightPixel ) ) )
    {
        //ds log complete configuration
        _logConfiguration( );
    }

    //ds no manual dynamic allocation
    ~CPinholeCamera( ){ }

public:

    const std::string m_strCameraLabel;

    //ds intrinsics
    const Eigen::Matrix3d m_matIntrinsic;
    const double m_dFx;
    const double m_dFy;
    const double m_dCx;
    const double m_dCy;
    const Eigen::Vector4d m_vecDistortionCoefficients;
    const Eigen::Matrix3d m_matRectification;
    const Eigen::Matrix< double, 3, 4 > m_matProjection;
    const Eigen::Vector2d m_vecPrincipalPoint;

    //ds extrinsics
    const Eigen::Quaterniond m_vecRotationToIMU;
    const Eigen::Vector3d m_vecTranslationToIMU;

    //ds misc
    const int32_t m_iWidthPixel;
    const int32_t m_iHeightPixel;
    const std::pair< double, double > m_prRangeWidthNormalized;
    const std::pair< double, double > m_prRangeHeightNormalized;

//ds access
public:

    const Eigen::Vector3d getNormalized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint(0)-m_dCx )/m_dFx, ( p_vecPoint(1)-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getNormalized( const cv::KeyPoint& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.pt.x-m_dCx )/m_dFx, ( p_vecPoint.pt.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getNormalized( const cv::Point2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getNormalized( const cv::Point2f& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy, 1.0 );
    }
    const double getNormalizedX( const double& p_dX ) const
    {
        return ( p_dX-m_dCx )/m_dFx;
    }
    const double getNormalizedY( const double& p_dY ) const
    {
        return ( p_dY-m_dCy )/m_dFy;
    }
    const Eigen::Vector2d getDenormalized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return Eigen::Vector2d( p_vecPoint(0)*m_dFx+m_dCx, p_vecPoint(1)*m_dFy+m_dCy );
    }
    const double getDenormalizedX( const double& p_dX ) const
    {
        return p_dX*m_dFx+m_dCx;
    }
    const double getDenormalizedY( const double& p_dY ) const
    {
        return p_dY*m_dFy+m_dCy;
    }
    const int32_t getU( const double& p_dX ) const
    {
        return p_dX*m_dFx+m_dCx;
    }
    const int32_t getV( const double& p_dY ) const
    {
        return p_dY*m_dFy+m_dCy;
    }

//ds helpers
private:

    void _logConfiguration( ) const
    {
        //ds log complete configuration
        std::cout << "--------------------------------------------------------------------------------------------------------------------------------------" << "\n"
                  << "CONFIGURATION CAMERA: " << m_strCameraLabel << "\n"
                  << "Fx: " << m_dFx << "\n"
                  << "Fy: " << m_dFy << "\n"
                  << "Cx: " << m_dCx << "\n"
                  << "Cy: " << m_dCy << "\n"
                  << "\nIntrinsic matrix (K):\n\n" << m_matIntrinsic << "\n\n"
                  << "Distortion coefficients (D): " << m_vecDistortionCoefficients.transpose( ) << "\n"
                  << "\nRectification matrix (R):\n\n" << m_matRectification << "\n\n"
                  << "\nProjection matrix (P):\n\n" << m_matProjection << "\n\n"
                  << "Principal point: " << m_vecPrincipalPoint.transpose( ) << "\n"
                  << "Translation (CAMERA to IMU): " << m_vecTranslationToIMU.transpose( ) << "\n"
                  << "\nRotation matrix (CAMERA to IMU):\n\n" << m_vecRotationToIMU.matrix( ) << "\n\n"
                  << "Resolution (w x h): " << m_iWidthPixel << " x " << m_iHeightPixel << "\n"
                  << "Normalized x range: [" << m_prRangeWidthNormalized.first << ", " << m_prRangeWidthNormalized.second << "]\n"
                  << "Normalized y range: [" << m_prRangeHeightNormalized.first << ", " << m_prRangeHeightNormalized.second << "]" << std::endl;
    }

//ds friends
friend class CStereoCamera;

};


#endif //CPINHOLECAMERA_H
