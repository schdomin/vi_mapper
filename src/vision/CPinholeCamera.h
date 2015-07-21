#ifndef CPINHOLECAMERA_H
#define CPINHOLECAMERA_H

#include <iostream>

#include "types/Typedefs.h"
#include "utility/CLogger.h"

class CPinholeCamera
{

public:

    /*CPinholeCamera( const std::string& p_strCameraLabel,
                    const Eigen::Matrix3d& p_matIntrinsic,
                    const Eigen::Vector4d& p_vecDistortionCoefficients,
                    const Eigen::Matrix3d& p_matRectification,
                    const Eigen::Matrix< double, 3, 4 >& p_matProjection,
                    const uint32_t p_uWidthPixel,
                    const uint32_t p_uHeightPixel ): m_strCameraLabel( p_strCameraLabel ),
                                                     m_matIntrinsic( p_matIntrinsic ),
                                                     m_dFx( m_matIntrinsic(0,0) ),
                                                     m_dFy( m_matIntrinsic(1,1) ),
                                                     m_dFxNormalized( m_dFx/p_uWidthPixel ),
                                                     m_dFyNormalized( m_dFy/p_uHeightPixel ),
                                                     m_dCx( m_matIntrinsic(0,2) ),
                                                     m_dCy( m_matIntrinsic(1,2) ),
                                                     m_dCxNormalized( m_dCx/p_uWidthPixel ),
                                                     m_dCyNormalized( m_dCy/p_uHeightPixel ),
                                                     m_vecDistortionCoefficients( p_vecDistortionCoefficients ),
                                                     m_matRectification( p_matRectification ),
                                                     m_matProjection( p_matProjection ),
                                                     m_vecPrincipalPoint( Eigen::Vector2d( m_dCx, m_dCy ) ),
                                                     m_vecPrincipalPointNormalized( Eigen::Vector3d( m_dCxNormalized, m_dCyNormalized, 1.0 ) ),
                                                     m_uWidthPixel( p_uWidthPixel ),
                                                     m_uHeightPixel( p_uHeightPixel ),
                                                     m_iWidthPixel( m_uWidthPixel ),
                                                     m_iHeightPixel( m_uHeightPixel ),
                                                     m_prRangeWidthNormalized( std::pair< double, double >( getNormalizedX( 0 ), getNormalizedX( p_uWidthPixel ) ) ),
                                                     m_prRangeHeightNormalized( std::pair< double, double >( getNormalizedY( 0 ), getNormalizedY( p_uHeightPixel ) ) )
    {
        //ds log complete configuration
        _logConfiguration( );
    }*/

    CPinholeCamera( const std::string& p_strCameraLabel,
                    const double p_matIntrinsic[9],
                    const double p_vecDistortionCoefficients[4],
                    const double p_matRectification[9],
                    const double p_matProjection[12],
                    const double p_matQuaternionFromIMU[4],
                    const double p_vecTranslationFromIMU[3],
                    const uint32_t& p_uWidthPixel,
                    const uint32_t& p_uHeightPixel,
                    const double& p_dFocalLengthMeters ): m_strCameraLabel( p_strCameraLabel ),
                                                      m_matIntrinsic( Eigen::Matrix3d( p_matIntrinsic ).transpose( ) ),
                                                      m_dFx( m_matIntrinsic(0,0) ),
                                                      m_dFy( m_matIntrinsic(1,1) ),
                                                      m_dFxNormalized( m_dFx/p_uWidthPixel ),
                                                      m_dFyNormalized( m_dFy/p_uHeightPixel ),
                                                      m_dCx( m_matIntrinsic(0,2) ),
                                                      m_dCy( m_matIntrinsic(1,2) ),
                                                      m_dCxNormalized( m_dCx/p_uWidthPixel ),
                                                      m_dCyNormalized( m_dCy/p_uHeightPixel ),
                                                      m_dFocalLengthMeters( p_dFocalLengthMeters ),
                                                      m_vecDistortionCoefficients( p_vecDistortionCoefficients ),
                                                      m_matRectification( Eigen::Matrix3d( p_matRectification ).transpose( ) ),
                                                      m_matProjection( Eigen::Matrix< double, 4, 3 >( p_matProjection ).transpose( ) ),
                                                      m_vecPrincipalPoint( Eigen::Vector2d( m_dCx, m_dCy ) ),
                                                      m_vecPrincipalPointNormalized( Eigen::Vector3d( m_dCxNormalized, m_dCyNormalized, 1.0 ) ),
                                                      m_vecRotationFromIMU( p_matQuaternionFromIMU ),
                                                      m_vecTranslationFromIMU( p_vecTranslationFromIMU ),
                                                      m_matTransformationFromIMU( Eigen::Matrix4d::Identity( ) ),
                                                      m_uWidthPixel( p_uWidthPixel ),
                                                      m_uHeightPixel( p_uHeightPixel ),
                                                      m_iWidthPixel( m_uWidthPixel ),
                                                      m_iHeightPixel( m_uHeightPixel ),
                                                      m_dWidthPixel( m_uWidthPixel ),
                                                      m_dHeightPixel( m_uHeightPixel ),
                                                      m_prRangeWidthNormalized( std::pair< double, double >( getNormalizedX( 0 ), getNormalizedX( p_uWidthPixel ) ) ),
                                                      m_prRangeHeightNormalized( std::pair< double, double >( getNormalizedY( 0 ), getNormalizedY( p_uHeightPixel ) ) )
    {
        m_matTransformationFromIMU.linear( )      = m_vecRotationFromIMU.toRotationMatrix( );
        m_matTransformationFromIMU.translation( ) = m_vecTranslationFromIMU;

        //ds TODO Eigen BUG? corner point 3,3 not set to zero
        //m_matTransformationFromIMU( 3, 3 ) = 1.0;

        //ds set inverse transform
        m_matTransformationToIMU = m_matTransformationFromIMU.inverse( );

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
    const double m_dFxNormalized;
    const double m_dFyNormalized;
    const double m_dCx;
    const double m_dCy;
    const double m_dCxNormalized;
    const double m_dCyNormalized;
    const double m_dFocalLengthMeters;
    const Eigen::Vector4d m_vecDistortionCoefficients;
    const Eigen::Matrix3d m_matRectification;
    const MatrixProjection m_matProjection;
    const Eigen::Vector2d m_vecPrincipalPoint;
    const Eigen::Vector3d m_vecPrincipalPointNormalized;

    //ds extrinsics
    const Eigen::Quaterniond m_vecRotationFromIMU;
    const Eigen::Vector3d m_vecTranslationFromIMU;
    Eigen::Isometry3d m_matTransformationFromIMU;
    Eigen::Isometry3d m_matTransformationToIMU;

    //ds misc
    const uint32_t m_uWidthPixel;
    const uint32_t m_uHeightPixel;
    const int32_t m_iWidthPixel;
    const int32_t m_iHeightPixel;
    const double m_dWidthPixel;
    const double m_dHeightPixel;
    const std::pair< double, double > m_prRangeWidthNormalized;
    const std::pair< double, double > m_prRangeHeightNormalized;

//ds access
public:

    const Eigen::Vector3d getHomogenized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint(0)-m_dCx )/m_dFx, ( p_vecPoint(1)-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getHomogenized( const cv::KeyPoint& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.pt.x-m_dCx )/m_dFx, ( p_vecPoint.pt.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getHomogenized( const cv::Point2d& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getHomogenized( const cv::Point2f& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector3d getHomogenized( const cv::Point2i& p_vecPoint ) const
    {
        return Eigen::Vector3d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy, 1.0 );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2i& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2f& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy );
    }
    const Eigen::Vector2d getNormalized( const cv::Point2d& p_vecPoint ) const
    {
        return Eigen::Vector2d( ( p_vecPoint.x-m_dCx )/m_dFx, ( p_vecPoint.y-m_dCy )/m_dFy );
    }
    const double getNormalizedX( const double& p_dX ) const
    {
        return ( p_dX-m_dCx )/m_dFx;
    }
    const double getNormalizedY( const double& p_dY ) const
    {
        return ( p_dY-m_dCy )/m_dFy;
    }
    const cv::Point2d getDenormalized( const Eigen::Vector2d& p_vecPoint ) const
    {
        return cv::Point2d( p_vecPoint(0)*m_dFx+m_dCx, p_vecPoint(1)*m_dFy+m_dCy );
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
    const CPoint2DHomogenized getHomogeneousProjection( const CPoint3DHomogenized& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*p_vecPoint );

        //ds return homogeneous
        return vecProjectionInhomogeneous/vecProjectionInhomogeneous(2);
    }
    const CPoint2DHomogenized getHomogeneousProjection( const CPoint3DInCameraFrame& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPoint(0), p_vecPoint(1), p_vecPoint(2), 1.0 ) );

        //ds return homogeneous
        return vecProjectionInhomogeneous/vecProjectionInhomogeneous(2);
    }
    const cv::Point2d getProjection( const CPoint3DInCameraFrame& p_vecPoint ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPoint(0), p_vecPoint(1), p_vecPoint(2), 1.0 ) );

        //ds return uv point
        return cv::Point2d( vecProjectionInhomogeneous(0)/vecProjectionInhomogeneous(2), vecProjectionInhomogeneous(1)/vecProjectionInhomogeneous(2) );
    }
    const CPoint2DInCameraFrame getUV( const CPoint3DInCameraFrame& p_vecPointXYZ ) const
    {
        //ds compute inhomo projection
        const Eigen::Vector3d vecProjectionInhomogeneous( m_matProjection*CPoint3DHomogenized( p_vecPointXYZ.x( ), p_vecPointXYZ.y( ), p_vecPointXYZ.z( ), 1.0 ) );

        //ds return uv point
        return CPoint2DInCameraFrame( vecProjectionInhomogeneous(0)/vecProjectionInhomogeneous(2), vecProjectionInhomogeneous(1)/vecProjectionInhomogeneous(2) );
    }

//ds helpers
private:

    void _logConfiguration( ) const
    {
        //ds log complete configuration
        CLogger::openBox( );
        std::cout << "Configuration camera: " << m_strCameraLabel << "\n\n"
                  << "Fx: " << m_dFx << "\n"
                  << "Fy: " << m_dFy << "\n"
                  << "Cx: " << m_dCx << "\n"
                  << "Cy: " << m_dCy << "\n"
                  << "\nIntrinsic matrix (K):\n\n" << m_matIntrinsic << "\n\n"
                  << "Distortion coefficients (D): " << m_vecDistortionCoefficients.transpose( ) << "\n"
                  << "\nRectification matrix (R):\n\n" << m_matRectification << "\n\n"
                  << "\nProjection matrix (P):\n\n" << m_matProjection << "\n\n"
                  << "Principal point: " << m_vecPrincipalPoint.transpose( ) << "\n"
                  << "\nTransformation matrix (IMU to CAMERA):\n\n" << m_matTransformationFromIMU.matrix( ) << "\n\n"
                  << "\nTransformation matrix (CAMERA to IMU):\n\n" << m_matTransformationToIMU.matrix( ) << "\n\n"
                  << "Resolution (w x h): " << m_uWidthPixel << " x " << m_uHeightPixel << "\n"
                  << "Normalized x range: [" << m_prRangeWidthNormalized.first << ", " << m_prRangeWidthNormalized.second << "]\n"
                  << "Normalized y range: [" << m_prRangeHeightNormalized.first << ", " << m_prRangeHeightNormalized.second << "]" << std::endl;
        CLogger::closeBox( );
    }

};


#endif //CPINHOLECAMERA_H
