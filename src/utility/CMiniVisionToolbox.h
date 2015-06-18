#ifndef CMINIVISIONTOOLBOX_H_
#define CMINIVISIONTOOLBOX_H_

#include "configuration/Types.h"

class CMiniVisionToolbox
{

//ds methods
public:

    static const cv::Point2i getPointBoundarized( const int32_t& p_iPointX, const int32_t& p_iPointY, const uint32_t& p_uImageRows, const uint32_t& p_uImageCols );

    static const Eigen::Vector3d toHomo( const Eigen::Vector2d& p_vecVector ){ return Eigen::Vector3d( p_vecVector(0), p_vecVector(1), 1.0 ); }
    static const Eigen::Vector4d toHomo( const Eigen::Vector3d& p_vecVector ){ return Eigen::Vector4d( p_vecVector(0), p_vecVector(1), p_vecVector(2), 1.0 ); }
    static const Eigen::Vector2d fromHomo( const Eigen::Vector3d& p_vecVector ){ return Eigen::Vector2d( p_vecVector(0), p_vecVector(1) ); }
    static const Eigen::Vector3d fromHomo( const Eigen::Vector4d& p_vecVector ){ return Eigen::Vector3d( p_vecVector(0), p_vecVector(1), p_vecVector(2) ); }
    static const Eigen::Vector3d getNormalized( const Eigen::Vector2d& p_vecPoint, const double& dFx, const double& dFy, const double& dCx, const double& dCy )
    {
        return Eigen::Vector3d( ( p_vecPoint(0)-dCx )/dFx, ( p_vecPoint(1)-dCy )/dFy, 1.0 );
    }
    static const Eigen::Vector2d getDenormalized( const Eigen::Vector3d& p_vecPoint, const double& dFx, const double& dFy, const double& dCx, const double& dCy )
    {
        return Eigen::Vector2d( p_vecPoint(0)*dFx+dCx, p_vecPoint(1)*dFy+dCy );
    }
    static const double getDenormalized( const double& dCoordinate, const double& dF, const double& dC )
    {
        return dCoordinate*dF+dC;
    }
    static const double getNormalized( const double& dCoordinate, const double& dF, const double& dC )
    {
        return (dCoordinate-dC)/dF;
    }

    static const Eigen::Matrix3d fromOrientationRodrigues( const Eigen::Vector3d& p_vecOrientation );
    static const Eigen::Quaterniond fromEulerAngles( const Eigen::Vector3d& p_vecEulerAngles );

    static const Eigen::Vector2d getPointUndistorted( const Eigen::Vector2i& p_vecPointDistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients );
    static const Eigen::Vector2d getPointUndistorted( const Eigen::Vector2d& p_vecPointDistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients );
    static const Eigen::Vector2d getPointUndistortedPlumbBob( const Eigen::Vector2i& p_vecPointDistorted, const Eigen::Matrix< double, 3, 4 >& p_matProjection, const Eigen::Vector4d& p_vecDistortionCoefficients );
    static const Eigen::Vector2d getPointUndistortedPlumbBob( const Eigen::Vector2d& p_vecPointDistorted, const Eigen::Matrix< double, 3, 4 >& p_matProjection, const Eigen::Vector4d& p_vecDistortionCoefficients );
    static const Eigen::Vector2d getPointDistortedPlumbBob( const Eigen::Vector2d& p_vecPointUndistorted, const Eigen::Vector2d& p_vecPointPrincipal, const Eigen::Vector4d& p_vecDistortionCoefficients );
    static const Eigen::Vector2d getPointDistorted( const Eigen::Vector2d& p_vecPointUndistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients );

    static const Eigen::Matrix3d getSkew( const Eigen::Vector3d& p_vecVector );
    static const Eigen::Matrix3d getEssential( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo );
    static const Eigen::Matrix3d getEssentialPrecomputed( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationToInverse );
    static const Eigen::Matrix3d getFundamental( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo, const Eigen::Matrix3d& p_matIntrinsic );

    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationSVDLS( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight );
    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationQRLS( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight );
    static const Eigen::Vector4d getPointHomogeneousStereoLinearTriangulationLU( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight );
    static const double getEpipolarSquaredNormDistance( const Eigen::Vector2d& p_vecPointFrom, const Eigen::Vector2d& p_vecPointTo, const Eigen::Matrix3d& p_matEssential );

    template< uint32_t uRows, uint32_t uCols > static const Eigen::Matrix< double, uRows, uCols > getGaussianWeighted( const Eigen::Matrix< uint8_t, uRows, uCols >& p_matInput, const double& p_dSigma )
    {
        //ds allocate eigen matrix
        Eigen::Matrix< double, uRows, uCols > matGaussianWeighted;

        //ds offset (for gaussian)
        const uint32_t uOffset( uRows/2 );

        //ds fill the matrix (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            for( uint32_t v = 0; v < uCols; ++v )
            {
                //ds get shifted coordinates
                const int32_t iX( u-uOffset );
                const int32_t iY( v-uOffset );

                matGaussianWeighted( u, v ) = std::exp( -(iX*iX)/p_dSigma )*std::exp( -(iY*iY)/p_dSigma )*p_matInput( u, v );
            }
        }

        return matGaussianWeighted;
    }

    static const double getTransformationDelta( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo );

//ds wrapping functions
public:

    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationSVDLS( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight );
    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationQRLS( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight );
    static const Eigen::Vector4d getPointHomogeneousStereoLinearTriangulationLU( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight );

    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationSVDLS( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT );
    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationQRLS( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT );
    static const CPoint3DInCameraFrame getPointStereoLinearTriangulationSVDDLT( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT );
};

#endif //#define CMINIVISIONTOOLBOX_H_
