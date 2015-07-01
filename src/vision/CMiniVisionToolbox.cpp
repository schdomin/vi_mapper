#include "CMiniVisionToolbox.h"
#include "utility/CWrapperOpenCV.h"

const cv::Point2i CMiniVisionToolbox::getPointBoundarized( const int32_t& p_iPointX, const int32_t& p_iPointY, const uint32_t& p_uImageRows, const uint32_t& p_uImageCols )
{
    //ds point to set
    cv::Point2i pPointFixed( p_iPointX, p_iPointY );

    //ds check coordinates against current setting
    if( 0 > p_iPointX ){ pPointFixed.x = 0; }
    else if( static_cast< int32_t >( p_uImageCols ) < p_iPointX ){ pPointFixed.x = static_cast< int32_t >( p_uImageCols ); }
    if( 0 > p_iPointY ){ pPointFixed.y = 0; }
    else if( static_cast< int32_t >( p_uImageRows ) < p_iPointY ){ pPointFixed.y = static_cast< int32_t >( p_uImageRows ); }

    //ds return fixed point
    return pPointFixed;
}

const Eigen::Matrix3d CMiniVisionToolbox::fromOrientationRodrigues( const Eigen::Vector3d& p_vecOrientation )
{
    //ds orientation matrix
    cv::Mat matOrientation;

    //ds fill the matrix
    cv::Rodrigues( CWrapperOpenCV::toCVVector< double, 3 >( p_vecOrientation ), matOrientation );

    //ds return in Eigen format
    return CWrapperOpenCV::fromCVMatrix< double, 3, 3 >( matOrientation );
}

const Eigen::Quaterniond CMiniVisionToolbox::fromEulerAngles( const Eigen::Vector3d& p_vecEulerAngles )
{
    //ds allocate a quaternion
    Eigen::Quaterniond vecOrientationQuaternion;

    //ds precompute
    const double dAlphaHalf( p_vecEulerAngles(0)/2 );
    const double dBetaHalf( p_vecEulerAngles(1)/2 );
    const double dGammaHalf( p_vecEulerAngles(2)/2 );

    //ds fill it: http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    vecOrientationQuaternion.x( ) = std::cos( dAlphaHalf )*std::cos( dBetaHalf )*std::cos( dGammaHalf )
                                   +std::sin( dAlphaHalf )*std::sin( dBetaHalf )*std::sin( dGammaHalf );
    vecOrientationQuaternion.y( ) = std::sin( dAlphaHalf )*std::cos( dBetaHalf )*std::cos( dGammaHalf )
                                   -std::cos( dAlphaHalf )*std::sin( dBetaHalf )*std::sin( dGammaHalf );
    vecOrientationQuaternion.z( ) = std::cos( dAlphaHalf )*std::sin( dBetaHalf )*std::cos( dGammaHalf )
                                   +std::sin( dAlphaHalf )*std::cos( dBetaHalf )*std::sin( dGammaHalf );
    vecOrientationQuaternion.w( ) = std::cos( dAlphaHalf )*std::cos( dBetaHalf )*std::sin( dGammaHalf )
                                   -std::sin( dAlphaHalf )*std::sin( dBetaHalf )*std::cos( dGammaHalf );

    return vecOrientationQuaternion;
}

const Eigen::Vector2d CMiniVisionToolbox::getPointUndistorted( const Eigen::Vector2i& p_vecPointDistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    //ds compute radius
    const double dDeltaX( ( p_vecPointDistorted(0)-p_vecPrincipalPoint(0) )/752 );
    const double dDeltaY( ( p_vecPointDistorted(1)-p_vecPrincipalPoint(1) )/480 );
    const double dRadius2( dDeltaX*dDeltaX + dDeltaX*dDeltaX );
    const double dRadius( std::sqrt( dRadius2 ) );
    const double dRadius3( dRadius2*dRadius );
    const double dRadius4( dRadius3*dRadius );

    //ds compute distortion factor
    const double dFactor = 1.0 + p_vecDistortionCoefficients(0)*dRadius + p_vecDistortionCoefficients(1)*dRadius2 + p_vecDistortionCoefficients(2)*dRadius3 + p_vecDistortionCoefficients(3)*dRadius4;

    //ds compute corrected point and return
    return Eigen::Vector2d( p_vecPrincipalPoint(0) + dFactor*dDeltaX*752, p_vecPrincipalPoint(1) + dFactor*dDeltaY*480 );
}

const Eigen::Vector2d CMiniVisionToolbox::getPointUndistorted( const Eigen::Vector2d& p_vecPointDistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    //ds compute radius
    const double dDeltaX( ( p_vecPointDistorted(0)-p_vecPrincipalPoint(0) )/752 );
    const double dDeltaY( ( p_vecPointDistorted(1)-p_vecPrincipalPoint(1) )/480 );
    const double dRadius2( dDeltaX*dDeltaX + dDeltaY*dDeltaY );
    const double dRadius( std::sqrt( dRadius2 ) );
    const double dRadius3( dRadius2*dRadius );
    const double dRadius4( dRadius3*dRadius );

    //ds compute distortion factor
    const double dFactor = 1.0 + p_vecDistortionCoefficients(0)*dRadius + p_vecDistortionCoefficients(1)*dRadius2 + p_vecDistortionCoefficients(2)*dRadius3 + p_vecDistortionCoefficients(3)*dRadius4;

    //ds compute corrected point and return
    return Eigen::Vector2d( p_vecPrincipalPoint(0) + dFactor*dDeltaX*752, p_vecPrincipalPoint(1) + dFactor*dDeltaY*480 );
}

const Eigen::Vector2d CMiniVisionToolbox::getPointUndistortedPlumbBob( const Eigen::Vector2i& p_vecPointDistorted, const Eigen::Matrix< double, 3, 4 >& p_matProjection, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    std::cout << p_matProjection << std::endl;

    //ds compute radius (normalized)
    const double dDistortedX( p_vecPointDistorted(0)/752.0 );
    const double dDistortedY( p_vecPointDistorted(1)/480.0 );
    const double dRadius2( dDistortedX*dDistortedX + dDistortedY*dDistortedY );
    const double dRadius4( dRadius2*dRadius2 );

    //ds compute radial distortion
    const double dDistortionRadial( 1+p_vecDistortionCoefficients(0)*dRadius2+p_vecDistortionCoefficients(1)*dRadius4 );

    //ds compute tangential distortion
    const double dDistortionXTangential( 2*p_vecDistortionCoefficients(2)*dDistortedX*dDistortedY + p_vecDistortionCoefficients(3)*( dRadius2+2*dDistortedX*dDistortedX ) );
    const double dDistortionYTangential( p_vecDistortionCoefficients(2)*( dRadius2+2*dDistortedY*dDistortedY ) + 2*p_vecDistortionCoefficients(3)*dDistortedX*dDistortedY );

    //ds compute undistorted coordinates
    const double dX( dDistortionRadial*dDistortedX+dDistortionXTangential );
    const double dY( dDistortionRadial*dDistortedY+dDistortionYTangential );

    //ds compute corrected point and return
    return Eigen::Vector2d( dX*p_matProjection(0,0)+p_matProjection(0,2), dY*p_matProjection(1,1)+p_matProjection(1,2) );
}

const Eigen::Vector2d CMiniVisionToolbox::getPointUndistortedPlumbBob( const Eigen::Vector2d& p_vecPointDistorted, const Eigen::Matrix< double, 3, 4 >& p_matProjection, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    //ds compute radius (normalized)
    const double dDistortedX( p_vecPointDistorted(0) );
    const double dDistortedY( p_vecPointDistorted(1) );
    const double dRadius2( dDistortedX*dDistortedX + dDistortedY*dDistortedY );
    const double dRadius4( dRadius2*dRadius2 );

    //ds compute radial distortion
    const double dDistortionRadial( 1+p_vecDistortionCoefficients(0)*dRadius2+p_vecDistortionCoefficients(1)*dRadius4 );

    //ds compute tangential distortion
    const double dDistortionXTangential( 2*p_vecDistortionCoefficients(2)*dDistortedX*dDistortedY + p_vecDistortionCoefficients(3)*( dRadius2+2*dDistortedX*dDistortedX ) );
    const double dDistortionYTangential( p_vecDistortionCoefficients(2)*( dRadius2+2*dDistortedY*dDistortedY ) + 2*p_vecDistortionCoefficients(3)*dDistortedX*dDistortedY );

    //ds compute undistorted coordinates
    const double dX( dDistortionRadial*dDistortedX+dDistortionXTangential );
    const double dY( dDistortionRadial*dDistortedY+dDistortionYTangential );

    //ds compute corrected point and return
    return Eigen::Vector2d( dX*p_matProjection(0,0)+p_matProjection(0,2), dY*p_matProjection(1,1)+p_matProjection(1,2) );
}

const Eigen::Vector2d CMiniVisionToolbox::getPointDistortedPlumbBob( const Eigen::Vector2d& p_vecPointUndistorted, const Eigen::Vector2d& p_vecPointPrincipal, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    const int32_t iUndistortedX( std::abs( p_vecPointUndistorted(0) ) );
    const int32_t iUndistortedY( std::abs( p_vecPointUndistorted(1) ) );

    //ds current ranges
    const int32_t iXStart( iUndistortedX-40 );
    const int32_t iXStop( iUndistortedX+40 );
    const int32_t iYStart( iUndistortedY-40 );
    const int32_t iYStop( iUndistortedY+40 );

    double dErrorLowest( 10.0 );
    Eigen::Vector2d vecPointDistorted( 0, 0 );

    //ds loop around the current point
    for( int32_t i = iXStart; i < iXStop; ++i )
    {
        for( int32_t j = iYStart; j < iYStop; ++j )
        {
            /*ds get current error
            const Eigen::Vector2d vecError( p_vecPointUndistorted-CMiniVisionToolbox::getPointUndistortedPlumbBob( Eigen::Vector2i( iUndistortedX, iUndistortedY ), p_vecPointPrincipal, p_vecDistortionCoefficients ) );

            //ds error
            const double dErrorCurrent( std::fabs( vecError(0) ) + std::fabs( vecError(1) ) );

            //ds if the error is lower
            if( dErrorLowest > dErrorCurrent )
            {
                vecPointDistorted = Eigen::Vector2d( iUndistortedX, iUndistortedY );
            }*/
        }
    }

    std::printf( "error: %f\n", dErrorLowest );

    return vecPointDistorted;
}

const Eigen::Vector2d CMiniVisionToolbox::getPointDistorted( const Eigen::Vector2d& p_vecPointUndistorted, const Eigen::Vector2d& p_vecPrincipalPoint, const Eigen::Vector4d& p_vecDistortionCoefficients )
{
    //ds transform to 1:1 aspect ratio
    const Eigen::Vector2d vecPointUndistorted( p_vecPointUndistorted(0)/752, p_vecPointUndistorted(1)/480 );
    const Eigen::Vector2d vecPrincipalPoint( p_vecPrincipalPoint(0)/752, p_vecPrincipalPoint(1)/480 );

    //ds get distances
    const double dDeltaX( vecPointUndistorted(0)-vecPrincipalPoint(0) );
    const double dDeltaY( vecPointUndistorted(1)-vecPrincipalPoint(1) );

    //ds if not on focal point
    if( 0.0 != dDeltaX )
    {
        //ds compute coefficients
        const double dFraction( dDeltaY/dDeltaX );
        const double dEpsilon( 1 + dFraction*dFraction );
        const double dSqrtEpsilon( std::sqrt( dEpsilon ) );
        const double dA( p_vecDistortionCoefficients(0)*dSqrtEpsilon );
        const double dB( p_vecDistortionCoefficients(1)*dEpsilon );
        const double dC( p_vecDistortionCoefficients(2)*dSqrtEpsilon*dSqrtEpsilon*dSqrtEpsilon );
        const double dD( p_vecDistortionCoefficients(3)*dEpsilon*dEpsilon );
        const double dE( vecPointUndistorted(0)-vecPrincipalPoint(0) );

        //ds sample resolution in aspect ratio at unity (from 0 to 1)
        const double dResolution( 0.001 );
        const uint64_t uSteps( 1.0/dResolution );
        double dErrorLowest( 1.0 );
        double dUBest( 0.0 );
        double dLBest( 0.0 );

        //ds sample a solution
        for( uint64_t u = 1; u < uSteps; ++u )
        {
            //ds set current sample
            const double dU( u*dResolution-vecPrincipalPoint(0) );
            const double dU2( dU*dU );
            const double dU3( dU2*dU );
            const double dU4( dU3*dU );
            const double dL( 1 + dA*dU + dB*dU2 + dC*dU3 + dD*dU4 );

            //ds compute error
            const double dError( std::fabs( dU*dL-dE ) );

            //ds if the error is lower pick the sample
            if( dErrorLowest > dError )
            {
                dErrorLowest = dError;
                dUBest       = dU;
                dLBest       = dL;
            }
        }

        //ds compute x (since u=x-x_c -> x=u+x_c)
        const double dDistortedX( dUBest+vecPrincipalPoint(0) );
        const double dDistortedY( vecPrincipalPoint(1) + dFraction*dUBest );

        const double dUndistortedX( vecPrincipalPoint(0) + dLBest*dUBest );

        std::printf( "error: %f\n", dErrorLowest );
        std::printf( "undistorted: %f %f\n", vecPointUndistorted(0)*752, vecPointUndistorted(1)*480 );
        std::printf( "distorted: %f %f\n", dDistortedX*752, dDistortedY*480 );

        const Eigen::Vector2d vecUndistortedCheck( CMiniVisionToolbox::getPointUndistorted( Eigen::Vector2d( dDistortedX*752, dDistortedY*480 ), p_vecPrincipalPoint, p_vecDistortionCoefficients ) );

        std::printf( "redistorted: %f %f\n", vecUndistortedCheck(0), vecUndistortedCheck(1) );
        std::printf( "redistorted X: %f\n", dUndistortedX*752 );

        return Eigen::Vector2d( dDistortedX*752, dDistortedY*480 );

        /*ds compute quadratic coefficients
        const double dFraction( dDeltaY/dDeltaX );
        const double dEpsilon( p_vecDistortionCoefficients(0)*std::sqrt( 1 + dFraction*dFraction ) );
        const double dA( dEpsilon );
        const double dB( 1 - 2*dEpsilon*vecPrincipalPoint(0) );
        const double dC( dEpsilon*vecPrincipalPoint(0)*vecPrincipalPoint(0) - vecPointUndistorted(0) );

        //ds squared element
        const double dRootContent( dB*dB - 4*dA*dC );

        if( 0 <= dRootContent )
        {
            //ds compute solutions
            const double dDistortedX1( 1/( 2*dA )*( -dB+std::sqrt( dRootContent ) ) );
            const double dDistortedX2( 1/( 2*dA )*( -dB-std::sqrt( dRootContent ) ) );

            //ds compute y coordinates
            const double dDistortedY1( vecPrincipalPoint(1) + dFraction*( dDistortedX1-vecPrincipalPoint(0) ) );
            const double dDistortedY2( vecPrincipalPoint(1) + dFraction*( dDistortedX2-vecPrincipalPoint(0) ) );

            //ds get distortion free
            const Eigen::Vector2d vecUndistortedCheck( CMiniVisionToolbox::getPointUndistorted( Eigen::Vector2d( dDistortedX1*752, dDistortedY1*480 ), p_vecPrincipalPoint, p_vecDistortionCoefficients ) );

            std::printf( "undistorted: %f %f\n", vecPrincipalPoint(0)*752, vecPrincipalPoint(1)*480 );
            std::printf( "distorted 1: %f %f\n", dDistortedX1*752, dDistortedY1*480 );
            std::printf( "distorted 2: %f %f\n", dDistortedX2*752, dDistortedY2*480 );
            std::printf( "redistorted: %f %f\n", vecUndistortedCheck(0), vecUndistortedCheck(1) );

            return Eigen::Vector2d( dDistortedX1*752, dDistortedY1*480 );
        }
        else
        {
            //ds compute solutions (real parts)
            const double dDistortedX( 1/( 2*dA )*( -dB ) );
            const double dDistortedY( vecPrincipalPoint(1) + dFraction*( dDistortedX-vecPrincipalPoint(0) ) );
            std::printf( "undistorted: %f %f\n", vecPrincipalPoint(0)*752, vecPrincipalPoint(1)*480 );
            std::printf( "distorted IMAGINARY: %f %f\n", dDistortedX*752, dDistortedY*480 );
            return Eigen::Vector2d( dDistortedX*752, dDistortedY*480 );
        }*/
    }
    else
    {
        return Eigen::Vector2d( p_vecPointUndistorted(0), p_vecPointUndistorted(1) );
    }
}

const Eigen::Matrix3d CMiniVisionToolbox::getSkew( const Eigen::Vector3d& p_vecVector )
{
    //ds skew matrix
    Eigen::Matrix3d matSkew;

    //ds formula: http://en.wikipedia.org/wiki/Cross_product#Skew-symmetric_matrix
    matSkew <<             0.0, -p_vecVector(2),  p_vecVector(1),
                p_vecVector(2),             0.0, -p_vecVector(0),
               -p_vecVector(1),  p_vecVector(0),             0.0;

    return matSkew;
}

const Eigen::Matrix3d CMiniVisionToolbox::getEssential( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo )
{
    //ds compute essential matrix: http://en.wikipedia.org/wiki/Essential_matrix TODO check math!
    const Eigen::Isometry3d matTransformation( p_matTransformationTo.inverse( )*p_matTransformationFrom );
    //const Eigen::Isometry3d matTransformation( p_matTransformationFrom.inverse( )*p_matTransformationTo );
    //const Eigen::Isometry3d matTransformation( p_matTransformationTo*p_matTransformationFrom.inverse( ) );
    const Eigen::Matrix3d matSkewTranslation( CMiniVisionToolbox::getSkew( matTransformation.translation( ) ) );

    //ds compute essential matrix
    return matTransformation.linear( )*matSkewTranslation;
}

const Eigen::Matrix3d CMiniVisionToolbox::getEssentialPrecomputed( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationToInverse )
{
    //ds compute essential matrix: http://en.wikipedia.org/wiki/Essential_matrix TODO check math!
    const Eigen::Isometry3d matTransformation( p_matTransformationToInverse*p_matTransformationFrom );
    const Eigen::Matrix3d matSkewTranslation( CMiniVisionToolbox::getSkew( matTransformation.translation( ) ) );

    //ds compute essential matrix
    return matTransformation.linear( )*matSkewTranslation;
}

const Eigen::Matrix3d CMiniVisionToolbox::getFundamental( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo, const Eigen::Matrix3d& p_matIntrinsic )
{
    //ds get essential matrix
    const Eigen::Matrix3d matEssential( CMiniVisionToolbox::getEssential( p_matTransformationFrom, p_matTransformationTo ) );

    //ds compute fundamental matrix: http://en.wikipedia.org/wiki/Fundamental_matrix_%28computer_vision%29
    const Eigen::Matrix3d matIntrinsicInverse( p_matIntrinsic.inverse( ) );
    return matIntrinsicInverse.transpose( )*matEssential*matIntrinsicInverse;
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix< double, 4 , 4 > matAHomogeneous;

    //ds fill the matrix
    matAHomogeneous.row(0) = p_vecPointLeft(0)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(0);
    matAHomogeneous.row(1) = p_vecPointLeft(1)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(1);
    matAHomogeneous.row(2) = p_vecPointRight(0)*p_matProjectionRight.row(2)-p_matProjectionRight.row(0);
    matAHomogeneous.row(3) = p_vecPointRight(1)*p_matProjectionRight.row(2)-p_matProjectionRight.row(1);

    //ds inhomogeneous solution
    const Eigen::Matrix< double, 4, 3 > matAInhomogeneous( matAHomogeneous.block< 4, 3 >( 0, 0 ) );
    const Eigen::Vector4d vecRHS( -matAHomogeneous.col( 3 ) );

    //ds solve the system and return
    return matAInhomogeneous.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV ).solve( vecRHS );
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix< double, 4 , 4 > matAHomogeneous;

    //ds fill the matrix
    matAHomogeneous.row(0) = p_vecPointLeft(0)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(0);
    matAHomogeneous.row(1) = p_vecPointLeft(1)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(1);
    matAHomogeneous.row(2) = p_vecPointRight(0)*p_matProjectionRight.row(2)-p_matProjectionRight.row(0);
    matAHomogeneous.row(3) = p_vecPointRight(1)*p_matProjectionRight.row(2)-p_matProjectionRight.row(1);

    //ds inhomogeneous solution
    const Eigen::Matrix< double, 4, 3 > matAInhomogeneous( matAHomogeneous.block< 4, 3 >( 0, 0 ) );
    const Eigen::Vector4d vecRHS( -matAHomogeneous.col( 3 ) );

    //ds solve the system and return
    return matAInhomogeneous.fullPivHouseholderQr( ).solve( vecRHS );
}

const Eigen::Vector4d CMiniVisionToolbox::getPointHomogeneousStereoLinearTriangulationLU( const Eigen::Vector2d& p_vecPointLeft, const Eigen::Vector2d& p_vecPointRight, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLeft, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRight )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix< double, 4 , 4 > matA;

    //ds fill the matrix
    matA.row(0) = p_vecPointLeft(0)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(0);
    matA.row(1) = p_vecPointLeft(1)*p_matProjectionLeft.row(2)-p_matProjectionLeft.row(1);
    matA.row(2) = p_vecPointRight(0)*p_matProjectionRight.row(2)-p_matProjectionRight.row(0);
    matA.row(3) = p_vecPointRight(1)*p_matProjectionRight.row(2)-p_matProjectionRight.row(1);

    //ds homogeneous solution
    return matA.fullPivLu( ).kernel( );
}

const double CMiniVisionToolbox::getEpipolarSquaredNormDistance( const Eigen::Vector2d& p_vecPointFrom, const Eigen::Vector2d& p_vecPointTo, const Eigen::Matrix3d& p_matEssential )
{
    //ds first product E*p
    const Eigen::Vector3d vecCurveFactors( p_matEssential*CMiniVisionToolbox::toHomo( p_vecPointFrom ) );

    //ds compute denomiator for normalization
    const double dDenominatorFactors( vecCurveFactors.squaredNorm( ) );

    //ds if the denominator is non-zero
    if( 0.0 < dDenominatorFactors )
    {
        //ds get the essential distance
        const double dEssentialDistance( CMiniVisionToolbox::toHomo( p_vecPointTo ).transpose( )*vecCurveFactors );

        //ds return squared norm
        return dEssentialDistance*dEssentialDistance/dDenominatorFactors;
    }
    else
    {
        //ds perfect match (result of E*p=0 -> pT*E*p=0)
        return 0.0;
    }
}

const double CMiniVisionToolbox::getTransformationDelta( const Eigen::Isometry3d& p_matTransformationFrom, const Eigen::Isometry3d& p_matTransformationTo )
{
    //ds compute pose change
    const Eigen::Isometry3d matTransformChange( p_matTransformationTo*p_matTransformationFrom.inverse( ) );

    //ds check point
    const Eigen::Vector4d vecSamplePoint( 40.2, -1.25, 2.5, 1.0 );
    double dNorm( vecSamplePoint.norm( ) );
    const Eigen::Vector4d vecDifference( vecSamplePoint-matTransformChange*vecSamplePoint );
    dNorm = ( dNorm + vecDifference.norm( ) )/2;

    //ds return norm
    return ( std::fabs( vecDifference(0) ) + std::fabs( vecDifference(1) ) + std::fabs( vecDifference(2) ) )/dNorm;
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight )
{
    //ds wrap the input
    return CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointLeft ),
                                                                     CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointRight ),
                                                                     CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionLeft ),
                                                                     CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionRight ) );
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight )
{
    //ds wrap the input
    return CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointLeft ),
                                                                             CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointRight ),
                                                                             CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionLeft ),
                                                                             CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionRight ) );
}

const Eigen::Vector4d CMiniVisionToolbox::getPointHomogeneousStereoLinearTriangulationLU( const cv::Point2d& p_vecPointLeft, const cv::Point2d& p_vecPointRight, const cv::Mat& p_matProjectionLeft, const cv::Mat& p_matProjectionRight )
{
    //ds wrap the input
    return CMiniVisionToolbox::getPointHomogeneousStereoLinearTriangulationLU( CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointLeft ),
                                                                             CWrapperOpenCV::fromCVVector< double, 2 >( p_vecPointRight ),
                                                                             CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionLeft ),
                                                                             CWrapperOpenCV::fromCVMatrix< double, 3, 4 >( p_matProjectionRight ) );
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix< double, 4 , 4 > matAHomogeneous;

    //ds fill the matrix
    matAHomogeneous.row(0) = p_ptPointLEFT.x*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(0);
    matAHomogeneous.row(1) = p_ptPointLEFT.y*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(1);
    matAHomogeneous.row(2) = p_ptPointRIGHT.x*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(0);
    matAHomogeneous.row(3) = p_ptPointRIGHT.y*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(1);

    //ds inhomogeneous solution
    const Eigen::Matrix< double, 4, 3 > matAInhomogeneous( matAHomogeneous.block< 4, 3 >( 0, 0 ) );
    const Eigen::Vector4d vecRHS( -matAHomogeneous.col( 3 ) );

    //ds solve the system and return
    return matAInhomogeneous.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV ).solve( vecRHS );
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationQRLS( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix< double, 4 , 4 > matAHomogeneous;

    //ds fill the matrix
    matAHomogeneous.row(0) = p_ptPointLEFT.x*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(0);
    matAHomogeneous.row(1) = p_ptPointLEFT.y*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(1);
    matAHomogeneous.row(2) = p_ptPointRIGHT.x*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(0);
    matAHomogeneous.row(3) = p_ptPointRIGHT.y*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(1);

    //ds inhomogeneous solution
    const Eigen::Matrix< double, 4, 3 > matAInhomogeneous( matAHomogeneous.block< 4, 3 >( 0, 0 ) );
    const Eigen::Vector4d vecRHS( -matAHomogeneous.col( 3 ) );

    //ds solve the system and return
    return matAInhomogeneous.fullPivHouseholderQr( ).solve( vecRHS );
}

const CPoint3DInCameraFrame CMiniVisionToolbox::getPointStereoLinearTriangulationSVDDLT( const cv::Point2d& p_ptPointLEFT, const cv::Point2d& p_ptPointRIGHT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionLEFT, const Eigen::Matrix< double, 3, 4 >& p_matProjectionRIGHT )
{
    //ds A matrix for system: A*X=0
    Eigen::Matrix4d matAHomogeneous;

    matAHomogeneous.row(0) = p_ptPointLEFT.x*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(0);
    matAHomogeneous.row(1) = p_ptPointLEFT.y*p_matProjectionLEFT.row(2)-p_matProjectionLEFT.row(1);
    matAHomogeneous.row(2) = p_ptPointRIGHT.x*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(0);
    matAHomogeneous.row(3) = p_ptPointRIGHT.y*p_matProjectionRIGHT.row(2)-p_matProjectionRIGHT.row(1);

    //ds solve system subject to ||A*x||=0 ||x||=1
    const Eigen::JacobiSVD< Eigen::Matrix4d > matSVD( matAHomogeneous, Eigen::ComputeFullU | Eigen::ComputeFullV );

    //ds solution x is the last column of V
    const Eigen::Vector4d vecX( matSVD.matrixV( ).col( 3 ) );

    return vecX.head( 3 )/vecX(3);
}
