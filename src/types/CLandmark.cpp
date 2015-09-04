#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptorLEFT,
           const CDescriptor& p_matDescriptorRIGHT,
           const double& p_dKeyPointSize,
           const CPoint3DWORLD& p_vecPointXYZ,
           const CPoint2DHomogenized& p_vecUVLEFTReference,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DCAMERA& p_vecPointXYZCAMERA,
           const CPoint3DWORLD& p_vecCameraPosition,
           const Eigen::Vector3d& p_vecCameraOrientation,
           //const Eigen::Matrix3d& p_matKRotation,
           //const Eigen::Vector3d& p_vecKTranslation,
           const MatrixProjection& p_matProjectionWORLDtoLEFT,
           const UIDFrame& p_uFrame ): uID( p_uID ),
                                                       matDescriptorReferenceLEFT( p_matDescriptorLEFT ),
                                                       matDescriptorLASTLEFT( p_matDescriptorLEFT ),
                                                       matDescriptorLASTRIGHT( p_matDescriptorRIGHT ),
                                                       dKeyPointSize( p_dKeyPointSize ),
                                                       vecPointXYZInitial( p_vecPointXYZ ),
                                                       vecPointXYZOptimized( vecPointXYZInitial ),
                                                       vecUVLEFTReference( p_vecUVLEFTReference ),
                                                       vecUVReferenceLEFT( p_ptUVLEFT.x, p_ptUVLEFT.y, 1.0 ),
                                                       vecPointXYZMean( p_vecPointXYZ ),
                                                       bIsCurrentlyVisible( false ),
                                                       m_dDepthLastOptimizationMeters( p_vecPointXYZCAMERA.z( ) ),
                                                       m_vecCameraPositionLAST( p_vecCameraPosition ),
                                                       m_vecCameraOrientationAccumulated( 0.0, 0.0, 0.0 )
{
    vecDescriptorsLEFT.clear( );
    m_vecMeasurements.clear( );

    //ds construct filestring and open dump file
    //char chBuffer[256];
    //std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    //m_pFilePositionOptimization = std::fopen( chBuffer, "w" );

    //assert( 0 != m_pFilePositionOptimization );

    //ds dump file format
    //std::fprintf( m_pFilePositionOptimization, "ID_FRAME | ID_LANDMARK | ITERATION MEASUREMENTS INLIERS | ERROR_ARSS | DELTA_XYZ |      X      Y      Z\n" );

    //ds add this position
    addMeasurement( p_uFrame, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZCAMERA, vecPointXYZInitial, p_vecCameraPosition, p_vecCameraOrientation, p_matProjectionWORLDtoLEFT, p_matDescriptorLEFT );
}

CLandmark::~CLandmark( )
{
    //ds close file
    //std::fclose( m_pFilePositionOptimization );

    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        delete pMeasurement;
    }
}

void CLandmark::addMeasurement( const UIDFrame& p_uFrame,
                                const cv::Point2d& p_ptUVLEFT,
                                const cv::Point2d& p_ptUVRIGHT,
                                const CPoint3DCAMERA& p_vecXYZLEFT,
                                const CPoint3DWORLD& p_vecXYZWORLD,
                                const CPoint3DWORLD& p_vecCameraPosition,
                                const Eigen::Vector3d& p_vecCameraOrientation,
                                //const Eigen::Matrix3d& p_matKRotation,
                                //const Eigen::Vector3d& p_vecKTranslation,
                                const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                const CDescriptor& p_matDescriptorLEFT )
{
    //ds input validation
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );
    assert( 0 < p_vecXYZLEFT.z( ) );

    //ds update mean
    vecPointXYZMean = ( vecPointXYZMean+p_vecXYZWORLD )/2.0;

    //ds accumulate orientation
    m_vecCameraOrientationAccumulated += p_vecCameraOrientation;

    //ds check if we can recalibrate the 3d position
    if( CLandmark::m_dDistanceDeltaForOptimizationMeters < ( m_vecCameraPositionLAST-p_vecCameraPosition ).squaredNorm( ) ||
        CLandmark::m_dAngleDeltaForOptimizationRadians < m_vecCameraOrientationAccumulated.squaredNorm( ) )
    {
        //ds update last
        m_vecCameraPositionLAST           = p_vecCameraPosition;
        m_vecCameraOrientationAccumulated = Eigen::Vector3d::Zero( );

        //ds get calibrated point
        //vecPointXYZOptimized = _getOptimizedLandmarkKLMA( p_uFrame, vecPointXYZOptimized );
        //vecPointXYZCalibrated = _getOptimizedLandmarkIDLMA( p_uFrame, vecMeanMeasurement );
        vecPointXYZOptimized = _getOptimizedLandmarkKRDLMA( p_uFrame, vecPointXYZOptimized );
        //vecPointXYZCalibrated = _getOptimizedLandmarkIDWA( );
    }

    //ds register descriptor
    vecDescriptorsLEFT.push_back( p_matDescriptorLEFT );

    //ds add the measurement to structure
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID, p_ptUVLEFT, p_ptUVRIGHT, p_vecXYZLEFT, p_vecXYZWORLD, vecPointXYZOptimized, p_vecCameraPosition, p_matProjectionWORLDtoLEFT ) );
}

void CLandmark::optimize( const UIDFrame& p_uFrame )
{
    //ds update position
    //vecPointXYZOptimized = _getOptimizedLandmarkKLMA( p_uFrame, vecPointXYZOptimized );
    vecPointXYZOptimized = _getOptimizedLandmarkKRDLMA( p_uFrame, vecPointXYZOptimized );
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkKLMA( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( 0.0, 0.0, 0.0, 0.0 );
    CPoint3DInWorldFrameHomogenized vecX( p_vecInitialGuess.x( ), p_vecInitialGuess.y( ), p_vecInitialGuess.z( ), 1.0 );
    Eigen::Matrix2d matOmega( Eigen::Matrix2d::Identity( ) );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::m_uCapIterations; ++uIteration )
    {
        //ds counts
        double dErrorSquaredTotalPixels = 0.0;
        uint32_t uInliers               = 0;

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds current element
            const double dUReference( pMeasurement->ptUVLEFT.x );
            const double dVReference( pMeasurement->ptUVLEFT.y );

            //ds compute projection
            const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*vecX );
            const double dA( vecProjectionHomogeneous(0) );
            const double dB( vecProjectionHomogeneous(1) );
            const double dC( vecProjectionHomogeneous(2) );
            const double dCSquared( dC*dC );

            //ds compute current error
            const Eigen::Vector2d vecError( static_cast< double >( dA/dC )-dUReference, static_cast< double >( dB/dC )-dVReference );

            //ds depth affects disparity (e.g x distance)
            assert( 0.0 < pMeasurement->vecPointXYZLEFT.z( ) );
            matOmega(0,0) = 1.0/pMeasurement->vecPointXYZLEFT.z( );

            //ds kernelized LS TODO: adjust graveness for distant points
            const double dErrorSquaredPixels = vecError.transpose( )*matOmega*vecError;
            double dContribution( 1.0 );
            if( CLandmark::m_dKernelMaximumError < dErrorSquaredPixels )
            {
                //ds safe since if squared error would was zero it would not enter the loop (assuming maximum error is bigger than zero)
                dContribution = CLandmark::m_dKernelMaximumError/dErrorSquaredPixels;
            }
            else
            {
                //ds inlier
                ++uInliers;
            }
            dErrorSquaredTotalPixels += dContribution*dErrorSquaredPixels;

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f (squared: %4.2f)\n", uID, uIteration, vecError.x( ), vecError.y( ), dSquaredError );

            //ds compute jacobian
            Eigen::Matrix< double, 2, 4 > matJacobian;
            matJacobian.row(0) = Eigen::Vector3d( 1/dC, 0, -dA/dCSquared ).transpose( )*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.row(1) = Eigen::Vector3d( 0, 1/dC, -dB/dCSquared ).transpose( )*pMeasurement->matProjectionWORLDtoLEFT;
            const Eigen::Matrix< double, 4, 2 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds compute H and B
            matH += dContribution*matJacobianTransposed*matOmega*matJacobian + CLandmark::m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
            vecB += dContribution*matJacobianTransposed*matOmega*vecError;
        }

        //ds solve constrained system (since dx(3) = 0.0)
        const CPoint3DWORLD vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

        //ds update x solution
        vecX.block< 3, 1 >(0,0) += vecDeltaX;

        //ds solve system H*x=-b -> shift the minus to delta addition
        //const CPoint3DInWorldFrameHomogenized vecDeltaX( matH.householderQr( ).solve( vecB ) );

        //ds update x solution (take the minus here)
        //vecX += vecDeltaX;

        //ds homogenize
        //vecX /= vecX(3);

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ) );

        //std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( CLandmark::m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            //ds compute average error
            const double dErrorSquaredAveragePixels = dErrorSquaredTotalPixels/m_vecMeasurements.size( );

            //ds if acceptable
            if( CLandmark::m_dMaximumErrorSquaredAveragePixels > dErrorSquaredAveragePixels && m_uMinimumInliers < uInliers )
            {
                //ds success
                ++uOptimizationsSuccessful;

                //ds update average
                dCurrentAverageSquaredError = dErrorSquaredAveragePixels;

                //std::printf( "[%04lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f (inliers: %u/%lu)\n",
                //             uID, uOptimizationsSuccessful, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError, uInliers, m_vecMeasurements.size( ) );
                return vecX.head< 3 >( );
            }
            else
            {
                ++uOptimizationsFailed;
                std::printf( "<CLandmark>(_getOptimizedLandmarkLMA) landmark [%06lu] optimization failed - solution unacceptable (average error: %f, inliers: %u)\n", uID, dErrorSquaredAveragePixels, uInliers );

                //ds if still here the calibration did not converge - keep the initial estimate
                return p_vecInitialGuess;
            }
        }
    }

    ++uOptimizationsFailed;
    std::printf( "<CLandmark>(_getOptimizedLandmarkLMA) landmark [%06lu] optimization failed - system did not converge\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkKRDLMA( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( Eigen::Vector4d::Zero( ) );
    CPoint3DInWorldFrameHomogenized vecX( p_vecInitialGuess.x( ), p_vecInitialGuess.y( ), p_vecInitialGuess.z( ), 1.0 );

    //std::printf( "[%06lu] vecX: %6.2f %6.2f %6.2f\n", uID, vecX.x( ), vecX.y( ), vecX.z( ) );

    const std::vector< CMeasurementLandmark* >::size_type uNumberOfMeasurements( m_vecMeasurements.size( ) );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::m_uCapIterations; ++uIteration )
    {
        double dRSS       = 0.0;
        uint32_t uInliers = 0;
        double dTotalRelativeDepthMeters = 0.0;

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds current element
            const double dUReference( pMeasurement->ptUVLEFT.x );
            const double dVReference( pMeasurement->ptUVLEFT.y );
            const double dRelativeDepthMeters( m_dDepthLastOptimizationMeters/pMeasurement->vecPointXYZLEFT.z( ) );

            //ds compute projection
            const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*vecX );
            const double dA( vecProjectionHomogeneous(0) );
            const double dB( vecProjectionHomogeneous(1) );
            const double dC( vecProjectionHomogeneous(2) );
            const double dCSquared( dC*dC );

            //ds compute current error
            const Eigen::Vector2d vecError( static_cast< double >( dA/dC )-dUReference, static_cast< double >( dB/dC )-dVReference );

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f\n", uID, uIteration, vecError.x( ), vecError.y( ) );

            //ds kernelized LS
            const double dSquaredError( vecError.squaredNorm( ) );
            double dContribution( 1.0 );
            if( CLandmark::m_dKernelMaximumError < dSquaredError )
            {
                //ds safe since if squared error was zero it would not enter the loop (assuming maximum error is bigger than zero)
                dContribution = std::sqrt( CLandmark::m_dKernelMaximumError/dSquaredError );
            }
            else
            {
                ++uInliers;
            }

            //ds kernelized rss
            dRSS += dContribution*dSquaredError;

            //ds compute jacobian
            Eigen::Matrix< double, 2, 4 > matJacobian;
            matJacobian.row(0) = Vector3dT( 1/dC, 0, -dA/dCSquared )*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.row(1) = Vector3dT( 0, 1/dC, -dB/dCSquared )*pMeasurement->matProjectionWORLDtoLEFT;
            const Eigen::Matrix< double, 4, 2 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds compute H and B
            matH += dContribution*dRelativeDepthMeters*matJacobianTransposed*matJacobian + CLandmark::m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
            vecB += dContribution*dRelativeDepthMeters*matJacobianTransposed*vecError;

            //ds update total for next run
            dTotalRelativeDepthMeters += dContribution*dRelativeDepthMeters;
        }

        //ds solve constrained system H*dx=-b (since dx(3) = 0.0)
        const CPoint3DWORLD vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

        //ds update x solution
        vecX.block< 3, 1 >(0,0) += vecDeltaX;

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ) );

        /*std::fprintf( m_pFilePositionOptimization, "  %06lu |      %06lu |       %03u          %03lu     %03u |     %6.2f |  %6.6f | %6.2f %6.2f %6.2f\n",
                        p_uFrame,
                        uID,
                        uIteration,
                        uNumberOfMeasurements,
                        uInliers,
                        dRSS/uNumberOfMeasurements,
                        vecDeltaX.squaredNorm( ),
                        vecX.x( ),
                        vecX.y( ),
                        vecX.z( ) );*/

        //ds check if we have converged
        if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            ++uOptimizationsSuccessful;

            double dSumSquaredErrors = 0.0;

            //ds loop over all previous measurements again
            for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
            {
                //ds get projected point
                const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*vecX );

                //ds compute pixel coordinates TODO remove cast
                const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

                //ds add up the squared error
                dSumSquaredErrors += vecUV.squaredNorm( );
            }

            //ds update last depth
            m_dDepthLastOptimizationMeters = dTotalRelativeDepthMeters/m_vecMeasurements.size( );

            //ds average the measurement
            dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

            //std::printf( "<CLandmark>(_getOptimizedLandmarkKIDLMA) landmark [%06lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f inliers: %u\n", uID, uCalibrations, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError, uInliers );
            return vecX.head< 3 >( );
        }
    }

    //ds failed
    ++uOptimizationsFailed;

    //ds if still here the calibration did not converge - keep the initial estimate, the landmark will get checked out
    return p_vecInitialGuess;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkIDWA( )
{
    //ds return vector
    CPoint3DWORLD vecPointXYZWORLD( Eigen::Vector3d::Zero( ) );

    //ds total accumulated depth
    double dInverseDepthAccumulated = 0.0;

    //ds loop over all measurements
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        //ds current inverse depth
        const double dInverseDepth = 1.0/pMeasurement->vecPointXYZLEFT.z( );

        //ds add current measurement with depth weight
        vecPointXYZWORLD += dInverseDepth*pMeasurement->vecPointXYZWORLD;

        //std::cout << "in camera frame: " << pMeasurement->vecPointXYZ.transpose( ) << std::endl;
        //std::cout << "adding: " << dInverseDepth << " x " << pMeasurement->vecPointXYZWORLD.transpose( ) << std::endl;

        //ds accumulate depth
        dInverseDepthAccumulated += dInverseDepth;
    }

    //ds compute average point
    vecPointXYZWORLD /= dInverseDepthAccumulated;

    //std::cout << "from: " << vecPointXYZCalibrated.transpose( ) << " to: " << vecPointXYZWORLD.transpose( ) << std::endl;

    ++uOptimizationsSuccessful;

    double dSumSquaredErrors = 0.0;

    //ds loop over all previous measurements again
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        //ds get projected point
        const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*CPoint3DHomogenized( vecPointXYZWORLD.x( ), vecPointXYZWORLD.y( ), vecPointXYZWORLD.z( ), 1.0 ) );

        //ds compute pixel coordinates TODO remove cast
        const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

        //ds compute squared error
        const double dSquaredError( vecUV.squaredNorm( ) );

        //ds add up
        dSumSquaredErrors += dSquaredError;
    }

    //ds average the measurement
    dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

    //ds return
    return vecPointXYZWORLD;
}
