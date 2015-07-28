#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptorLEFT,
           const CDescriptor& p_matDescriptorRIGHT,
           const double& p_dKeyPointSize,
           const CPoint3DInWorldFrame& p_vecPointXYZ,
           const CPoint2DHomogenized& p_vecUVLEFTReference,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DInCameraFrame& p_vecPointXYZCAMERA,
           const CPoint3DInWorldFrame& p_vecCameraPosition,
           const Eigen::Vector3d& p_vecCameraOrientation,
           //const Eigen::Matrix3d& p_matKRotation,
           //const Eigen::Vector3d& p_vecKTranslation,
           const MatrixProjection& p_matProjectionWORLDtoLEFT,
           const uint64_t& p_uFrame ): uID( p_uID ),
                                                       matDescriptorReferenceLEFT( p_matDescriptorLEFT ),
                                                       matDescriptorLASTLEFT( p_matDescriptorLEFT ),
                                                       matDescriptorLASTRIGHT( p_matDescriptorRIGHT ),
                                                       dKeyPointSize( p_dKeyPointSize ),
                                                       vecPointXYZInitial( p_vecPointXYZ ),
                                                       vecPointXYZOptimized( vecPointXYZInitial ),
                                                       vecUVLEFTReference( p_vecUVLEFTReference ),
                                                       vecUVReferenceLEFT( p_ptUVLEFT.x, p_ptUVLEFT.y, 1.0 ),
                                                       uFailedSubsequentTrackings( 0 ),
                                                       uOptimizationsSuccessful( 0 ),
                                                       uOptimizationsFailed( 0 ),
                                                       dCurrentAverageSquaredError( 0.0 ),
                                                       vecPointXYZMean( p_vecPointXYZ ),
                                                       bIsCurrentlyVisible( false ),
                                                       m_dDepthLastOptimizationMeters( p_vecPointXYZCAMERA.z( ) ),
                                                       m_vecCameraPositionLAST( p_vecCameraPosition ),
                                                       m_vecCameraOrientationAccumulated( 0.0, 0.0, 0.0 )
{
    //ds construct filestring and open dump file
    //char chBuffer[256];
    //std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    //m_pFilePositionOptimization = std::fopen( chBuffer, "w" );

    //assert( 0 != m_pFilePositionOptimization );

    //ds dump file format
    //std::fprintf( m_pFilePositionOptimization, "ID_FRAME | ID_LANDMARK | ITERATION MEASUREMENTS INLIERS | ERROR_ARSS | DELTA_XYZ |      X      Y      Z\n" );

    //ds add this position
    addMeasurement( p_uFrame, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZCAMERA, vecPointXYZInitial, p_vecCameraPosition, p_vecCameraOrientation, p_matProjectionWORLDtoLEFT );
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

void CLandmark::addMeasurement( const uint64_t& p_uFrame,
                                const cv::Point2d& p_ptUVLEFT,
                                const cv::Point2d& p_ptUVRIGHT,
                                const CPoint3DInCameraFrame& p_vecXYZLEFT,
                                const CPoint3DInWorldFrame& p_vecXYZWORLD,
                                const CPoint3DInWorldFrame& p_vecCameraPosition,
                                const Eigen::Vector3d& p_vecCameraOrientation,
                                //const Eigen::Matrix3d& p_matKRotation,
                                //const Eigen::Vector3d& p_vecKTranslation,
                                const MatrixProjection& p_matProjectionWORLDtoLEFT )
{
    //ds input validation
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );
    assert( 0 < p_vecXYZLEFT.z( ) );

    //ds update mean
    vecPointXYZMean = ( vecPointXYZMean+p_vecXYZWORLD )/2.0;

    //ds accumulate position
    m_vecCameraOrientationAccumulated += p_vecCameraOrientation;

    //ds check if we can recalibrate the 3d position
    if( CLandmark::m_dDistanceDeltaForOptimizationMeters < ( m_vecCameraPositionLAST-p_vecCameraPosition ).squaredNorm( ) ||
        CLandmark::m_dAngleDeltaForOptimizationRadians < m_vecCameraOrientationAccumulated.squaredNorm( ) )
    {
        //ds update last
        m_vecCameraPositionLAST           = p_vecCameraPosition;
        m_vecCameraOrientationAccumulated = Eigen::Vector3d( 0.0, 0.0, 0.0 );

        //ds get calibrated point
        //vecPointXYZCalibrated = _getOptimizedLandmarkKLMA( p_uFrame, vecPointXYZCalibrated );
        //vecPointXYZCalibrated = _getOptimizedLandmarkIDLMA( p_uFrame, vecMeanMeasurement );
        vecPointXYZOptimized = _getOptimizedLandmarkKRDLMA( p_uFrame, vecPointXYZOptimized );
        //vecPointXYZCalibrated = _getOptimizedLandmarkIDWA( );
    }

    //ds add the measurement to structure
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID, p_ptUVLEFT, p_ptUVRIGHT, p_vecXYZLEFT, p_vecXYZWORLD, vecPointXYZOptimized, p_vecCameraPosition, p_matProjectionWORLDtoLEFT ) );
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkKLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( 0.0, 0.0, 0.0, 0.0 );
    CPoint3DInWorldFrameHomogenized vecX( p_vecInitialGuess.x( ), p_vecInitialGuess.y( ), p_vecInitialGuess.z( ), 1.0 );

    //std::printf( "[%06lu] vecX: %6.2f %6.2f %6.2f\n", uID, vecX.x( ), vecX.y( ), vecX.z( ) );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::m_uCapIterations; ++uIteration )
    {
        //ds reset rss
        double dRSSCurrent = 0.0;
        uint32_t uInliers( 0 );

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

            //ds kernelized LS TODO: adjust graveness for distant points
            const double dSquaredError( vecError.squaredNorm( ) );
            double dContribution( 1.0 );
            if( CLandmark::m_dKernelMaximumError < dSquaredError )
            {
                //ds safe since if squared error would was zero it would not enter the loop (assuming maximum error is bigger than zero)
                dContribution = std::sqrt( CLandmark::m_dKernelMaximumError/dSquaredError );
            }
            else
            {
                //ds inlier
                ++uInliers;
            }

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f (squared: %4.2f)\n", uID, uIteration, vecError.x( ), vecError.y( ), dSquaredError );

            //ds accumulate rss
            dRSSCurrent += dContribution*dSquaredError;

            //ds compute jacobian
            Eigen::Matrix< double, 2, 4 > matJacobian;
            matJacobian.row(0) = Eigen::Vector3d( 1/dC, 0, -dA/dCSquared ).transpose( )*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.row(1) = Eigen::Vector3d( 0, 1/dC, -dB/dCSquared ).transpose( )*pMeasurement->matProjectionWORLDtoLEFT;
            const Eigen::Matrix< double, 4, 2 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds compute H and B
            matH += dContribution*matJacobianTransposed*matJacobian + CLandmark::m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
            vecB += dContribution*matJacobianTransposed*vecError;
        }

        //ds solve constrained system (since x(3) = 0.0)
        const CPoint3DInWorldFrame vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

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
            ++uOptimizationsSuccessful;

            double dSumSquaredErrors( 0.0 );

            //ds loop over all previous measurements again
            for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
            {
                //ds get projected point
                const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*vecX );

                //ds compute pixel coordinates TODO remove cast
                const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

                //ds compute squared error
                const double dSquaredError( vecUV.squaredNorm( ) );

                //ds add up
                dSumSquaredErrors += dSquaredError;
            }

            //ds average the measurement
            dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

            //std::printf( "[%04lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f\n", uID, uCalibrations, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError );
            return vecX.head< 3 >( );
        }
    }

    ++uOptimizationsFailed;
    std::printf( "<CLandmark>(_getOptimizedLandmarkLMA) landmark [%06lu] optimization failed\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkIDLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( Eigen::Vector4d::Zero( ) );
    CPoint3DInWorldFrameHomogenized vecX( p_vecInitialGuess.x( ), p_vecInitialGuess.y( ), p_vecInitialGuess.z( ), 1.0 );

    //std::printf( "[%06lu] vecX: %6.2f %6.2f %6.2f\n", uID, vecX.x( ), vecX.y( ), vecX.z( ) );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::m_uCapIterations; ++uIteration )
    {
        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds current element
            const double dUReference( pMeasurement->ptUVLEFT.x );
            const double dVReference( pMeasurement->ptUVLEFT.y );
            const double dInverseDepthMeters( 1.0/pMeasurement->vecPointXYZLEFT.z( ) );

            //ds compute projection
            const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matProjectionWORLDtoLEFT*vecX );
            const double dA( vecProjectionHomogeneous(0) );
            const double dB( vecProjectionHomogeneous(1) );
            const double dC( vecProjectionHomogeneous(2) );
            const double dCSquared( dC*dC );

            //ds compute current error
            const Eigen::Vector2d vecError( static_cast< double >( dA/dC )-dUReference, static_cast< double >( dB/dC )-dVReference );

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f\n", uID, uIteration, vecError.x( ), vecError.y( ) );

            //ds compute jacobian
            Eigen::Matrix< double, 2, 4 > matJacobian;
            matJacobian.row(0) = Vector3dT( 1/dC, 0, -dA/dCSquared )*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.row(1) = Vector3dT( 0, 1/dC, -dB/dCSquared )*pMeasurement->matProjectionWORLDtoLEFT;
            const Eigen::Matrix< double, 4, 2 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds compute H and B
            matH += dInverseDepthMeters*matJacobianTransposed*matJacobian + CLandmark::m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
            vecB += dInverseDepthMeters*matJacobianTransposed*vecError;
        }

        //ds solve constrained system H*dx=-b (since dx(3) = 0.0)
        const CPoint3DInWorldFrame vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

        //ds update x solution
        vecX.block< 3, 1 >(0,0) += vecDeltaX;

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ) );

        //std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( CLandmark::m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
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

            //ds average the measurement
            dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

            //std::printf( "[%04lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f\n", uID, uCalibrations, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError );
            return vecX.head< 3 >( );
        }
    }

    ++uOptimizationsFailed;
    std::printf( "<CLandmark>(_getOptimizedLandmarkIDLMA) landmark [%06lu] optimization failed\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkKRDLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess )
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
            dTotalRelativeDepthMeters += dRelativeDepthMeters;
        }

        //ds solve constrained system H*dx=-b (since dx(3) = 0.0)
        const CPoint3DInWorldFrame vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

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
            m_dDepthLastOptimizationMeters /= dTotalRelativeDepthMeters;

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

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkIDWA( )
{
    //ds return vector
    CPoint3DInWorldFrame vecPointXYZWORLD( Eigen::Vector3d::Zero( ) );

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
