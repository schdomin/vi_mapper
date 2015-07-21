#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptorLEFT,
           const CDescriptor& p_matDescriptorRIGHT,
           const double& p_dKeyPointSize,
           const CPoint3DInWorldFrame& p_vecPointXYZ,
           const CPoint2DInCameraFrameHomogenized& p_vecUVLEFTReference,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DInCameraFrame& p_vecPointXYZCamera,
           const Eigen::Vector3d& p_vecCameraPosition,
           //const Eigen::Matrix3d& p_matKRotation,
           //const Eigen::Vector3d& p_vecKTranslation,
           const MatrixProjection& p_matProjectionWORLDtoLEFT,
           const uint64_t& p_uFrame ): uID( p_uID ),
                                                       matDescriptorReference( p_matDescriptorLEFT ),
                                                       matDescriptorLastLEFT( p_matDescriptorLEFT ),
                                                       matDescriptorLastRIGHT( p_matDescriptorRIGHT ),
                                                       dKeyPointSize( p_dKeyPointSize ),
                                                       vecPointXYZInitial( p_vecPointXYZ ),
                                                       vecPointXYZCalibrated( vecPointXYZInitial ),
                                                       vecUVLEFTReference( p_vecUVLEFTReference ),
                                                       uFailedSubsequentTrackings( 0 ),
                                                       uCalibrations( 0 ),
                                                       dCurrentAverageSquaredError( 0.0 ),
                                                       vecMeanMeasurement( p_vecPointXYZ ),
                                                       bIsCurrentlyVisible( false ),
                                                       m_vecLastCameraPosition( p_vecCameraPosition )
{
    //ds construct filestring and open dump file
    //char chBuffer[256];
    //std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    //m_pFilePosition = std::fopen( chBuffer, "w" );

    //ds dump file format
    //std::fprintf( m_pFilePosition, "FRAME LANDMARK ITERATION MEASUREMENTS INLIERS ERROR\n" );

    //ds add this position
    addPosition( p_uFrame, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZCamera, vecPointXYZInitial, p_vecCameraPosition, p_matProjectionWORLDtoLEFT );
}

CLandmark::~CLandmark( )
{
    //ds close file
    //std::fclose( m_pFilePosition );

    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        delete pMeasurement;
    }
}

void CLandmark::addPosition( const uint64_t& p_uFrame,
                             const cv::Point2d& p_ptUVLEFT,
                             const cv::Point2d& p_ptUVRIGHT,
                             const CPoint3DInCameraFrame& p_vecPointXYZLEFT,
                             const CPoint3DInWorldFrame& p_vecPointXYZ,
                             const Eigen::Vector3d& p_vecCameraPosition,
                             //const Eigen::Matrix3d& p_matKRotation,
                             //const Eigen::Vector3d& p_vecKTranslation,
                             const MatrixProjection& p_matProjectionWORLDtoLEFT )
{
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZLEFT, p_vecPointXYZ, p_vecCameraPosition, p_matProjectionWORLDtoLEFT ) );

    //ds epipolar constraint
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );

    //ds update mean
    vecMeanMeasurement = ( vecMeanMeasurement + p_vecPointXYZ )/2.0;

    //ds check if we can recalibrate the 3d position
    if( m_dDistanceDeltaForCalibration < ( m_vecLastCameraPosition-p_vecCameraPosition ).squaredNorm( ) )
    {
        //ds update last
        m_vecLastCameraPosition = p_vecCameraPosition;

        //ds get calibrated point
        //vecPointXYZCalibrated = _getOptimizedLandmarkKLMA( p_uFrame, vecPointXYZCalibrated );
        vecPointXYZCalibrated = _getOptimizedLandmarkIDLMA( p_uFrame, vecMeanMeasurement );
        //vecPointXYZCalibrated = _getOptimizedLandmarkIDWA( );
    }
}

const cv::Point2d CLandmark::getLastDetectionLEFT( ) const
{
    return m_vecMeasurements.back( )->ptUVLEFT;
}
const cv::Point2d CLandmark::getLastDetectionRIGHT( ) const
{
    return m_vecMeasurements.back( )->ptUVRIGHT;
}

const CMeasurementLandmark* CLandmark::getLastMeasurement( ) const
{
    return m_vecMeasurements.back( );
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkKLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( 0.0, 0.0, 0.0, 0.0 );
    CPoint3DInWorldFrameHomogenized vecX( p_vecInitialGuess.x( ), p_vecInitialGuess.y( ), p_vecInitialGuess.z( ), 1.0 );

    //std::printf( "[%06lu] vecX: %6.2f %6.2f %6.2f\n", uID, vecX.x( ), vecX.y( ), vecX.z( ) );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < m_uIterations; ++uIteration )
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
            if( m_dMaximumError < dSquaredError )
            {
                //ds safe since if squared error would was zero it would not enter the loop (assuming maximum error is bigger than zero)
                dContribution = std::sqrt( m_dMaximumError/dSquaredError );
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
            matH += dContribution*matJacobianTransposed*matJacobian + m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
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
        if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            ++uCalibrations;

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

    //std::cout << "[" << uID << "] FAILED" << std::endl;
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
    for( uint32_t uIteration = 0; uIteration < m_uIterations; ++uIteration )
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
            matH += dInverseDepthMeters*matJacobianTransposed*matJacobian + m_dLevenbergDamping*Eigen::Matrix4d::Identity( );
            vecB += dInverseDepthMeters*matJacobianTransposed*vecError;
        }

        //ds solve constrained system H*dx=-b (since dx(3) = 0.0)
        const CPoint3DInWorldFrame vecDeltaX( matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB ) );

        //ds update x solution
        vecX.block< 3, 1 >(0,0) += vecDeltaX;

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ) );

        //std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            ++uCalibrations;

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

    std::printf( "<CLandmark>(_getOptimizedLandmarkIDLMA) landmark [%06lu] optimization failed\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
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

    ++uCalibrations;

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
