#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptor,
           const double& p_dKeyPointSize,
           const CPoint3DInWorldFrame& p_vecPointXYZ,
           const CPoint2DInCameraFrameHomogenized& p_vecUVLEFTReference,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DInCameraFrame& p_vecPointXYZCamera,
           const Eigen::Vector3d& p_vecCameraPosition,
           const Eigen::Matrix3d& p_matKRotation,
           const Eigen::Vector3d& p_vecKTranslation,
           const uint64_t& p_uFrame ): uID( p_uID ),
                                                       matDescriptorReference( p_matDescriptor ),
                                                       matDescriptorLast( p_matDescriptor ),
                                                       dKeyPointSize( p_dKeyPointSize ),
                                                       vecPointXYZInitial( p_vecPointXYZ ),
                                                       vecPointXYZCalibrated( vecPointXYZInitial ),
                                                       vecUVLEFTReference( p_vecUVLEFTReference ),
                                                       uFailedSubsequentTrackings( 0 ),
                                                       uCalibrations( 0 ),
                                                       dCurrentAverageSquaredError( 0.0 ),
                                                       m_vecLastCameraPosition( p_vecCameraPosition )
{
    //ds construct filestring and open dump file
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    m_pFilePosition = std::fopen( chBuffer, "w" );

    //ds dump file format
    std::fprintf( m_pFilePosition, "FRAME LANDMARK ITERATION MEASUREMENTS INLIERS ERROR\n" );

    //ds add this position
    addPosition( p_uFrame, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZCamera, vecPointXYZInitial, p_vecCameraPosition, p_matKRotation, p_vecKTranslation );

    //ds check initialization
    assert( 0.0 < m_dMaximumError );
}

CLandmark::~CLandmark( )
{
    //ds close file
    std::fclose( m_pFilePosition );

    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        delete pMeasurement;
    }
}

void CLandmark::addPosition( const uint64_t& p_uFrame,
                             const cv::Point2d& p_ptUVLEFT,
                             const cv::Point2d& p_ptUVRIGHT,
                             const CPoint3DInCameraFrame& p_vecPointXYZ,
                             const CPoint3DInWorldFrame& p_vecPointXYZWORLD,
                             const Eigen::Vector3d& p_vecCameraPosition,
                             const Eigen::Matrix3d& p_matKRotation,
                             const Eigen::Vector3d& p_vecKTranslation )
{
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZ, p_vecPointXYZWORLD, p_vecCameraPosition, p_matKRotation, p_vecKTranslation ) );

    //ds check if we can recalibrate the 3d position
    if( m_dDistanceDeltaForCalibration < ( m_vecLastCameraPosition-p_vecCameraPosition ).squaredNorm( ) )
    {
        //ds update last
        m_vecLastCameraPosition = p_vecCameraPosition;

        //ds get calibrated point
        //vecPointXYZCalibrated = _getOptimizedLandmarkLMA( p_uFrame, vecPointXYZCalibrated );
        vecPointXYZCalibrated = _getOptimizedLandmarkIDWA( );
    }
}

const cv::Point2d CLandmark::getLastDetectionLEFT( ) const
{
    return m_vecMeasurements.back( )->ptUVLEFT;
}

const CMeasurementLandmark* CLandmark::getLastMeasurement( ) const
{
    //ds current last element
    return m_vecMeasurements.back( );
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkLMA( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix3d matH( Eigen::Matrix3d::Zero( ) );
    Eigen::Vector3d vecB( 0.0, 0.0, 0.0 );
    CPoint3DInWorldFrame vecX( p_vecInitialGuess );

    //ds previous iterations RSS
    //double dRSSPrevious             = 0.0;
    //double dLevenbergDampingCurrent = m_dLevenbergDamping;

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
            const int32_t iUReference( pMeasurement->ptUVLEFT.x );
            const int32_t iVReference( pMeasurement->ptUVLEFT.y );

            //ds compute projection
            const Eigen::Vector3d vecProjectionHomogeneous( pMeasurement->matKRotationLEFT*vecX + pMeasurement->vecKTranslationLEFT );
            const double dX( vecProjectionHomogeneous(0) );
            const double dY( vecProjectionHomogeneous(1) );
            const double dZ( vecProjectionHomogeneous(2) );

            //ds compute current error
            const Eigen::Vector2d vecError( static_cast< double >( dX/dZ )-iUReference, static_cast< double >( dY/dZ )-iVReference );

            //ds kernelized LS
            const double dSquaredError( vecError.squaredNorm( ) );
            float dContribution( 1.0 );
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

            //ds accumulate rss
            dRSSCurrent += dContribution*dSquaredError;

            //ds compute jacobian
            Eigen::Matrix< double, 2, 3 > matJacobian;
            matJacobian.row(0) = Eigen::Vector3d( 1/dZ, 0, -dX/(dZ*dZ) ).transpose( )*pMeasurement->matKRotationLEFT;
            matJacobian.row(1) = Eigen::Vector3d( 0, 1/dZ, -dY/(dZ*dZ) ).transpose( )*pMeasurement->matKRotationLEFT;
            const Eigen::Matrix< double, 3, 2 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds compute H and B
            matH += dContribution*matJacobianTransposed*matJacobian + m_dLevenbergDamping*Eigen::Matrix3d::Identity( );
            vecB += dContribution*matJacobianTransposed*vecError;
        }

        //ds solve system H*x=-b -> shift the minus to delta addition
        const CPoint3DInWorldFrame vecDeltaX( matH.householderQr( ).solve( vecB ) );

        //ds update x solution (take the minus here)
        vecX -= vecDeltaX;

        /*ds check if we have to adjust the damping
        if( dRSSCurrent > dRSSPrevious )
        {
            dLevenbergDampingCurrent /= m_dFactorDamping;
            dRSSPrevious = dRSSCurrent;
            //std::printf( "levenberg damping: %f\n", dLevenbergDampingCurrent );
        }*/

        //std::printf( "iteration[%04lu][%04u]: %6.2f %6.2f %6.2f\n", uID, u, vecX(0), vecX(1), vecX(2) );

        std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            ++uCalibrations;

            double dSumSquaredErrors( 0.0 );

            //ds loop over all previous measurements again
            for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
            {
                //ds get projected point
                const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matKRotationLEFT*vecX + pMeasurement->vecKTranslationLEFT );

                //ds compute pixel coordinates TODO remove cast
                const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

                //ds compute squared error
                const double dSquaredError( vecUV.squaredNorm( ) );

                //std::cout << "current error: " << dSquaredError << std::endl;

                //ds add up
                dSumSquaredErrors += dSquaredError;
            }

            //ds average the measurement
            dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

            //std::printf( "[%04lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f\n", uID, uCalibrations, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError );
            return vecX;
        }
    }

    //std::cout << "[" << uID << "] FAILED" << std::endl;
    std::printf( "LANDMARK OPTIMIZATION FAILED\n" );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DInWorldFrame CLandmark::_getOptimizedLandmarkIDWA( )
{
    //ds return vector
    CPoint3DInWorldFrame vecPointXYZWORLD( 0.0, 0.0, 0.0 );

    //ds total accumulated depth
    double dInverseDepthAccumulated = 0.0;

    //ds loop over all measurements
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        //ds current inverse depth
        const double dInverseDepth = 1.0/pMeasurement->vecPointXYZ.z( );

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

    //ds return
    return vecPointXYZWORLD;
}
