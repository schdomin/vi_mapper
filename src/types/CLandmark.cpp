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
           const Eigen::Vector3d& p_vecKTranslation ): uID( p_uID ),
                                                       matDescriptorReference( p_matDescriptor ),
                                                       matDescriptorLast( p_matDescriptor ),
                                                       dKeyPointSize( p_dKeyPointSize ),
                                                       vecPointXYZCalibrated( p_vecPointXYZ ),
                                                       vecUVLEFTReference( p_vecUVLEFTReference ),
                                                       uFailedSubsequentTrackings( 0 ),
                                                       uCalibrations( 0 ),
                                                       dCurrentAverageSquaredError( 0.0 ),
                                                       m_vecLastCameraPosition( p_vecCameraPosition )
{
    //ds add this position
    addPosition( p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZCamera, p_vecCameraPosition, p_matKRotation, p_vecKTranslation );

    //ds check initialization
    assert( 0.0 < m_dMaximumError );
}

CLandmark::~CLandmark( )
{
    //ds free positions
    for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
    {
        delete pMeasurement;
    }
}

void CLandmark::addPosition( const cv::Point2d& p_ptUVLEFT,
                             const cv::Point2d& p_ptUVRIGHT,
                             const CPoint3DInCameraFrame& p_vecPointXYZ,
                             const Eigen::Vector3d& p_vecCameraPosition,
                             const Eigen::Matrix3d& p_matKRotation,
                             const Eigen::Vector3d& p_vecKTranslation )
{
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID, p_ptUVLEFT, p_ptUVRIGHT, p_vecPointXYZ, p_vecCameraPosition, p_matKRotation, p_vecKTranslation ) );

    //ds check if we can recalibrate the 3d position
    if( m_dDistanceDeltaForCalibration < ( m_vecLastCameraPosition-p_vecCameraPosition ).squaredNorm( ) )
    {
        //ds update last
        m_vecLastCameraPosition = p_vecCameraPosition;

        //ds get calibrated point
        vecPointXYZCalibrated = _getCalibratedLSHH( vecPointXYZCalibrated );
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

const CPoint3DInWorldFrame CLandmark::_getCalibratedLSHH( const CPoint3DInWorldFrame& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix3d matH( Eigen::Matrix3d::Zero( ) );
    Eigen::Vector3d vecB( 0.0, 0.0, 0.0 );
    CPoint3DInWorldFrame vecX( p_vecInitialGuess );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t u = 0; u < m_uIterations; ++u )
    {
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

            const double dSquaredError( vecError.squaredNorm( ) );
            float dContribution( 1.0 );
            if( m_dMaximumError < dSquaredError )
            {
                //ds safe since if squared error would was zero it would not enter the loop (assuming maximum error is bigger than zero)
                dContribution = std::sqrt( m_dMaximumError/dSquaredError );
            }


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

        //std::printf( "iteration[%04lu][%04u]: %6.2f %6.2f %6.2f\n", uID, u, vecX(0), vecX(1), vecX(2) );

        //ds check if we have converged
        if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
        {
            ++uCalibrations;

            double dSumSquaredErrors( 0.0 );

            std::cout << std::endl;

            //ds loop over all previous measurements again
            for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
            {
                //ds get projected point
                const CPoint2DHomogenized vecProjectionHomogeneous( pMeasurement->matKRotationLEFT*vecX + pMeasurement->vecKTranslationLEFT );

                //ds compute pixel coordinates
                const Eigen::Vector2d vecUV = vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) - CWrapperOpenCV::fromCVVector( pMeasurement->ptUVLEFT ) ;

                //ds compute squared error
                const double dSquaredError( vecUV.squaredNorm( ) );

                std::cout << "current error: " << dSquaredError << std::endl;

                //ds add up
                dSumSquaredErrors += dSquaredError;
            }

            //ds average the measurement
            dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

            std::cout << "converged in [" << u << "] iterations to (" << vecX.transpose( ) << ") initial: (" << p_vecInitialGuess.transpose( ) << ")" << std::endl;
            std::cout << "sum of squared errors: " << dCurrentAverageSquaredError << std::endl;
            return vecX;
        }
    }

    //std::cout << " FAILED" << std::endl;

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}
