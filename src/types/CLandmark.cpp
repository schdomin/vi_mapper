#include "CLandmark.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptor,
           const CPoint3DInWorldFrame& p_vecPositionXYZ,
           const CPoint2DInCameraFrameHomogenized& p_cPositionUVReference ): uID( p_uID ),
                                                                             matDescriptorReference( p_matDescriptor ),
                                                                             matDescriptorLast( p_matDescriptor ),
                                                                             vecPositionXYZ( p_vecPositionXYZ ),
                                                                             vecPositionUVReference( p_cPositionUVReference ),
                                                                             uFailedSubsequentTrackings( 0 ),
                                                                             m_vecLastCameraPosition( 0.0, 0.0, 0.0 )
{
    //ds nothing to do
}

CLandmark::~CLandmark( )
{
    //ds free positions
    for( const CPositionRaw* pPosition: m_vecPosition )
    {
        delete pPosition;
    }
}

void CLandmark::addPosition( const CPoint3DInWorldFrame& p_vecPointTriangulated,
                             const cv::Point2d& p_ptPointDetected,
                             const Eigen::Vector3d& p_vecCameraPosition,
                             const Eigen::Matrix3d& p_matKRotation,
                             const Eigen::Vector3d& p_vecKTranslation )
{
    m_vecPosition.push_back( new CPositionRaw( p_vecPointTriangulated, p_ptPointDetected, p_vecCameraPosition, p_matKRotation, p_vecKTranslation ) );

    //ds current setup
    const uint64_t uSize( m_vecPosition.size( ) );

    //ds check if we can calibrate the 3d position
    if( 0 == uSize%m_uCalibrationPoints )
    {
        //ds initial values
        Eigen::Matrix3d matH( Eigen::Matrix3d::Zero( ) );
        Eigen::Vector3d vecB( 0.0, 0.0, 0.0 );
        Eigen::Vector3d vecX( vecPositionXYZ );

        std::cout << "initial: (" << vecPositionXYZ.transpose( );

        //ds iterations (break-out if convergence reached early)
        for( uint32_t u = 0; u < m_uIterations; ++u )
        {
            //ds do calibration for the last n elements
            for( uint32_t v = 1; v < m_uCalibrationPoints+1; ++v )
            {
                assert( 0 <= uSize-v );

                //ds current element
                const CPositionRaw* cPosition( m_vecPosition[uSize-v] );
                const int32_t iUReference( cPosition->ptPosition.x );
                const int32_t iVReference( cPosition->ptPosition.y );

                //ds compute projection
                const Eigen::Vector3d vecProjection( cPosition->matKRotation*vecX + cPosition->vecKTranslation );
                const double dX( vecProjection(0) );
                const double dY( vecProjection(1) );
                const double dZ( vecProjection(2) );

                //ds compute current error
                const Eigen::Vector2d vecError( static_cast< double >( dX/dZ )-iUReference, static_cast< double >( dY/dZ )-iVReference );

                //ds compute jacobian
                Eigen::Matrix< double, 2, 3 > matJacobian;
                matJacobian.row(0) = Eigen::Vector3d( 1/dZ, 0, -dX/(dZ*dZ) ).transpose( )*cPosition->matKRotation;
                matJacobian.row(1) = Eigen::Vector3d( 0, 1/dZ, -dY/(dZ*dZ) ).transpose( )*cPosition->matKRotation;

                //ds compute H and B
                matH += matJacobian.transpose( )*matJacobian + m_dLevenbergDamping*Eigen::Matrix3d::Identity( );
                vecB += matJacobian.transpose( )*vecError;
            }

            //ds solve system H*x=-b -> shift the minus to delta addition
            const Eigen::Vector3d vecDeltaX( matH.householderQr( ).solve( vecB ) );

            //ds update x solution
            vecX -= vecDeltaX;

            //std::cout << "iteration[" << u << "]: " << vecX.transpose( ) << std::endl;

            //ds check if we have converged
            if( m_dConvergenceDelta > vecDeltaX.squaredNorm( ) )
            {
                std::cout << ") converged in [" << u << "] iterations to (" << vecX.transpose( );
                break;
            }
        }

        std::cout << ")" << std::endl;
    }
}

const cv::Point2d CLandmark::getLastPosition( ) const
{
    return m_vecPosition.back( )->ptPosition;
}
