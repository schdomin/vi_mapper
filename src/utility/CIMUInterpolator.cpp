#include "CIMUInterpolator.h"

#include "CLogger.h"

CIMUInterpolator::CIMUInterpolator( ): m_vecOffsetAccelerationLinear( 0.0, 0.0, 0.0 ), m_vecOffsetVelocityAngular( 0.0, 0.0, 0.0 )
{
    m_vecMeasurements.clear( );
    m_vecCalibration.clear( );

    CLogger::openBox( );
    std::printf( "<CIMUInterpolator>(CIMUInterpolator) uMeasurementsForOptimization: %lu\n", uMeasurementsForOptimization );
    std::printf( "<CIMUInterpolator>(CIMUInterpolator) m_dCalibrationConvergenceDeltaRotation: %f\n", m_dCalibrationConvergenceDeltaRotation );
    std::printf( "<CIMUInterpolator>(CIMUInterpolator) m_dCalibrationConvergenceDeltaFinal: %f\n", m_dCalibrationConvergenceDeltaFinal );
    std::printf( "<CIMUInterpolator>(CIMUInterpolator) instance allocated\n" );
    CLogger::closeBox( );
}

CIMUInterpolator::~CIMUInterpolator( )
{
    //ds nothing to do
}

void CIMUInterpolator::addMeasurement( const CLinearAccelerationIMU& p_vecAccelerationLinear, const CAngularVelocityIMU& p_vecVelocityAngular )
{
    m_vecMeasurements.push_back( std::make_pair( p_vecAccelerationLinear, p_vecVelocityAngular ) );
}

void CIMUInterpolator::addMeasurementCalibration( const CLinearAccelerationIMU& p_vecAccelerationLinear, const CAngularVelocityIMU& p_vecVelocityAngular )
{
    m_vecCalibration.push_back( std::make_pair( p_vecAccelerationLinear, p_vecVelocityAngular ) );

    //ds check if we can optimize
    if( 0 == m_vecCalibration.size( )%uMeasurementsForOptimization )
    {
        if( m_bIsRotationComputed )
        {
            calibrateOffsets( );
        }
        else
        {
            calibrateRotation( );
        }
    }
}

void CIMUInterpolator::calibrateOffsets( )
{
    //ds should not be called when calibrated
    assert( !m_bIsCalibrated );

    Eigen::Vector3d vecOffsetAccelerationLinearWORLD( 0.0, 0.0, 0.0 );
    Eigen::Vector3d vecOffsetVelocityAngular( 0.0, 0.0, 0.0 );

    std::for_each( m_vecCalibration.begin( ),m_vecCalibration.end( ),[&]( std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > prMeasurement )
    {
        vecOffsetAccelerationLinearWORLD += m_matRotationIMUtoWORLD*prMeasurement.first;
        vecOffsetVelocityAngular         += prMeasurement.second;
    });

    //ds compute means
    vecOffsetAccelerationLinearWORLD /= m_vecCalibration.size( );
    vecOffsetVelocityAngular         /= m_vecCalibration.size( );

    //ds compute variances
    Eigen::Vector3d vecVarianceAccelerationLinearWORLD( 0.0, 0.0, 0.0 );
    Eigen::Vector3d vecVarianceVelocityAngular( 0.0, 0.0, 0.0 );
    std::for_each( m_vecCalibration.begin( ),m_vecCalibration.end( ),[&]( std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > prMeasurement )
    {
        const Eigen::Vector3d vecLinearAccelerationWORLD( m_matRotationIMUtoWORLD*prMeasurement.first );
        vecVarianceAccelerationLinearWORLD.x( ) += ( vecLinearAccelerationWORLD.x( )-vecOffsetAccelerationLinearWORLD.x( ) )*( vecLinearAccelerationWORLD.x( )-vecOffsetAccelerationLinearWORLD.x( ) );
        vecVarianceAccelerationLinearWORLD.y( ) += ( vecLinearAccelerationWORLD.y( )-vecOffsetAccelerationLinearWORLD.y( ) )*( vecLinearAccelerationWORLD.y( )-vecOffsetAccelerationLinearWORLD.y( ) );
        vecVarianceAccelerationLinearWORLD.z( ) += ( vecLinearAccelerationWORLD.z( )-vecOffsetAccelerationLinearWORLD.z( ) )*( vecLinearAccelerationWORLD.z( )-vecOffsetAccelerationLinearWORLD.z( ) );
        vecVarianceVelocityAngular.x( )    += ( prMeasurement.second.x( )-vecOffsetVelocityAngular.x( ) )*( prMeasurement.second.x( )-vecOffsetVelocityAngular.x( ) );
        vecVarianceVelocityAngular.y( )    += ( prMeasurement.second.y( )-vecOffsetVelocityAngular.y( ) )*( prMeasurement.second.y( )-vecOffsetVelocityAngular.y( ) );
        vecVarianceVelocityAngular.z( )    += ( prMeasurement.second.z( )-vecOffsetVelocityAngular.z( ) )*( prMeasurement.second.z( )-vecOffsetVelocityAngular.z( ) );
    });
    vecVarianceAccelerationLinearWORLD /= m_vecCalibration.size( );
    vecVarianceVelocityAngular         /= m_vecCalibration.size( );

    //ds update measurements if useful
    if( m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetAccelerationLinearWORLD.x( )-m_vecOffsetAccelerationLinear.x( ) ) ||
        m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetAccelerationLinearWORLD.y( )-m_vecOffsetAccelerationLinear.y( ) ) ||
        m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetAccelerationLinearWORLD.z( )-m_vecOffsetAccelerationLinear.z( ) ) ||
        m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetVelocityAngular.x( )-m_vecOffsetVelocityAngular.x( ) ) ||
        m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetVelocityAngular.y( )-m_vecOffsetVelocityAngular.y( ) ) ||
        m_dCalibrationConvergenceDeltaFinal < std::fabs( vecOffsetVelocityAngular.z( )-m_vecOffsetVelocityAngular.z( ) ) )
    {
        m_vecOffsetAccelerationLinear = vecOffsetAccelerationLinearWORLD;
        m_vecOffsetVelocityAngular    = vecOffsetVelocityAngular;

        std::printf( "<CIMUInterpolator>(calibrateOffsets) calibrated linear acceleration | BIAS: %6.3f %6.3f %6.3f | VARIANCE: %7.5f %7.5f %7.5f\n",
                     m_vecOffsetAccelerationLinear.x( ), m_vecOffsetAccelerationLinear.y( ), m_vecOffsetAccelerationLinear.z( ), vecVarianceAccelerationLinearWORLD.x( ), vecVarianceAccelerationLinearWORLD.y( ), vecVarianceAccelerationLinearWORLD.z( ) );
        std::printf( "<CIMUInterpolator>(calibrateOffsets) calibrated angular velocity    | BIAS: %6.3f %6.3f %6.3f | VARIANCE: %7.5f %7.5f %7.5f\n",
                     m_vecOffsetVelocityAngular.x( ), m_vecOffsetVelocityAngular.y( ), m_vecOffsetVelocityAngular.z( ), vecVarianceVelocityAngular.x( ), vecVarianceVelocityAngular.y( ), vecVarianceVelocityAngular.z( ) );
    }
    else
    {
        std::printf( "<CIMUInterpolator>(calibrateOffsets) converged - IMU calibration complete (measurements: %lu)\n", m_vecCalibration.size( ) );
        m_bIsCalibrated = true;
    }
}

void CIMUInterpolator::calibrateRotation( )
{
    //ds should not be called when calibrated
    assert( !m_bIsRotationComputed );

    Eigen::Vector3d vecOffsetAccelerationLinear( 0.0, 0.0, 0.0 );

    std::for_each( m_vecCalibration.begin( ),m_vecCalibration.end( ),[&]( std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > prMeasurement )
    {
        vecOffsetAccelerationLinear += prMeasurement.first;
    });

    //ds compute means
    vecOffsetAccelerationLinear /= m_vecCalibration.size( );

    //ds compute variances
    Eigen::Vector3d vecVarianceAccelerationLinear( 0.0, 0.0, 0.0 );
    std::for_each( m_vecCalibration.begin( ),m_vecCalibration.end( ),[&]( std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > prMeasurement )
    {
        vecVarianceAccelerationLinear.x( ) += ( prMeasurement.first.x( )-vecOffsetAccelerationLinear.x( ) )*( prMeasurement.first.x( )-vecOffsetAccelerationLinear.x( ) );
        vecVarianceAccelerationLinear.y( ) += ( prMeasurement.first.y( )-vecOffsetAccelerationLinear.y( ) )*( prMeasurement.first.y( )-vecOffsetAccelerationLinear.y( ) );
        vecVarianceAccelerationLinear.z( ) += ( prMeasurement.first.z( )-vecOffsetAccelerationLinear.z( ) )*( prMeasurement.first.z( )-vecOffsetAccelerationLinear.z( ) );
    });
    vecVarianceAccelerationLinear /= m_vecCalibration.size( );

    //ds update measurements if useful
    if( m_dCalibrationConvergenceDeltaRotation < std::fabs( vecOffsetAccelerationLinear.x( )-m_vecOffsetAccelerationLinear.x( ) ) ||
        m_dCalibrationConvergenceDeltaRotation < std::fabs( vecOffsetAccelerationLinear.y( )-m_vecOffsetAccelerationLinear.y( ) ) ||
        m_dCalibrationConvergenceDeltaRotation < std::fabs( vecOffsetAccelerationLinear.z( )-m_vecOffsetAccelerationLinear.z( ) ) )
    {
        m_vecOffsetAccelerationLinear = vecOffsetAccelerationLinear;

        std::printf( "<CIMUInterpolator>(calibrateRotation) calibrated linear acceleration | BIAS: %6.3f %6.3f %6.3f | VARIANCE: %7.5f %7.5f %7.5f\n",
                     m_vecOffsetAccelerationLinear.x( ), m_vecOffsetAccelerationLinear.y( ), m_vecOffsetAccelerationLinear.z( ), vecVarianceAccelerationLinear.x( ), vecVarianceAccelerationLinear.y( ), vecVarianceAccelerationLinear.z( ) );
    }
    else
    {
        //ds set member
        m_matRotationIMUtoWORLD = _getRotationIMUtoWORLD( m_vecOffsetAccelerationLinear );
        m_matRotationWORLDtoIMU = m_matRotationIMUtoWORLD.transpose( );
        m_bIsRotationComputed   = true;

        //ds reset offsets for linear acceleration
        m_vecOffsetAccelerationLinear = Eigen::Vector3d::Zero( );

        std::cout << "<CIMUInterpolator>(calibrateRotation) converged - found rotation matrix (IMU to WORLD): \n" << std::endl;
        std::cout << m_matRotationIMUtoWORLD.matrix( ) << std::endl;
        std::printf( "\n<CIMUInterpolator>(calibrateRotation) starting BIAS and VARIANCE calibration\n" );
    }
}

const bool CIMUInterpolator::isCalibrated( ) const
{
    return m_bIsCalibrated;
}

const Eigen::Isometry3d CIMUInterpolator::getTransformationWORLDtoCAMERA( const Eigen::Matrix3d& p_matRotationIMUtoCAMERA ) const
{
    //ds allocate isometry
    Eigen::Isometry3d matTransformationWORLDtoCAMERA( Eigen::Matrix4d::Identity( ) );

    //ds set rotational part
    matTransformationWORLDtoCAMERA.linear( ) = p_matRotationIMUtoCAMERA*m_matRotationWORLDtoIMU;

    return matTransformationWORLDtoCAMERA;
}

const Eigen::Isometry3d CIMUInterpolator::getTransformationWORLDtoCAMERA( const Eigen::Matrix3d& p_matRotationIMUtoCAMERA,
                                                                          const std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type& p_uNumberOfMeasurements ) const
{
    //ds allocate isometry
    Eigen::Isometry3d matTransformationWORLDtoCAMERA( Eigen::Matrix4d::Identity( ) );

    //ds compute IMU to WORLD
    const Eigen::Matrix3d matRotationWORLDtoIMU( ( _getRotationIMUtoWORLD( getLinearAccelerationAveraged( p_uNumberOfMeasurements ) ) ).transpose( ) );

    //ds set rotational part
    matTransformationWORLDtoCAMERA.linear( ) = p_matRotationIMUtoCAMERA*matRotationWORLDtoIMU;

    return matTransformationWORLDtoCAMERA;
}

const CLinearAccelerationIMU CIMUInterpolator::getLinearAccelerationAveraged( const std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type& p_uNumberOfMeasurements ) const
{
    assert( p_uNumberOfMeasurements <= m_vecMeasurements.size( ) );

    //ds set start index
    const std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type uIndexStart = m_vecMeasurements.size( )-p_uNumberOfMeasurements;

    assert( 0 <= uIndexStart );
    assert( uIndexStart < m_vecMeasurements.size( ) );

    //ds acceleration sum
    CLinearAccelerationIMU vecLinearAcceleration( 0.0, 0.0, 0.0 );

    //ds loop over the elements
    for( std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type u = uIndexStart; u < m_vecMeasurements.size( ); ++u )
    {
        vecLinearAcceleration += m_vecMeasurements[u].first;
    }

    //ds return average
    return vecLinearAcceleration/p_uNumberOfMeasurements;
}

const Eigen::Matrix3d CIMUInterpolator::getRotationWORLDtoCAMERA( const Eigen::Matrix3d& p_matRotationIMUtoCAMERA,
                                                                  const std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type& p_uNumberOfMeasurements ) const
{
    //ds compute IMU to WORLD
    const Eigen::Matrix3d matRotationWORLDtoIMU( ( _getRotationIMUtoWORLD( getLinearAccelerationAveraged( p_uNumberOfMeasurements ) ) ).transpose( ) );

    //ds return the rotation matrix
    return p_matRotationIMUtoCAMERA*matRotationWORLDtoIMU;
}

const Eigen::Matrix3d CIMUInterpolator::getRotationCAMERAtoWORLD( const Eigen::Matrix3d& p_matRotationIMUtoCAMERA,
                                                                  const std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type& p_uNumberOfMeasurements ) const
{
    //ds return the rotation matrix
    return ( getRotationWORLDtoCAMERA( p_matRotationIMUtoCAMERA, p_uNumberOfMeasurements ) ).transpose( );
}

/*void addMeasurement( Eigen::Vector3d& p_vecAccelerationLinear, const Eigen::Vector3d& p_vecVelocityAngular, const double& p_dTimestampSeconds )
{
    //std::cout << p_vecAccelerationLinear.transpose( ) << std::endl;

    //ds calibrated measurements
    const Eigen::Vector3d vecAccelerationLinearClean( p_vecAccelerationLinear-m_vecOffsetAccelerationLinear );
    const Eigen::Vector3d vecVelocityAngularClean( p_vecVelocityAngular-m_vecOffsetVelocityAngular );

    //ds compute time delta
    const double dTimespanSeconds = p_dTimestampSeconds-m_dLastTimestamp;

    if( m_dMaximumDeltaTimeSeconds > dTimespanSeconds )
    {
        //std::cout << vecAccelerationLinearClean.transpose( ) << std::endl;
        //std::cout << vecVelocityAngularClean.transpose( ) << std::endl;

        //ds set current transformation
        Eigen::Isometry3d matTransformationIMU( Eigen::Matrix4d::Identity( ) );
        matTransformationIMU.translation( ) = m_vecLastVelocityLinear*dTimespanSeconds + 1/2.0*vecAccelerationLinearClean*dTimespanSeconds*dTimespanSeconds;
        matTransformationIMU.linear( )      = CMiniVisionToolbox::fromOrientationRodrigues( static_cast< Eigen::Vector3d >( dTimespanSeconds*m_vecLastVelocityAngular ) );

        //std::cout << matTransformationIMU.matrix( ) << std::endl;

        //ds compute next transformation and add it to the map
        m_mapTransformationsIMU.insert( std::pair< double, Eigen::Isometry3d >( p_dTimestampSeconds, matTransformationIMU*m_mapTransformationsIMU.at( m_dLastTimestamp ) ) );

        //std::cout << m_mapTransformationsIMU.at( p_dTimestampSeconds ).matrix( ) << std::endl;

        //ds update references
        m_vecLastVelocityLinear += vecAccelerationLinearClean*dTimespanSeconds;
        m_vecLastVelocityAngular = vecVelocityAngularClean;
    }
    else
    {
        //ds add identity transform
        m_mapTransformationsIMU.insert( std::pair< double, Eigen::Isometry3d >( p_dTimestampSeconds, Eigen::Isometry3d( Eigen::Matrix4d::Identity( ) ) ) );
    }

    //ds update timestamp
    m_dLastTimestamp = p_dTimestampSeconds;
}

const Eigen::Isometry3d getTransformation( const double& p_dTimestampSeconds ) const
{
    return m_mapTransformationsIMU.at( p_dTimestampSeconds );
}*/

const CLinearAccelerationWORLD CIMUInterpolator::getLinearAccelerationFiltered( const CLinearAccelerationWORLD& p_vecLinearAcceleration )
{
    const double dAccelerationX = p_vecLinearAcceleration.x( )-CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[0];
    const double dAccelerationY = p_vecLinearAcceleration.y( )-CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[1];
    const double dAccelerationZ = p_vecLinearAcceleration.z( )-CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[2];

    const double dAbsoluteAccelerationX = std::fabs( dAccelerationX )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
    const double dAbsoluteAccelerationY = std::fabs( dAccelerationY )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
    const double dAbsoluteAccelerationZ = std::fabs( dAccelerationZ )-CIMUInterpolator::m_dImprecisionLinearAcceleration;

    CLinearAccelerationWORLD vecLinearAccelerationFiltered( 0.0, 0.0, 0.0 );

    if( 0.0 < dAbsoluteAccelerationX ){ vecLinearAccelerationFiltered.x( ) = CIMUInterpolator::sign( dAccelerationX )*std::sqrt( dAbsoluteAccelerationX ); }
    if( 0.0 < dAbsoluteAccelerationY ){ vecLinearAccelerationFiltered.y( ) = CIMUInterpolator::sign( dAccelerationY )*std::sqrt( dAbsoluteAccelerationY ); }
    if( 0.0 < dAbsoluteAccelerationZ ){ vecLinearAccelerationFiltered.z( ) = CIMUInterpolator::sign( dAccelerationZ )*std::sqrt( dAbsoluteAccelerationZ ); }

    return vecLinearAccelerationFiltered;
}

const CAngularVelocityIMU CIMUInterpolator::getAngularVelocityFiltered( const CAngularVelocityIMU& p_vecAngularVelocity )
{
    //ds filter imprecision and bias
    const double dRotationX = p_vecAngularVelocity.x( );
    const double dRotationY = p_vecAngularVelocity.y( );
    const double dRotationZ = p_vecAngularVelocity.z( );
    const double dAbsoluteRotationX = std::fabs( dRotationX )-CIMUInterpolator::m_dImprecisionAngularVelocity;
    const double dAbsoluteRotationY = std::fabs( dRotationY )-CIMUInterpolator::m_dImprecisionAngularVelocity;
    const double dAbsoluteRotationZ = std::fabs( dRotationZ )-CIMUInterpolator::m_dImprecisionAngularVelocity;

    //ds noise free vectors
    CAngularVelocityLEFT vecAngularVelocityFiltered( 0.0, 0.0, 0.0 );

    //ds update only if meaningful
    if( 0 < dAbsoluteRotationX ){ vecAngularVelocityFiltered.x( ) = CIMUInterpolator::sign( dRotationX )*dAbsoluteRotationX; }
    if( 0 < dAbsoluteRotationY ){ vecAngularVelocityFiltered.y( ) = CIMUInterpolator::sign( dRotationY )*dAbsoluteRotationY; }
    if( 0 < dAbsoluteRotationZ ){ vecAngularVelocityFiltered.z( ) = CIMUInterpolator::sign( dRotationZ )*dAbsoluteRotationZ; }

    return vecAngularVelocityFiltered;
}

//ds nasty ghastly hacky function
const int8_t CIMUInterpolator::sign( const double& p_fNumber )
{
    assert( 0.0 != p_fNumber );

    if( 0.0 < p_fNumber ){ return 1; }
    if( 0.0 > p_fNumber ){ return -1; }

    //ds never gets called, just pleasing the compiler
    assert( false );
    return 0;
}

const Eigen::Matrix3d CIMUInterpolator::_getRotationIMUtoWORLD( const Eigen::Vector3d& p_LinearAcceleration ) const
{
    //ds converged - we can compute the rotation
    const Eigen::Vector3d vecIMU( p_LinearAcceleration.normalized( ) );
    const Eigen::Vector3d vecGraviation( 0.0, 0.0, -1.0 );

    //ds compute rotation matrix
    const Eigen::Vector3d vecCross( vecIMU.cross( vecGraviation ) );
    const double dSine   = vecCross.norm( );
    const double dCosine = vecIMU.transpose( )*vecGraviation;
    const Eigen::Matrix3d matSkew( CMiniVisionToolbox::getSkew( vecCross ) );

    //ds return the rotation matrix
    return Eigen::Matrix3d::Identity( ) + matSkew + ( 1-dCosine )/( dSine*dSine )*matSkew*matSkew;
}
