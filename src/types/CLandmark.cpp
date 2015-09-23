#include "CLandmark.h"
#include "utility/CWrapperOpenCV.h"
#include "vision/CMiniVisionToolbox.h"

CLandmark::CLandmark( const UIDLandmark& p_uID,
           const CDescriptor& p_matDescriptorLEFT,
           const CDescriptor& p_matDescriptorRIGHT,
           const double& p_dKeyPointSize,
           const cv::Point2d& p_ptUVLEFT,
           const cv::Point2d& p_ptUVRIGHT,
           const CPoint3DCAMERA& p_vecPointXYZLEFT,
           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
           const MatrixProjection& p_matProjectionLEFT,
           const MatrixProjection& p_matProjectionRIGHT,
           const MatrixProjection& p_matProjectionWORLDtoLEFT,
           const MatrixProjection& p_matProjectionWORLDtoRIGHT,
           const UIDFrame& p_uFrame ): uID( p_uID ),
                                       matDescriptorReferenceLEFT( p_matDescriptorLEFT ),
                                       matDescriptorReferenceRIGHT( p_matDescriptorRIGHT ),
                                       dKeyPointSize( p_dKeyPointSize ),
                                       vecPointXYZInitial( p_matTransformationLEFTtoWORLD*p_vecPointXYZLEFT ),
                                       vecPointXYZOptimized( vecPointXYZInitial ),
                                       vecUVReferenceLEFT( p_ptUVLEFT.x, p_ptUVLEFT.y, 1.0 ),
                                       vecPointXYZMean( vecPointXYZInitial ),
                                       m_matProjectionLEFT( p_matProjectionLEFT ),
                                       m_matProjectionRIGHT( p_matProjectionRIGHT )
{
    vecDescriptorsLEFT.clear( );
    vecDescriptorsRIGHT.clear( );
    m_vecMeasurements.clear( );

    //ds construct filestring and open dump file
    //char chBuffer[256];
    //std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks/landmark%06lu.txt", uID );
    //m_pFilePositionOptimization = std::fopen( chBuffer, "w" );

    //assert( 0 != m_pFilePositionOptimization );

    //ds dump file format
    //std::fprintf( m_pFilePositionOptimization, "ID_FRAME | ID_LANDMARK | ITERATION MEASUREMENTS INLIERS | ERROR_ARSS | DELTA_XYZ |      X      Y      Z\n" );

    //ds add this position
    addMeasurement( p_uFrame,
                    p_ptUVLEFT,
                    p_ptUVRIGHT,
                    p_matDescriptorLEFT,
                    p_matDescriptorRIGHT,
                    p_vecPointXYZLEFT,
                    p_matTransformationLEFTtoWORLD,
                    p_matTransformationWORLDtoLEFT,
                    p_matProjectionWORLDtoLEFT,
                    p_matProjectionWORLDtoRIGHT );
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
                                const CDescriptor& p_matDescriptorLEFT,
                                const CDescriptor& p_matDescriptorRIGHT,
                                const CPoint3DCAMERA& p_vecXYZLEFT,
                                const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                const MatrixProjection& p_matProjectionWORLDtoLEFT,
                                const MatrixProjection& p_matProjectionWORLDtoRIGHT )
{
    //ds input validation
    assert( p_ptUVLEFT.y == p_ptUVRIGHT.y );
    assert( 0 < p_vecXYZLEFT.z( ) );

    //ds add to history
    vecDescriptorsLEFT.push_back( p_matDescriptorLEFT );
    vecDescriptorsRIGHT.push_back( p_matDescriptorRIGHT );

    //ds compute world point
    const CPoint3DWORLD vecXYZWORLD( p_matTransformationLEFTtoWORLD*p_vecXYZLEFT );

    //ds update mean
    vecPointXYZMean = ( vecPointXYZMean+vecXYZWORLD )/2.0;

    //ds add the measurement to structure
    m_vecMeasurements.push_back( new CMeasurementLandmark( uID,
                                                           p_ptUVLEFT,
                                                           p_ptUVRIGHT,
                                                           p_vecXYZLEFT,
                                                           vecXYZWORLD,
                                                           vecPointXYZOptimized,
                                                           p_matTransformationWORLDtoLEFT,
                                                           p_matProjectionWORLDtoLEFT,
                                                           p_matProjectionWORLDtoRIGHT,
                                                           uOptimizationsSuccessful ) );
}

void CLandmark::optimize( const UIDFrame& p_uFrame )
{
    //ds default false - gets set in optimization
    bIsOptimal = false;

    //ds update position
    //vecPointXYZOptimized = _getOptimizedLandmarkLEFT3D( p_uFrame, vecPointXYZOptimized );
    vecPointXYZOptimized = _getOptimizedLandmarkSTEREOUV( p_uFrame, vecPointXYZOptimized );
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkLEFT3D( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix3d matH( Eigen::Matrix3d::Zero( ) );
    Eigen::Vector3d vecB( Eigen::Vector3d::Zero( ) );
    const Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );
    double dErrorSquaredTotalMetersPREVIOUS = 0.0;

    //ds 3d point to optimize
    CPoint3DWORLD vecX( p_vecInitialGuess );

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::uCapIterations; ++uIteration )
    {
        //ds counts
        double dErrorSquaredTotalMeters = 0.0;
        uint32_t uInliers               = 0;

        //ds initialize setup
        matH.setZero( );
        vecB.setZero( );

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds get error
            const Eigen::Vector3d vecError( vecX-pMeasurement->vecPointXYZWORLD );

            //ds current error
            const double dErrorSquaredMeters = vecError.transpose( )*vecError;

            //ds check if outlier
            double dWeight = 1.0;
            if( 0.1 < dErrorSquaredMeters )
            {
                dWeight = 0.1/dErrorSquaredMeters;
            }
            else
            {
                ++uInliers;
            }
            dErrorSquaredTotalMeters += dWeight*dErrorSquaredMeters;

            //ds accumulate (special case as jacobian is the identity)
            matH += dWeight*matOmega;
            vecB += dWeight*vecError;
        }

        //ds update x solution
        vecX += matH.ldlt( ).solve( -vecB );

        //ds check if we have converged
        if( CLandmark::dConvergenceDelta > std::fabs( dErrorSquaredTotalMetersPREVIOUS-dErrorSquaredTotalMeters ) )
        {
            //ds compute average error
            const double dErrorSquaredAverageMeters = dErrorSquaredTotalMeters/m_vecMeasurements.size( );

            //ds if acceptable (don't mind about the actual number of inliers - we could be at this point with only 2 measurements -> 2 inliers -> 100%)
            if( 0 < uInliers )
            {
                //ds success
                ++uOptimizationsSuccessful;

                //ds update average
                dCurrentAverageSquaredError = dErrorSquaredAverageMeters;

                //ds check if optimal
                if( 0.075 > dErrorSquaredAverageMeters )
                {
                    bIsOptimal = true;
                }

                //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) [%06lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f (inliers: %u/%lu)\n",
                //             uID, uOptimizationsSuccessful, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredErrorPixels, uInliers, m_vecMeasurements.size( ) );

                return vecX;
            }
            else
            {
                ++uOptimizationsFailed;
                //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) landmark [%06lu] optimization failed - solution unacceptable (average error: %f, inliers: %u, iteration: %u)\n", uID, dErrorSquaredAverageMeters, uInliers, uIteration );

                //ds if still here the calibration did not converge - keep the initial estimate
                return p_vecInitialGuess;
            }
        }
        else
        {
            //ds update error
            dErrorSquaredTotalMetersPREVIOUS = dErrorSquaredTotalMeters;
        }
    }

    ++uOptimizationsFailed;
    //std::printf( "<CLandmark>(_getOptimizedLandmarkWORLD) landmark [%06lu] optimization failed - system did not converge\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
    return p_vecInitialGuess;
}

const CPoint3DWORLD CLandmark::_getOptimizedLandmarkSTEREOUV( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecInitialGuess )
{
    //ds initial values
    Eigen::Matrix4d matH( Eigen::Matrix4d::Zero( ) );
    Eigen::Vector4d vecB( Eigen::Vector4d::Zero( ) );
    CPoint3DHomogenized vecX( CMiniVisionToolbox::getHomogeneous( p_vecInitialGuess ) );
    //Eigen::Matrix2d matOmega( Eigen::Matrix2d::Identity( ) );
    double dErrorSquaredTotalPixelsPREVIOUS = 0.0;

    //ds iterations (break-out if convergence reached early)
    for( uint32_t uIteration = 0; uIteration < CLandmark::uCapIterations; ++uIteration )
    {
        //ds counts
        double dErrorSquaredTotalPixels = 0.0;
        uint32_t uInliers               = 0;

        //ds initialize setup
        matH.setZero( );
        vecB.setZero( );

        //ds do calibration over all recorded values
        for( const CMeasurementLandmark* pMeasurement: m_vecMeasurements )
        {
            //ds apply the projection to the transformed point
            const Eigen::Vector3d vecABCLEFT  = pMeasurement->matProjectionWORLDtoLEFT*vecX;
            const Eigen::Vector3d vecABCRIGHT = pMeasurement->matProjectionWORLDtoRIGHT*vecX;

            //ds buffer c value
            const double dCLEFT  = vecABCLEFT.z( );
            const double dCRIGHT = vecABCRIGHT.z( );

            //ds compute error
            const Eigen::Vector2d vecUVLEFT( vecABCLEFT.x( )/dCLEFT, vecABCLEFT.y( )/dCLEFT );
            const Eigen::Vector2d vecUVRIGHT( vecABCRIGHT.x( )/dCRIGHT, vecABCRIGHT.y( )/dCRIGHT );
            const Eigen::Vector4d vecError( vecUVLEFT.x( )-pMeasurement->ptUVLEFT.x,
                                            vecUVLEFT.y( )-pMeasurement->ptUVLEFT.y,
                                            vecUVRIGHT.x( )-pMeasurement->ptUVRIGHT.x,
                                            vecUVRIGHT.y( )-pMeasurement->ptUVRIGHT.y );

            //ds current error
            const double dErrorSquaredPixels = vecError.transpose( )*vecError;

            //std::printf( "[%06lu][%04u] error: %4.2f %4.2f %4.2f %4.2f (squared: %4.2f)\n", uID, uIteration, vecError(0), vecError(1), vecError(2), vecError(3) , dErrorSquaredPixels );

            //ds check if outlier
            double dWeight = 1.0;
            if( dKernelMaximumErrorSquaredPixels < dErrorSquaredPixels )
            {
                dWeight = dKernelMaximumErrorSquaredPixels/dErrorSquaredPixels;
            }
            else
            {
                ++uInliers;
            }
            dErrorSquaredTotalPixels += dWeight*dErrorSquaredPixels;

            //ds jacobian of the homogeneous division
            Eigen::Matrix< double, 2, 3 > matJacobianLEFT;
            matJacobianLEFT << 1/dCLEFT,          0, -vecABCLEFT.x( )/( dCLEFT*dCLEFT ),
                                      0,   1/dCLEFT, -vecABCLEFT.y( )/( dCLEFT*dCLEFT );

            Eigen::Matrix< double, 2, 3 > matJacobianRIGHT;
            matJacobianRIGHT << 1/dCRIGHT,           0, -vecABCRIGHT.x( )/( dCRIGHT*dCRIGHT ),
                                        0,   1/dCRIGHT, -vecABCRIGHT.y( )/( dCRIGHT*dCRIGHT );

            //ds final jacobian
            Eigen::Matrix< double, 4, 4 > matJacobian;
            matJacobian.setZero( );
            matJacobian.block< 2,4 >(0,0) = matJacobianLEFT*pMeasurement->matProjectionWORLDtoLEFT;
            matJacobian.block< 2,4 >(2,0) = matJacobianRIGHT*pMeasurement->matProjectionWORLDtoRIGHT;

            //ds precompute transposed
            const Eigen::Matrix< double, 4, 4 > matJacobianTransposed( matJacobian.transpose( ) );

            //ds accumulate
            matH += dWeight*matJacobianTransposed*matJacobian;
            vecB += dWeight*matJacobianTransposed*vecError;
        }

        //ds solve constrained system (since dx(3) = 0.0) and update x solution
        vecX.block< 3,1 >(0,0) += matH.block< 4, 3 >(0,0).householderQr( ).solve( -vecB );

        //std::printf( "[%06lu][%04u]: %6.2f %6.2f %6.2f %6.2f (delta 2norm: %f inliers: %u)\n", uID, uIteration, vecX.x( ), vecX.y( ), vecX.z( ), vecX(3), vecDeltaX.squaredNorm( ), uInliers );

        //std::fprintf( m_pFilePosition, "%04lu %06lu %03u %03lu %03u %6.2f\n", p_uFrame, uID, uIteration, m_vecMeasurements.size( ), uInliers, dRSSCurrent );

        //ds check if we have converged
        if( CLandmark::dConvergenceDelta > std::fabs( dErrorSquaredTotalPixelsPREVIOUS-dErrorSquaredTotalPixels ) )
        {
            //ds compute average error
            const double dErrorSquaredAveragePixels = dErrorSquaredTotalPixels/m_vecMeasurements.size( );

            //ds if acceptable inlier/outlier ratio
            if( CLandmark::dMinimumRatioInliersToOutliers < static_cast< double >( uInliers )/m_vecMeasurements.size( ) )
            {
                //ds success
                ++uOptimizationsSuccessful;

                //ds update average
                dCurrentAverageSquaredError = dErrorSquaredAveragePixels;

                //ds check if optimal
                if( dMaximumErrorSquaredAveragePixels > dErrorSquaredAveragePixels )
                {
                    bIsOptimal = true;
                }

                //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) [%06lu] converged (%2u) in %3u iterations to (%6.2f %6.2f %6.2f) from (%6.2f %6.2f %6.2f) ARSS: %6.2f (inliers: %u/%lu)\n",
                //             uID, uOptimizationsSuccessful, uIteration, vecX(0), vecX(1), vecX(2), p_vecInitialGuess(0), p_vecInitialGuess(1), p_vecInitialGuess(2), dCurrentAverageSquaredError, uInliers, m_vecMeasurements.size( ) );

                return vecX.block< 3,1 >(0,0);
            }
            else
            {
                ++uOptimizationsFailed;
                //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) landmark [%06lu] optimization failed - solution unacceptable (average error: %f, inliers: %u, iteration: %u)\n", uID, dErrorSquaredAveragePixels, uInliers, uIteration );

                //ds if still here the calibration did not converge - keep the initial estimate
                return p_vecInitialGuess;
            }
        }
        else
        {
            //ds update error
            dErrorSquaredTotalPixelsPREVIOUS = dErrorSquaredTotalPixels;
        }
    }

    ++uOptimizationsFailed;
    //std::printf( "<CLandmark>(_getOptimizedLandmarkSTEREOUV) landmark [%06lu] optimization failed - system did not converge\n", uID );

    //ds if still here the calibration did not converge - keep the initial estimate
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
        //ds get point into camera (cast to avoid eclipse error..)
        const CPoint3DHomogenized vecPointXYZLEFT( CMiniVisionToolbox::getHomogeneous( static_cast< CPoint3DCAMERA >( pMeasurement->matTransformationWORLDtoLEFT*vecPointXYZWORLD ) ) );

        //ds get projected point
        const CPoint2DHomogenized vecProjectionHomogeneous( m_matProjectionLEFT*vecPointXYZLEFT );

        //ds compute pixel coordinates TODO remove cast
        const Eigen::Vector2d vecUV = CWrapperOpenCV::getInterDistance( static_cast< Eigen::Vector2d >( vecProjectionHomogeneous.head< 2 >( )/vecProjectionHomogeneous(2) ), pMeasurement->ptUVLEFT );

        //ds compute squared error
        const double dSquaredError( vecUV.squaredNorm( ) );

        //ds add up
        dSumSquaredErrors += dSquaredError;
    }

    //ds average the measurement
    dCurrentAverageSquaredError = dSumSquaredErrors/m_vecMeasurements.size( );

    //ds if optimal
    if( 5.0 > dCurrentAverageSquaredError )
    {
        bIsOptimal = false;
    }

    //ds return
    return vecPointXYZWORLD;
}
