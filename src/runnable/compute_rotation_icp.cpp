#include <iostream>
#include <fstream>

//ds custom
#include "utility/CLogger.h"

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds arguments: ground_truth vi_mapper_trajectory_file
    if( 3 != argc )
    {
        std::printf( "(main) please specify the arguments <trajectory_ground_truth> <trajectory_vi_mapper_RAW>\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds files
    const std::string strInfileReference = argv[1];
    const std::string strInfileTest      = argv[2];
    const double dMaximumErrorForInlier  = 0.01; //100; //atof( argv[3] );

    //ds open files
    std::ifstream ifReference( strInfileReference, std::ifstream::in );
    std::ifstream ifTest( strInfileTest, std::ifstream::in );

    //ds on failure
    if( !ifTest.good( ) )
    {
        std::printf( "(main) unable to open trajectory files\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds log configuration
    CLogger::openBox( );
    std::printf( "(main) strInfileReference     := '%s'\n", strInfileReference.c_str( ) );
    std::printf( "(main) strInfileTest          := '%s'\n", strInfileTest.c_str( ) );
    std::printf( "(main) dMaximumErrorForInlier := '%s'\n", std::to_string( dMaximumErrorForInlier ).c_str( ) );
    CLogger::closeBox( );

    //ds pose position vectors to align
    std::vector< Eigen::Vector3d > vecPositionsReference( 0 );
    std::vector< Eigen::Vector3d > vecPositionsTest( 0 );
    std::vector< Eigen::Vector3d > vecPositionsReferenceDownsampled( 0 );

    //ds parse both files
    try
    {
        //ds reference set
        while( ifReference.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifReference, strLineBuffer );

            //ds if we read something
            if( !strLineBuffer.empty( ) )
            {
                //ds get it to a stringstream
                std::istringstream issLine( strLineBuffer );

                //ds information fields (KITTI format)
                Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );
                for( uint8_t u = 0; u < 3; ++u )
                {
                    for( uint8_t v = 0; v < 4; ++v )
                    {
                        issLine >> matTransformationRAW(u,v);
                    }
                }

                //ds compute for kitti system
                const Eigen::Isometry3d matTransformationNOWtoWORLD( matTransformationRAW );

                //ds store translation
                vecPositionsReference.push_back( matTransformationNOWtoWORLD.translation( ) );
            }
        }

        //ds then test
        while( ifTest.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifTest, strLineBuffer );

            //ds if we read something
            if( !strLineBuffer.empty( ) )
            {
                //ds get it to a stringstream
                std::istringstream issLine( strLineBuffer );

                //ds parse current frame id
                UIDFrame uIDCurrent = 0;
                issLine >> uIDCurrent;

                //ds information fields (KITTI format)
                Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );
                for( uint8_t u = 0; u < 3; ++u )
                {
                    for( uint8_t v = 0; v < 4; ++v )
                    {
                        issLine >> matTransformationRAW(u,v);
                    }
                }

                //ds compute for kitti system
                const Eigen::Isometry3d matTransformationNOWtoWORLD( matTransformationRAW );

                //ds store translation
                vecPositionsTest.push_back( matTransformationNOWtoWORLD.translation( ) );
                vecPositionsReferenceDownsampled.push_back( vecPositionsReference[uIDCurrent] );
            }
        }

        //ds check
        assert( vecPositionsReferenceDownsampled.size( ) == vecPositionsTest.size( ) );
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "(main) ERROR: unable to parse trajectory files, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        return 1;
    }

    //ds success
    std::printf( "(main) successfully loaded poses: %lu/%lu\n", vecPositionsReferenceDownsampled.size( ), vecPositionsReference.size( ) );

    //ds initial guess
    Eigen::Matrix3d matRotationToKITTI;
    matRotationToKITTI << 1, 0,  0,
                          0, 0, -1,
                          0, 1,  0;

    std::printf( "\ninitial rotation: \n" );
    std::cout << "\n" << matRotationToKITTI << "\n" << std::endl;

    Eigen::Isometry3d matTransformationToKITTI( Eigen::Matrix4d::Identity( ) );
    matTransformationToKITTI.linear( ) = matRotationToKITTI;

    //ds convergence criteria
    const double dErrorDeltaForConvergence      = 1e-5;
    double dErrorSquaredTotalPrevious           = 0.0;

    //ds LS setup
    Eigen::Matrix< double, 6, 6 > matH;
    Eigen::Matrix< double, 6, 1 > vecB;
    Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

    //ds run least-squares maximum 100 times
    for( uint8_t uLS = 0; uLS < 10000; ++uLS )
    {
        //ds error
        double dErrorSquaredTotal = 0.0;
        uint32_t uInliers         = 0;

        //ds initialize setup
        matH.setZero( );
        vecB.setZero( );

        //ds for all the points
        for( std::vector< Eigen::Vector3d >::size_type u = 0; u < vecPositionsReferenceDownsampled.size( ); ++u )
        {
            //ds compute projection into closure
            const Eigen::Vector3d vecPositionTest( matTransformationToKITTI*vecPositionsTest[u] );

            //std::cout << " test: " << vecPositionTest.transpose( ) << std::endl;
            //std::cout << "truth: " << vecPositionsReferenceDownsampled[u].transpose( ) << std::endl;

            //ds set y omega
            //matOmega(1,1) = 1.0/( vecPositionsReferenceDownsampled[u].norm( ) );

            //ds compute error
            const Eigen::Vector3d vecError( ( vecPositionTest-vecPositionsReferenceDownsampled[u] )/vecPositionsReferenceDownsampled[u].norm( ) );
            const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

            //ds check if outlier
            double dWeight = 1.0;
            if( dMaximumErrorForInlier < dErrorSquared )
            {
                dWeight = dMaximumErrorForInlier/dErrorSquared;
            }
            else
            {
                ++uInliers;
            }
            dErrorSquaredTotal += dWeight*dErrorSquared;

            //std::cout << "error squared: " << dWeight*dErrorSquared << " weight: "<< dWeight <<std::endl;
            //getchar( );

            //ds skew matrix
            Eigen::Matrix3d matSkewPosition;

            //ds formula: http://en.wikipedia.org/wiki/Cross_product#Skew-symmetric_matrix
            matSkewPosition <<             0.0, -vecPositionTest(2),  vecPositionTest(1),
                            vecPositionTest(2),                 0.0, -vecPositionTest(0),
                           -vecPositionTest(1),  vecPositionTest(0),                 0.0;

            //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
            Eigen::Matrix< double, 3, 6 > matJacobianTransform;
            matJacobianTransform.setZero( );
            matJacobianTransform.block<3,3>(0,0).setIdentity( );
            matJacobianTransform.block<3,3>(0,3) = -2*matSkewPosition;

            //ds precompute transposed
            const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

            //ds accumulate
            matH += dWeight*matJacobianTransformTransposed*matOmega*matJacobianTransform;
            vecB += dWeight*matJacobianTransformTransposed*matOmega*vecError;
        }

        //ds result
        const Eigen::Matrix< double, 6, 1 > vecLinearizedSolution( matH.ldlt( ).solve( -vecB ) );

        //ds resulting transformation matrix
        Eigen::Isometry3d matTransformationSolution( Eigen::Matrix4d::Identity( ) );

        //ds translation
        matTransformationSolution.translation( ) = vecLinearizedSolution.head<3>( );

        //ds rotation
        double dW = vecLinearizedSolution.block<3,1>(3,0).squaredNorm( );

        //ds if not at unity
        if( 1.0 > dW )
        {
            //ds normalize and set
            dW = sqrt( 1.0 - dW );
            matTransformationSolution.linear( ) = Eigen::Quaterniond( dW, vecLinearizedSolution( 3 ), vecLinearizedSolution( 4 ), vecLinearizedSolution( 5 ) ).toRotationMatrix( );
        }
        else
        {
            matTransformationSolution.linear( ).setIdentity( );
        }

        //ds update transform
        matTransformationToKITTI = matTransformationSolution*matTransformationToKITTI;

        //ds enforce rotation symmetry
        const Eigen::Matrix3d matRotation        = matTransformationToKITTI.linear( );
        Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
        matRotationSquared.diagonal( ).array( ) -= 1;
        matTransformationToKITTI.linear( )      -= 0.5*matRotation*matRotationSquared;

        //ds check if converged (no descent required)
        if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
        {
            std::printf( "\nsystem converged in %u iterations (error: %f inliers: %u) - final rotation: \n", uLS, dErrorSquaredTotal, uInliers );
            std::cout << "\n" << matTransformationToKITTI.matrix( ) << "\n" << std::endl;
            std::cout << "linear:";

            for( uint8_t a = 0; a < 4; ++a )
            {
                for( uint8_t b = 0; b < 4; ++b )
                {
                    std::cout << " " << matTransformationToKITTI(a,b);
                }
            }

            std::cout << std::endl;

            break;
        }
        else
        {
            std::printf( "iteration %u error: %f inliers: %u\n", uLS, dErrorSquaredTotal, uInliers );
            dErrorSquaredTotalPrevious = dErrorSquaredTotal;
        }

        //ds if not converged
        if( 99 == uLS )
        {
            std::printf( "system did not converge\n" );
        }
    }

    //ds close infiles
    ifReference.close( );
    ifTest.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    return 0;
}
