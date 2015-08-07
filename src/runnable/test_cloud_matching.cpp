#include <iostream>

//ds custom
#include "utility/CCloudStreamer.h"

int main( int argc, char** argv )
{
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds if no clouds provided (at least two)
    if( 3 > argc )
    {
        std::printf( "(main) insufficient amount of clouds (provide at least 2)\n" );
        std::printf( "(main) calling syntax: test_cloud_matching <cloud> <cloud>\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout);
        return 0;
    }

    //ds clouds
    std::vector< CDescriptorPointCloud > vecClouds;

    try
    {
        //ds load clouds
        while( 1 < argc-vecClouds.size( ) )
        {
            std::printf( "(main) loading cloud: %s - ", argv[vecClouds.size( )+1] );
            vecClouds.push_back( CCloudstreamer::loadCloud( argv[vecClouds.size( )+1], vecClouds.size( ) ) );
            std::printf( "points: %lu\n", vecClouds.back( ).vecPoints.size( ) );
        }
    }
    catch( const CExceptionInvalidFile& p_cException )
    {
        std::printf( "\n(main) unable to load cloud, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout);
        return 0;
    }

    std::printf( "(main) successfully loaded %lu clouds\n", vecClouds.size( ) );
    const double dWeightEuclidian        = 100.0;
    const double dMatchingDistanceCutoff = 25.0;

    //ds match the query clouds
    const CDescriptorPointCloud& cCloudQuery = vecClouds.back( );

    //ds against one training cloud each
    for( const CDescriptorPointCloud& cCloudTrain: vecClouds )
    {
        //ds if not the same cloud
        if( cCloudTrain.uID != cCloudQuery.uID )
        {
            //ds count matches
            UIDLandmark uMatches = 0;

            //ds try to match the query into the training
            for( const CDescriptorPoint3DWORLD& cPointQuery: cCloudQuery.vecPoints )
            {
                //ds current matching distance
                double dMatchingDistanceBest           = 1e6;
                //UIDDescriptorPoint3D uIDQueryBest      = 0;
                //UIDDescriptor uDescriptorMatchingsBest = 0;

                //ds match this point against all training points
                for( const CDescriptorPoint3DWORLD cPointTrain: cCloudTrain.vecPoints )
                {
                    //ds compute current matching distance against training point - EUCLIDIAN
                    double dMatchingDistance = dWeightEuclidian*( cPointQuery.vecPointXYZWORLD-cPointTrain.vecPointXYZWORLD ).squaredNorm( );

                    //ds get start and end iterators of training descriptors
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainStart( cPointTrain.vecDescriptors.begin( ) );
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainEnd( cPointTrain.vecDescriptors.end( )-1 );

                    double dMatchingDistanceHammingForward   = 0.0;
                    double dMatchingDistanceHammingBackwards = 0.0;
                    UIDDescriptor uDescriptorMatchings       = 0;

                    //ds compute descriptor matching in ascending versus descending order
                    for( const CDescriptor cDescriptorQuery: cPointQuery.vecDescriptors )
                    {
                        dMatchingDistanceHammingForward   += cv::norm( cDescriptorQuery, *itDescriptorTrainStart, cv::NORM_HAMMING );
                        dMatchingDistanceHammingBackwards += cv::norm( cDescriptorQuery, *itDescriptorTrainEnd, cv::NORM_HAMMING );
                        ++uDescriptorMatchings;

                        //ds move forward respective backwards
                        ++itDescriptorTrainStart;

                        //ds check if descriptor arrays overlap
                        if( cPointTrain.vecDescriptors.end( ) == itDescriptorTrainStart || cPointTrain.vecDescriptors.begin( ) == itDescriptorTrainEnd )
                        {
                            break;
                        }

                        //ds since begin returns still a valid element whereas end does not
                        --itDescriptorTrainEnd;
                    }

                    //ds average matchings
                    dMatchingDistanceHammingForward /= uDescriptorMatchings;
                    dMatchingDistanceHammingBackwards /= uDescriptorMatchings;

                    //std::printf( "matching forward: %f backwards: %f (matchings: %u)\n", dMatchingDistanceHammingForward, dMatchingDistanceHammingBackwards, uDescriptorMatchings );

                    //ds take the best run
                    if( dMatchingDistanceHammingForward < dMatchingDistanceHammingBackwards )
                    {
                        dMatchingDistance += dMatchingDistanceHammingForward;
                    }
                    else
                    {
                        dMatchingDistance += dMatchingDistanceHammingBackwards;
                    }

                    //ds check if better
                    if( dMatchingDistanceBest > dMatchingDistance )
                    {
                        dMatchingDistanceBest    = dMatchingDistance;
                        //uIDQueryBest             = cPointTrain.uID;
                        //uDescriptorMatchingsBest = uDescriptorMatchings;
                    }
                }

                //ds check if match
                if( dMatchingDistanceCutoff > dMatchingDistanceBest )
                {
                    ++uMatches;
                }

                //std::printf( "(main) best match for query: %03lu -> train: %03lu (matching distance: %9.4f, descriptor matchings: %03lu)\n", cPointQuery.uID, uIDQueryBest, dMatchingDistanceBest, uDescriptorMatchingsBest );
            }

            std::printf( "(main) cloud [%02lu] -> [%02lu] - matches: %06lu/%06lu\n", cCloudQuery.uID, cCloudTrain.uID, uMatches, cCloudQuery.vecPoints.size( ) );
        }
    }

    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}
