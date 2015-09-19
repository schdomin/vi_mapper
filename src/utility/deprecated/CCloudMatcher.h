#ifndef CCLOUDMATCHER_H
#define CCLOUDMATCHER_H

#include "types/TypesCloud.h"

class CCloudMatcher
{

//ds fields
private:

    static constexpr double m_dWeightEuclidian        = 10.0;
    static constexpr double m_dMatchingDistanceCutoff = 100.0;
    static constexpr double m_dMatchingDistanceInitial = 1e6;

public:

    static std::shared_ptr< const std::vector< CMatchCloud > > getMatches( const CDescriptorVectorPointCloud* p_pCloudQuery, const CDescriptorVectorPointCloud* p_pCloudReference )
    {
        std::shared_ptr< std::vector< CMatchCloud > > vecMatches( std::make_shared< std::vector< CMatchCloud > >( ) );

        //ds try to match the query into the training
        for( const CDescriptorVectorPoint3DWORLD& cPointQuery: p_pCloudQuery->vecPoints )
        {
            //ds current matching distance
            double dMatchingDistanceBest = 1e6;
            //UIDDescriptor uDescriptorMatchingsBest = 0;
            CPoint3DCAMERA vecPointXYZCAMERAMatch;

            //ds match this point against all training points
            for( const CDescriptorVectorPoint3DWORLD& cPointReference: p_pCloudReference->vecPoints )
            {
                //ds compute current matching distance against training point - EUCLIDIAN
                double dMatchingDistance = CCloudMatcher::m_dWeightEuclidian*( cPointQuery.vecPointXYZWORLD-cPointReference.vecPointXYZWORLD ).squaredNorm( );

                //ds get start and end iterators of training descriptors
                std::vector< CDescriptor >::const_iterator itDescriptorTrainStart( cPointReference.vecDescriptors.begin( ) );
                std::vector< CDescriptor >::const_iterator itDescriptorTrainEnd( cPointReference.vecDescriptors.end( )-1 );

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
                    if( cPointReference.vecDescriptors.end( ) == itDescriptorTrainStart || cPointReference.vecDescriptors.begin( ) == itDescriptorTrainEnd )
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
                    //uDescriptorMatchingsBest = uDescriptorMatchings;
                    vecPointXYZCAMERAMatch   = cPointReference.vecPointXYZCAMERA;
                }
            }

            //ds check if match
            if( CCloudMatcher::m_dMatchingDistanceCutoff > dMatchingDistanceBest )
            {
                //vecMatches->push_back( CMatchCloud( cPointQuery.uID, uIDMatchBest, cPointQuery.vecPointXYZCAMERA, vecPointXYZCAMERAMatch ) );
            }

            //std::printf( "(main) best match for query: %03lu -> train: %03lu (matching distance: %9.4f, descriptor matchings: %03lu)\n", cPointQuery.uID, uIDQueryBest, dMatchingDistanceBest, uDescriptorMatchingsBest );
        }

        return vecMatches;
    }

    static std::shared_ptr< const std::vector< CMatchCloud > > getMatches( const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudQuery,
                                                                           const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudReference )
    {
        return CCloudMatcher::getMatches( *p_vecCloudQuery, *p_vecCloudReference );
    }

    static std::shared_ptr< const std::vector< CMatchCloud > > getMatches( const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecCloudQuery,
                                                                           const std::vector< CDescriptorVectorPoint3DWORLD >& p_vecCloudReference )
    {
        std::shared_ptr< std::vector< CMatchCloud > > vecMatches( std::make_shared< std::vector< CMatchCloud > >( ) );

        //ds treated points
        std::set< UIDLandmark > vecMatchedIDs;

        //ds try to match the query into the training
        for( const CDescriptorVectorPoint3DWORLD& cPointQuery: p_vecCloudQuery )
        {
            //ds current matching distance
            double dMatchingDistanceBest = CCloudMatcher::m_dMatchingDistanceInitial;
            std::vector< CDescriptorVectorPoint3DWORLD >::const_iterator itPointReferenceBest;

            //ds match this point against all training remaining points
            for( std::vector< CDescriptorVectorPoint3DWORLD >::const_iterator itPointReference = p_vecCloudReference.begin( ); itPointReference != p_vecCloudReference.end( ); ++itPointReference )
            {
                //ds if not already added
                if( vecMatchedIDs.end( ) == vecMatchedIDs.find( itPointReference->uID ) )
                {
                    //ds compute current matching distance against training point - EUCLIDIAN
                    double dMatchingDistance = CCloudMatcher::m_dWeightEuclidian*( cPointQuery.vecPointXYZCAMERA-itPointReference->vecPointXYZCAMERA ).squaredNorm( );

                    //ds get start and end iterators of training descriptors
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainStart( itPointReference->vecDescriptors.begin( ) );
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainEnd( itPointReference->vecDescriptors.end( )-1 );

                    double dMatchingDistanceHammingForward   = 0.0;
                    double dMatchingDistanceHammingBackwards = 0.0;
                    UIDDescriptor uDescriptorMatchings       = 0;

                    //ds compute descriptor matching in ascending versus descending order
                    for( const CDescriptor& cDescriptorQuery: cPointQuery.vecDescriptors )
                    {
                        dMatchingDistanceHammingForward   += cv::norm( cDescriptorQuery, *itDescriptorTrainStart, cv::NORM_HAMMING );
                        dMatchingDistanceHammingBackwards += cv::norm( cDescriptorQuery, *itDescriptorTrainEnd, cv::NORM_HAMMING );
                        ++uDescriptorMatchings;

                        //ds move forward respective backwards
                        ++itDescriptorTrainStart;

                        //ds check if descriptor arrays overlap
                        if( itPointReference->vecDescriptors.end( ) == itDescriptorTrainStart || itPointReference->vecDescriptors.begin( ) == itDescriptorTrainEnd )
                        {
                            break;
                        }

                        //ds since begin returns still a valid element whereas end does not
                        --itDescriptorTrainEnd;
                    }

                    //ds average matchings
                    dMatchingDistanceHammingForward   /= uDescriptorMatchings;
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
                        dMatchingDistanceBest = dMatchingDistance;
                        itPointReferenceBest  = itPointReference;
                    }
                }
            }

            //ds check if match
            if( CCloudMatcher::m_dMatchingDistanceCutoff > dMatchingDistanceBest )
            {
                //ds add the match
                vecMatches->push_back( CMatchCloud( cPointQuery, CDescriptorVectorPoint3DWORLD( *itPointReferenceBest ), dMatchingDistanceBest ) );
                vecMatchedIDs.insert( itPointReferenceBest->uID );
            }

            //std::printf( "(main) best match for query: %03lu -> train: %03lu (matching distance: %9.4f, descriptor matchings: %03lu)\n", cPointQuery.uID, uIDQueryBest, dMatchingDistanceBest, uDescriptorMatchingsBest );
        }

        return vecMatches;
    }

};

#endif //CCLOUDMATCHER_H
