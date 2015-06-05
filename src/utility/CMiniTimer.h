#ifndef CMINITIMER_H
#define CMINITIMER_H

#include <chrono>
#include <ctime>

class CMiniTimer
{

//ds fields
private:

    static std::vector< std::chrono::time_point< std::chrono::system_clock > > vec_tmStart;

//ds methods
public:

    static const uint64_t tic( )
    {
        //ds set start time
        CMiniTimer::vec_tmStart.push_back( std::chrono::system_clock::now( ) );

        //ds return token to call toc
        return CMiniTimer::vec_tmStart.size( )-1;
    }
    static const double toc( const uint64_t& p_uIndex )
    {
        assert( p_uIndex < CMiniTimer::vec_tmStart.size( ) );

        //ds get stop time
        const std::chrono::time_point< std::chrono::system_clock > tmEnd( std::chrono::system_clock::now( ) );

        //ds return duration in seconds
        return ( std::chrono::duration< double >( tmEnd-CMiniTimer::vec_tmStart[p_uIndex] ) ).count( );
    }

};

#endif //CMINITIMER_H
