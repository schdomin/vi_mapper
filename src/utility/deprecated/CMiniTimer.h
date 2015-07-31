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

    static const std::string getTimestamp( )
    {
        //ds current time
        const std::time_t tmCurrent = std::time( NULL );

        //ds compute stamp and return
        char chBufferTimestamp[100];
        std::strftime( chBufferTimestamp, sizeof( chBufferTimestamp ), "%Y-%m-%d-%H%M%S", std::localtime( &tmCurrent ) );
        return chBufferTimestamp;
    }

};

#endif //CMINITIMER_H
