#ifndef CLOGGER_H
#define CLOGGER_H

#include <chrono>

class CLogger
{

public:

    static void openBox( )
    {
        std::printf( "|''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''|\n" );
    }

    static void closeBox( )
    {
        std::printf( "|..........................................................................................................|\n" );
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
    static const double getTimeSeconds( )
    {
        return std::chrono::system_clock::now( ).time_since_epoch( ).count( )/1e9;
    }

};

#endif //CLOGGER_H
