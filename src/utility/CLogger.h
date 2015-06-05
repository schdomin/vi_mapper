#ifndef CLOGGER_H
#define CLOGGER_H

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

};

#endif //CLOGGER_H
