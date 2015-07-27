#ifndef CEXCEPTIONENDOFFILE_H_
#define CEXCEPTIONENDOFFILE_H_

#include <exception>

class CExceptionEndOfFile: public std::exception
{

public:

    CExceptionEndOfFile( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionEndOfFile( )
    {

    }

private:

    const std::string m_strExceptionDescription;

public:

    virtual const char* what( ) const throw( )
    {
        return m_strExceptionDescription.c_str( );
    }
};

#endif //CEXCEPTIONENDOFFILE_H_
