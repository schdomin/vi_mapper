#ifndef CMESSAGESYNCHRONIZER_H_
#define CMESSAGESYNCHRONIZER_H_

#include <thread>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "CStack.h"

class CMessageSynchronizer
{

//ds ctor/dtor
public:

    CMessageSynchronizer( ): m_bIsActive( false )
    {
        //ds nothing to do
    }
    ~CMessageSynchronizer( )
    {
        //ds nothing to do
    }

//ds members
private:

    //ds messages to synchronize
    CStack< txt_io::PinholeImageMessage > m_stackMessagesCamera_0;
    CStack< txt_io::PinholeImageMessage > m_stackMessagesCamera_1;
    CStack< txt_io::CIMUMessage > m_stackMessagesIMU;

    //ds control
    bool m_bIsActive;

    //ds callback function for triplets
    std::function< void( txt_io::PinholeImageMessage&, txt_io::PinholeImageMessage&, const txt_io::CIMUMessage& ) > m_cCallbackFunction;

//ds accessors
public:

    //ds set callback function to received synchronized messages
    void setSynchronizedMessageCallback( std::function< void( txt_io::PinholeImageMessage&, txt_io::PinholeImageMessage&, const txt_io::CIMUMessage& ) > p_cCallbackFunction )
    {
        //ds set callback function
        m_cCallbackFunction = p_cCallbackFunction;
    }

    std::thread startSynchronization( )
    {
        //ds set active
        m_bIsActive = true;

        //ds launch internal thread and return
        return std::thread( &CMessageSynchronizer::_filterMessageTriplets, this );
    }

    void stopSynchronization( )
    {
        m_bIsActive = false;
    }

    //ds pump messages into the synchronizer
    void addMessageCamera_0( txt_io::PinholeImageMessage* p_pMessage ){ m_stackMessagesCamera_0.push( *p_pMessage ); };
    void addMessageCamera_1( txt_io::PinholeImageMessage* p_pMessage ){ m_stackMessagesCamera_1.push( *p_pMessage ); };
    void addMessageIMU( const txt_io::CIMUMessage* p_pMessage ){ m_stackMessagesIMU.push( *p_pMessage ); };

//ds helpers
private:

    void _filterMessageTriplets( )
    {
        std::printf( "<CMessageSynchronizer>(_filterMessageTriplets) thread launched\n" );

        //ds hold steady
        while( m_bIsActive )
        {
            //ds as long as we have data in all the stacks - process
            while( !m_stackMessagesCamera_0.isEmpty( ) && !m_stackMessagesCamera_1.isEmpty( ) && !m_stackMessagesIMU.isEmpty( ) && m_bIsActive )
            {
                //ds pop the camera images
                txt_io::PinholeImageMessage cImageCamera_0( m_stackMessagesCamera_0.pop( ) );
                txt_io::PinholeImageMessage cImageCamera_1( m_stackMessagesCamera_1.pop( ) );

                //ds current triplet timestamp
                double dTimestamp_0( cImageCamera_0.timestamp( ) );
                double dTimestamp_1( cImageCamera_1.timestamp( ) );

                //ds sequence numbers have to match
                if( dTimestamp_0 == dTimestamp_1 )
                {
                    //ds get the most recent imu measurement
                    txt_io::CIMUMessage cMessageIMU( m_stackMessagesIMU.pop( ) );

                    //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                    while( dTimestamp_0 < cMessageIMU.timestamp( ) && !m_stackMessagesIMU.isEmpty( ) && m_bIsActive )
                    {
                        cMessageIMU = m_stackMessagesIMU.pop( );
                    }

                    //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                    while( dTimestamp_0 > cMessageIMU.timestamp( ) && m_bIsActive )
                    {
                        //ds pop the most recent imu measurement and check
                        if( !m_stackMessagesIMU.isEmpty( ) )
                        {
                            cMessageIMU = m_stackMessagesIMU.pop( );
                        }
                    }

                    //ds callback with triplet
                    m_cCallbackFunction( cImageCamera_1, cImageCamera_0, cMessageIMU );
                }
                else
                {
                    //ds skipped timestamp
                    double dTimestampSkipped = 0.0;

                    //ds if the first timestamp is older
                    if( dTimestamp_0 < dTimestamp_1 )
                    {
                        //ds save timestamp
                        dTimestampSkipped = dTimestamp_0;

                        //ds wait for the next stamp from 0
                        while( m_stackMessagesCamera_0.isEmpty( ) );

                        //ds pop the next item
                        cImageCamera_0 = m_stackMessagesCamera_0.pop( );

                        //ds update the timestamp
                        dTimestamp_0 = cImageCamera_0.timestamp( );
                    }
                    else
                    {
                        //ds save timestamp
                        dTimestampSkipped = dTimestamp_1;

                        //ds wait for the next stamp from 1
                        while( m_stackMessagesCamera_1.isEmpty( ) );

                        //ds pop the next item
                        cImageCamera_1 = m_stackMessagesCamera_1.pop( );

                        //ds update the timestamp
                        dTimestamp_1 = cImageCamera_1.timestamp( );
                    }

                    //ds sequence numbers have to match now TODO: remove boilerplate code
                    if( dTimestamp_0 == dTimestamp_1 )
                    {
                        //ds log skipped frame
                        std::printf( "<CMessageSynchronizer>(_filterMessageTriplets) WARNING: skipped single frame - timestamp: %.5lf s\n", dTimestampSkipped );

                        //ds get the most recent imu measurement
                        txt_io::CIMUMessage cMessageIMU( m_stackMessagesIMU.pop( ) );

                        //ds look for the matching timestamp in the stack (assuming that IMU messages have arrived in chronological order)
                        while( dTimestamp_0 < cMessageIMU.timestamp( ) && !m_stackMessagesIMU.isEmpty( ) && m_bIsActive )
                        {
                            cMessageIMU = m_stackMessagesIMU.pop( );
                        }

                        //ds in case we have not received the matching timestamp yet - TODO this blocks indefinitely if there is no IMU data arriving
                        while( dTimestamp_0 > cMessageIMU.timestamp( ) && m_bIsActive )
                        {
                            //ds pop the most recent imu measurement and check
                            if( !m_stackMessagesIMU.isEmpty( ) )
                            {
                                cMessageIMU = m_stackMessagesIMU.pop( );
                            }
                        }

                        //ds callback with triplet
                        m_cCallbackFunction( cImageCamera_1, cImageCamera_0, cMessageIMU );
                    }
                    else
                    {
                        std::printf( "<CMessageSynchronizer>(_filterMessageTriplets) ERROR: could not find matching frames, aborting\n" );
                        m_bIsActive = false;
                    }
                }
            }

            //ds flush buffer
            std::fflush( stdout );
        }

        std::printf( "<CMessageSynchronizer>(_filterMessageTriplets) thread terminated\n" );
    }

};

#endif //#define CMESSAGESYNCHRONIZER_H_
