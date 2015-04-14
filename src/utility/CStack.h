#ifndef CSTACK_H_
#define CSTACK_H_

//ds std
#include <atomic>

template< typename T > class CStack
{

//ds ctor/dtor
public:

    CStack< T >( ): m_pNodeHead( 0 )
    {
        //ds nothing to do
    }
    ~CStack< T >( )
    {
        //ds free all remaining nodes - get current head situation
        SNode* pCurrentNodeHead = m_pNodeHead.load( std::memory_order_relaxed );

        //ds while the current head is set
        while( 0 != pCurrentNodeHead )
        {
            //ds remember the next node
            SNode* pNodeNext = pCurrentNodeHead->m_pNodeNext;

            //ds free the current node
            delete pCurrentNodeHead;

            //ds update the head
            pCurrentNodeHead = pNodeNext;
        }
    }

//ds members
private:

    struct SNode
    {
        //ds ctor
        SNode( const T& p_tData ) : m_tData( p_tData ), m_pNodeNext( 0 ){ }

        //ds members
        T m_tData;
        SNode* m_pNodeNext;
    };

    //ds the head
    std::atomic< SNode * > m_pNodeHead;

//ds accessors
public:

    //ds stack push function
    void push( const T& p_tData )
    {
        //ds allocate a new node
        SNode* pNodeNew = new SNode( p_tData );

        //ds put the current value of head into new_node->next
        pNodeNew->m_pNodeNext = m_pNodeHead.load( std::memory_order_relaxed );

        //ds try to add the node
        while( !m_pNodeHead.compare_exchange_weak( pNodeNew->m_pNodeNext, pNodeNew, std::memory_order_release, std::memory_order_relaxed ) );
    }

    //ds stack pop function
    const T pop( )
    {
        //ds backup the head
        SNode* pOldNodeHead = m_pNodeHead.load( std::memory_order_relaxed );

        //ds try to set the head back
        while( !m_pNodeHead.compare_exchange_weak( pOldNodeHead, pOldNodeHead->m_pNodeNext, std::memory_order_release, std::memory_order_relaxed ) );

        //ds get a copy of the data from the head
        const T tData = pOldNodeHead->m_tData;

        //ds free the node to return from the stack
        delete pOldNodeHead;

        //ds return the data
        return tData;
    }

    //ds empty query - if head is not set
    const bool isEmpty( ) const { return ( 0 == m_pNodeHead.load( std::memory_order_relaxed ) ); };

};

#endif //#define CSTACK_H_
