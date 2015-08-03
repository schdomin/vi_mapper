#include <QtCore>
#include <memory>

#include "gui/CViewerScene.h"
#include "types/CLandmark.h"

class CThreadGUI: public QThread
{

public:

    CThreadGUI( const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks ): m_pViewer( new CViewerScene( p_vecLandmarks ) )
    {

    }
    ~CThreadGUI( )
    {
        //ds nothing to do
    }

private:

    CViewerScene* m_pViewer;
    bool m_bIsViewerActive = false;

protected:

    virtual void run( )
    {
        //ds start viewer
        m_pViewer->setWindowTitle( "CViewerScene: WORLD" );
        m_pViewer->showMaximized( );

        //ds enable thread
        m_bIsViewerActive = true;
        std::printf( "<CThreadGUI>(run) launched\n" );

        //ds stay alive
        while( m_bIsViewerActive && m_pViewer->isVisible( ) )
        {
            //ds sleep to avoid 100% cpu
            msleep( 1 );
        }

        //ds shutdown
        m_pViewer->close( );
        std::printf( "<CThreadGUI>(run) stopped\n" );
    }

public:

    //ds QT macro for thread communication
    //public slots:
    void updateFrame( const std::pair< bool, Eigen::Isometry3d > p_prFrame ){ m_pViewer->addFrame( p_prFrame ); }
    void close( ){ m_bIsViewerActive = false; }
};
