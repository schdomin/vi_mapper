#include "CViewerScene.h"

#include <QKeyEvent>
#include <QGLViewer/manipulatedFrame.h>

#include "optimization/CBridgeG2O.h"

CViewerScene::CViewerScene( const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                            const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_vecLandmarks( p_vecLandmarks ), m_vecKeyFrames( p_vecKeyFrames )
{
    //ds nothing to do
}
CViewerScene::~CViewerScene( )
{
    //ds nothing to do
}

void CViewerScene::draw()
{
    glPushAttrib( GL_ENABLE_BIT );
    glDisable( GL_LIGHTING );
    glPointSize( 5.0 );

    //ds draw WORLD coordinate frame
    glColor3f( 1.0, 1.0, 1.0 );
    glLineWidth( 1.0 );
    drawAxis( 0.5 );

    //ds first keyframe to draw
    CKeyFrame* pKeyFramePrevious( 0 );

    //ds buffer position
    CPoint3DWORLD vecPositionXYZPrevious( 0.0, 0.0, 0.0 );

    //ds draw a green dot to mark the start of the optimization
    glLineWidth( 2.5 );
    glPushMatrix( );
    glColor3f( 0.0, 1.0, 0.0 );
    glTranslatef( vecPositionXYZPrevious.x( ), vecPositionXYZPrevious.y( ), vecPositionXYZPrevious.z( ) );
    glBegin( GL_POINTS );
    glVertex3f( 0, 0, 0 );
    glEnd( );
    glPopMatrix( );

    //ds current ID
    UIDKeyFrame uCurrent = 0;

    //ds while the key frames are optimized
    while( m_vecKeyFrames->size( ) > uCurrent && m_vecKeyFrames->at( uCurrent )->bIsOptimized )
    {
        CKeyFrame* pKeyFrameCurrent( m_vecKeyFrames->at( uCurrent ) );

        //ds buffer positions
        const CPoint3DWORLD vecPositionXYZCurrent( pKeyFrameCurrent->matTransformationLEFTtoWORLD.translation( ) );

        //ds draw the line between the previous and current keyframe
        glColor3f( 0.25, 0.25, 1.0 );
        glBegin( GL_LINES );
        glVertex3f( vecPositionXYZPrevious.x( ), vecPositionXYZPrevious.y( ), vecPositionXYZPrevious.z( ) );
        glVertex3f( vecPositionXYZCurrent.x( ), vecPositionXYZCurrent.y( ), vecPositionXYZCurrent.z( ) );
        glEnd( );

        //ds draw a dot to mark the keyframe
        glPushMatrix( );
        glColor3f( 0.0, 1.0, 1.0 );
        glTranslatef( vecPositionXYZCurrent.x( ), vecPositionXYZCurrent.y( ), vecPositionXYZCurrent.z( ) );
        glBegin( GL_POINTS );
        glVertex3f( 0, 0, 0 );
        glEnd( );
        glPopMatrix( );

        pKeyFramePrevious      = pKeyFrameCurrent;
        vecPositionXYZPrevious = pKeyFramePrevious->matTransformationLEFTtoWORLD.translation( );
        ++uCurrent;
    }

    //ds always draw loop closure lines
    glLineWidth( 1.75 );
    glColor3f( 0.0, 1.0, 1.0 );
    glBegin( GL_LINES );
    for( CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        //ds draw loop closing line if set
        if( 0 != pKeyFrame->pLoopClosure )
        {
            const CPoint3DWORLD vecPositionXYZCurrent( pKeyFrame->matTransformationLEFTtoWORLD.translation( ) );
            const CPoint3DWORLD vecPositionLoopClosure( pKeyFrame->pLoopClosure->matTransformationLEFTtoWORLD.translation( ) );
            glVertex3f( vecPositionXYZCurrent.x( ), vecPositionXYZCurrent.y( ), vecPositionXYZCurrent.z( ) );
            glVertex3f( vecPositionLoopClosure.x( ), vecPositionLoopClosure.y( ), vecPositionLoopClosure.z( ) );
        }
    }
    glEnd( ); //GL_LINES

    //ds frame drawing start
    std::vector< std::pair< bool, qglviewer::Frame > >::size_type uStart = 0;

    //ds if we registered already some keyframes we have to move the head
    if( 0 != pKeyFramePrevious )
    {
        uStart = pKeyFramePrevious->uFrame;
    }

    //ds dodging if case for drawing the head coordinate frame
    qglviewer::Frame cFrameCurrent;

    glLineWidth( 2.5 );

    //ds draw all frames
    for( std::vector< std::pair< bool, qglviewer::Frame > >::size_type u = uStart+1; u < m_vecFrames.size( ); ++u )
    {
        //ds positions
        const qglviewer::Vec vecPositionPrevious( m_vecFrames[u-1].second.position( ) );
        const qglviewer::Vec vecPositionNow( m_vecFrames[u].second.position( ) );

        //ds draw the line between the previous and current keyframe
        glColor3f( 0.25, 0.25, 1.0 );
        glBegin( GL_LINES );
        glVertex3f( vecPositionPrevious.x, vecPositionPrevious.y, vecPositionPrevious.z );
        glVertex3f( vecPositionNow.x, vecPositionNow.y, vecPositionNow.z );
        glEnd( );

        //ds mark key frames
        if( m_vecFrames[u].first )
        {
            glPushMatrix( );
            glColor3f( 0.0, 1.0, 1.0 );
            glTranslatef( vecPositionNow.x, vecPositionNow.y, vecPositionNow.z );
            glBegin( GL_POINTS );
            glVertex3f( 0, 0, 0 );
            glEnd( );
            glPopMatrix( );
        }

        cFrameCurrent = m_vecFrames[u].second;
    }

    //ds orientation for head only
    glColor3f( 1.0, 1.0, 1.0 );
    glLineWidth( 1.5 );
    glPushMatrix( );
    glMultMatrixd( cFrameCurrent.matrix( ) );
    drawAxis( 0.25 );
    glPopMatrix( );

    //ds set line width and point size for landmarks
    glLineWidth( 1.0 );
    glPointSize( 2.0 );

    assert( 0 != m_vecLandmarks );

    //ds draw all existing landmarks so far
    for( const CLandmark* pLandmarkInScene: *m_vecLandmarks )
    {
        assert( 0 != pLandmarkInScene );

        //ds buffer positions
        const qglviewer::Vec vecPositionInitial( pLandmarkInScene->vecPointXYZInitial );
        const qglviewer::Vec vecPositionOptimized( pLandmarkInScene->vecPointXYZOptimized );

        //ds check if landmark is visible
        if( pLandmarkInScene->bIsCurrentlyVisible )
        {
            //ds draw the line between original and optimized
            glColor3f( 1.0, 0.1, 0.1 );
            glBegin( GL_LINES );
            glVertex3f( vecPositionInitial.x, vecPositionInitial.y, vecPositionInitial.z );
            glVertex3f( vecPositionOptimized.x, vecPositionOptimized.y, vecPositionOptimized.z );
            glEnd( );

            //ds draw optimized position
            glPushMatrix( );
            glColor3f( 0.0, 1.0, 0.0 );
            glTranslatef( vecPositionOptimized.x, vecPositionOptimized.y, vecPositionOptimized.z );
            glBegin( GL_POINTS );
            glVertex3f( 0, 0, 0 );
            glEnd( );
            glPopMatrix( );

            //ds draw original position
            glPushMatrix( );
            glColor3f( 0.75, 0.75, 0.75 );
            glTranslatef( vecPositionInitial.x, vecPositionInitial.y, vecPositionInitial.z );
            glBegin( GL_POINTS );
            glVertex3f( 0, 0, 0 );
            glEnd( );
            glPopMatrix( );
        }
        else
        {
            //ds draw landmark if valid
            if( CBridgeG2O::isOptimized( pLandmarkInScene ) )
            {
                //ds enable transparency
                glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
                glEnable( GL_BLEND );

                //ds draw optimized position
                glPushMatrix( );
                glColor4f( 0.75, 0.75, 0.75, 0.25 );
                glTranslatef( vecPositionOptimized.x, vecPositionOptimized.y, vecPositionOptimized.z );
                glBegin( GL_POINTS );
                glVertex3f( 0, 0, 0 );
                glEnd( );
                glPopMatrix( );

                //ds disable transparency
                glDisable( GL_BLEND );
            }
        }
    }

    glPopAttrib( );

    /*ds enable transparency
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );

    //ds draw all existing landmarks
    for( const std::pair< UIDLandmark, CPoint3DInWorldFrame >& cLandmarkInScene: m_mapLandmarks )
    {
        //ds draw optimized position
        glColor3f( 0.5, 0.5, 0.5 );
        glPushMatrix( );
        glTranslatef( cLandmarkInScene.second.x( ), cLandmarkInScene.second.y( ), cLandmarkInScene.second.z( ) );
        _drawBox( 0.02, 0.02, 0.02 );
        glPopMatrix( );
    }

    //ds disable transparency
    glDisable( GL_BLEND );

    //ds check if we have measurements
    if( 0 != m_pLiveMeasurements )
    {
        //ds loop over currently live measurements
        for( const CMeasurementLandmark* pLandmark: *m_pLiveMeasurements )
        {
            //ds buffer positions
            const qglviewer::Vec vecPositionOriginal( pLandmark->vecPointXYZWORLD );
            const qglviewer::Vec vecPositionOptimized( pLandmark->vecPointXYZWORLDOptimized );

            //ds draw the line between original and optimized
            glColor3f( 1.0, 0.1, 0.1 );
            glLineWidth( 1.0 );
            glPushAttrib( GL_ENABLE_BIT );
            glDisable( GL_LIGHTING );
            glBegin( GL_LINES );
            glVertex3f( vecPositionOriginal.x, vecPositionOriginal.y, vecPositionOriginal.z );
            glVertex3f( vecPositionOptimized.x, vecPositionOptimized.y, vecPositionOptimized.z );
            glEnd( );
            glPopAttrib( );

            //ds draw original position
            glColor3f( 0.75, 0.75, 0.75 );
            glPushMatrix( );
            glTranslatef( vecPositionOriginal.x, vecPositionOriginal.y, vecPositionOriginal.z );
            _drawBox( 0.025, 0.025, 0.025 );
            glPopMatrix( );

            //ds draw optimized position
            glColor3f( 0.25, 1.0, 0.25 );
            glPushMatrix( );
            glTranslatef( vecPositionOptimized.x, vecPositionOptimized.y, vecPositionOptimized.z );
            _drawBox( 0.025, 0.025, 0.025 );
            glPopMatrix( );

            try
            {
                //ds check if the live measurement updates a existing landmark
                m_mapLandmarks.at( pLandmark->uID ) = pLandmark->vecPointXYZWORLDOptimized;
            }
            catch( const std::out_of_range& p_cException )
            {
                //ds landmark not found, we can add it
                m_mapLandmarks.insert( std::pair< UIDLandmark, CPoint3DInWorldFrame >( pLandmark->uID, pLandmark->vecPointXYZWORLDOptimized ) );
            }
        }
    }*/
}

void CViewerScene::init( )
{
    //ds initialize
    m_vecFrames.clear( );
    //m_mapLandmarks.clear( );
    //restoreStateFromFile( );
    setSceneRadius( 25.0 );
}

QString CViewerScene::helpString( ) const
{
  QString text("<h2>HUBBA BABA</h2>");
  text += "A <i>KeyFrameInterpolator</i> holds an interpolated path defined by key frames. ";
  text += "It can then smoothly make its associed frame follow that path. Key frames can interactively be manipulated, even ";
  text += "during interpolation.<br><br>";
  text += "Note that the camera holds 12 such keyFrameInterpolators, binded to F1-F12. Press <b>Alt+Fx</b> to define new key ";
  text += "frames, and then press <b>Fx</b> to make the camera follow the path. Press <b>C</b> to visualize these paths.<br><br>";
  text += "<b>+/-</b> changes the interpolation speed. Negative values are allowed.<br><br>";
  text += "<b>Return</b> starts-stops the interpolation.<br><br>";
  text += "Use the left and right arrows to change the manipulated KeyFrame. ";
  text += "Press <b>Control</b> to move it or simply hover over it.";
  return text;
}

/*void CViewerScene::addKeyFrame( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< const CMeasurementLandmark* > > p_pLandmarks )
{
    //ds position
    const CPoint3DInWorldFrame vecPosition( p_matTransformationLEFTtoWORLD.translation( ) );
    const Eigen::Quaterniond vecQuaternion( p_matTransformationLEFTtoWORLD.linear( ) );

    //ds setup the new frame
    qglviewer::Frame cFrameNew;
    cFrameNew.setPosition( vecPosition.x( ), vecPosition.y( ), vecPosition.z( ) );
    cFrameNew.setOrientation( vecQuaternion.x( ), vecQuaternion.y( ), vecQuaternion.z( ), vecQuaternion.w( ) );

    //ds add it to the vector
    m_vecFrames.push_back( cFrameNew );

    //ds set live measurements
    m_pLiveMeasurements = p_pLandmarks;

    //ds force redraw
    draw( );
    updateGL( );
}*/

void CViewerScene::addFrame( const std::pair< bool, Eigen::Isometry3d > p_prFrame )
{
    //ds position
    const CPoint3DWORLD vecPosition( p_prFrame.second.translation( ) );
    const Eigen::Quaterniond vecQuaternion( p_prFrame.second.linear( ) );

    //ds setup the new frame
    qglviewer::Frame cFrameNew;
    cFrameNew.setPosition( vecPosition.x( ), vecPosition.y( ), vecPosition.z( ) );
    cFrameNew.setOrientation( vecQuaternion.x( ), vecQuaternion.y( ), vecQuaternion.z( ), vecQuaternion.w( ) );

    //ds add it to the vector
    m_vecFrames.push_back( std::pair< bool, qglviewer::Frame >( p_prFrame.first, cFrameNew ) );
}

void CViewerScene::updateFrame( const UIDFrame& p_uID, const bool& p_bIsKeyFrame )
{
    //ds modify the vector
    m_vecFrames[p_uID].first = p_bIsKeyFrame;
}

void CViewerScene::manualDraw( )
{
    draw( );
    updateGL( );
}

/*void CViewerScene::keyPressEvent( QKeyEvent* p_pEvent )
{
    switch( p_pEvent->key( ) )
    {
        default:
        {
            QGLViewer::close( );
            break;
        }
    }
}*/

void CViewerScene::_drawBox(GLfloat l, GLfloat w, GLfloat h) const
{
    GLfloat sx = l*0.5f;
    GLfloat sy = w*0.5f;
    GLfloat sz = h*0.5f;

    glBegin(GL_QUADS);
    // bottom
    glNormal3f( 0.0f, 0.0f,-1.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, -sy, -sz);
    // top
    glNormal3f( 0.0f, 0.0f,1.0f);
    glVertex3f(-sx, -sy, sz);
    glVertex3f(-sx, sy, sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(sx, -sy, sz);
    // back
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(-sx, sy, sz);
    glVertex3f(-sx, -sy, sz);
    // front
    glNormal3f( 1.0f, 0.0f, 0.0f);
    glVertex3f(sx, -sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(sx, -sy, sz);
    // left
    glNormal3f( 0.0f, -1.0f, 0.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(sx, -sy, -sz);
    glVertex3f(sx, -sy, sz);
    glVertex3f(-sx, -sy, sz);
    //right
    glNormal3f( 0.0f, 1.0f, 0.0f);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(-sx, sy, sz);
    glEnd();
}
