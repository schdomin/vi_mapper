#ifndef CVIEWERSCENE_H
#define CVIEWERSCENE_H

#include <QGLViewer/qglviewer.h>
#include <QMatrix4x4>
#include <vector>
#include <map>

//ds custom
#include "types/CLandmark.h"

class CViewerScene: public QGLViewer
{

    //ds landmark type
    struct CLandmarkInScene
    {
        CLandmarkInScene( const UIDLandmark& p_uID,
                          const CPoint3DInWorldFrame& p_vecPositionXYZOriginal,
                          const CPoint3DInWorldFrame& p_vecPositionXYZOptimized ): uID( p_uID ),
                                                                             vecPositionXYZOriginal( p_vecPositionXYZOriginal.x( ), p_vecPositionXYZOriginal.y( ), p_vecPositionXYZOriginal.z( ) ),
                                                                             vecPositionXYZOptimized( p_vecPositionXYZOptimized.x( ), p_vecPositionXYZOptimized.y( ), p_vecPositionXYZOptimized.z( ) )
        {
            //ds nothing to do
        }

        const UIDLandmark uID;
        const qglviewer::Vec vecPositionXYZOriginal;
        qglviewer::Vec vecPositionXYZOptimized;
    };

public:

    CViewerScene( const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks );
    ~CViewerScene( );

protected:

    virtual void draw( );
    //virtual void keyPressEvent( QKeyEvent* p_pEvent );
    virtual void init( );
    virtual QString helpString( ) const;

public:

    void addKeyFrame( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< const CMeasurementLandmark* > > p_pLandmarks );
    void addFrame( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD );
    void addFrame( const QMatrix4x4& p_matTransformationLEFTtoWORLD );

private:

    //ds snippet of g2o: opengl_primitives
    void _drawBox(GLfloat l, GLfloat w, GLfloat h) const;

private:

    std::vector< qglviewer::Frame > m_vecFrames;
    std::map< UIDLandmark, CPoint3DInWorldFrame > m_mapLandmarks;

    std::shared_ptr< std::vector< const CMeasurementLandmark* > > m_pLiveMeasurements;

    //ds slandmarks
    const std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;

};

#endif //CVIEWERSCENE_H
