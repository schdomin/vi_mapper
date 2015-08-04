#include "CLogger.h"

//ds initialization
std::FILE* CLogger::CLogLandmarkCreation::m_pFile       = 0;
std::FILE* CLogger::CLogLandmarkCreationMocked::m_pFile = 0;
std::FILE* CLogger::CLogTrajectory::m_pFile             = 0;
std::FILE* CLogger::CLogLandmarkFinal::m_pFile          = 0;
std::FILE* CLogger::m_pFileLandmarkFinalOptimized = 0;
std::FILE* CLogger::m_pFileDetectionEpipolar      = 0;
std::FILE* CLogger::m_pFileOdometry               = 0;

