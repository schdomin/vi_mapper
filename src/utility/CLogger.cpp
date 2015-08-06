#include "CLogger.h"

//ds initialization
std::FILE* CLogger::CLogLandmarkCreation::m_pFile       = 0;
std::FILE* CLogger::CLogLandmarkCreationMocked::m_pFile = 0;
std::FILE* CLogger::CLogTrajectory::m_pFile             = 0;
std::FILE* CLogger::CLogLandmarkFinal::m_pFile          = 0;
std::FILE* CLogger::CLogLandmarkFinalOptimized::m_pFile = 0;
std::FILE* CLogger::CLogDetectionEpipolar::m_pFile      = 0;
std::FILE* CLogger::CLogOptimizationOdometry::m_pFile   = 0;
std::FILE* CLogger::CLogLinearAcceleration::m_pFile     = 0;
