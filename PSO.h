#include <vector>

#ifndef __PSO_H_INCLUDED__
#define __PSO_H_INCLUDED__

typedef float (*callback_function)(const std::vector<float>&,const std::vector<void*>&);

class PSO
{
public:
    static std::vector<float> ParticleSwarmOptimization(float,float,float,int,int,int,callback_function,std::vector<void*>);
    static std::vector<float> ParticleSwarmOptimiationInjectPositions(float,float,float,int,int,int,callback_function,std::vector<void*>, std::vector<float>);
    static std::vector<float> ParticleSwarmOptimizationGbest(float,float,float,int,int,int,callback_function,std::vector<void*>);
    static std::vector<float> ParticleSwarmOptimizationSeederSwarms(float,float,float,int,int,int,int,callback_function,std::vector<void*>);
};

#endif
