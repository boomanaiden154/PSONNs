#include "PSO.h"
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "omp.h"

typedef float (*callback_function)(const std::vector<float>&,const std::vector<void*>&);

std::vector<float> PSO::ParticleSwarmOptimiationInjectPositions(float c1,float c2,float w,int iterations,int ParticleNumber,int dimensions, callback_function GetFitness, std::vector<void*> fitnessFunctionParameters,std::vector<float> injectPositions)
{
    std::vector<float> ParticlePositions(ParticleNumber * dimensions, 0);
	if(injectPositions.empty())
	{
		ParticlePositions = injectPositions;
	}
    std::vector<float> ParticleVelocities(ParticleNumber * dimensions, 0);
    std::vector<float> PersonalBests(ParticleNumber * dimensions, 0);
    std::vector<float> GlobalBest(dimensions, 0);
    std::vector<float> PersonalBestScores(ParticleNumber,0);
    float GlobalBestScore = -1000000;

    for(int x = 0; x < ParticleNumber; x++)
    {
        for(int y = 0; y < dimensions; y++)
        {
            ParticlePositions[x*dimensions+y] = (float)rand()/(float)(RAND_MAX);
            ParticleVelocities[x*dimensions+y] = (float)rand()/(float)(RAND_MAX);
            PersonalBests[x*dimensions+y] = ParticlePositions[x*dimensions+y];
        }
        PersonalBestScores[x] = -1000000;
    }

    for(int i = 0; i < iterations; i++)
    {
#pragma omp parallel for schedule(guided)
        for(int x = 0; x < ParticleNumber; x++)
        {
            std::vector<float> ParticlePosition(ParticlePositions.begin() + x * dimensions, ParticlePositions.begin() + (x+1) * dimensions);
            float fitness = GetFitness(ParticlePosition, fitnessFunctionParameters);
            if(fitness > PersonalBestScores[x])
            {
                PersonalBestScores[x] = fitness;
                std::copy(ParticlePosition.begin(), ParticlePosition.end(), ParticlePositions.begin() + x * dimensions);
                if(fitness > GlobalBestScore)
                {
#pragma omp critical
                	{
                		GlobalBestScore = fitness;
                		GlobalBest = ParticlePosition;
                	}
                }
            }
        }
        //calculate and update velocities
        for(int x = 0; x < ParticleNumber; x++)
        {
            //update velocity
            float rand1 = (float)rand()/(float)(RAND_MAX);
            float rand2 = (float)rand()/(float)(RAND_MAX);
            for(int y = 0; y < dimensions; y++)
            {
                int z = x*dimensions+y;
                ParticleVelocities[z] = w * ParticleVelocities[z] + (c1 * rand1 * (PersonalBests[z] - ParticlePositions[z]) + (c2 * rand2 * (GlobalBest[y] - ParticlePositions[z])));
            }
        }
        //update positions
        for(int x = 0; x < ParticleNumber * dimensions; x++)
        {
            ParticlePositions[x] = ParticlePositions[x] + ParticleVelocities[x];
        }
    }

    return ParticlePositions;
}

std::vector<float> PSO::ParticleSwarmOptimization(float c1, float c2, float w, int iterations, int ParticleNumber, int dimensions, callback_function GetFitness, std::vector<void*> fitnessFunctionParameters)
{
	std::vector<float> Positions;
	return ParticleSwarmOptimiationInjectPositions(c1,c2,w,iterations,ParticleNumber,dimensions,GetFitness,fitnessFunctionParameters, Positions);
}

std::vector<float> PSO::ParticleSwarmOptimizationGbest(float c1, float c2, float w, int iterations, int ParticleNumber, int dimensions, callback_function GetFitness, std::vector<void*> fitnessFunctionParameters)
{
	std::vector<float> Particles = ParticleSwarmOptimization(c1,c2,w,iterations,ParticleNumber,dimensions,GetFitness,fitnessFunctionParameters);
	std::vector<float> GlobalBest(dimensions,0);
	float GlobalBestScore = -100000000;
#pragma omp parallel
	for(int i = 0; i < ParticleNumber; i++)
	{
		std::vector<float> ParticlePosition(Particles.begin() + i * dimensions, Particles.begin() + (i + 1) * dimensions);
		float score = GetFitness(ParticlePosition,fitnessFunctionParameters);
		if(score > GlobalBestScore)
		{
#pragma omp critical
			{
				GlobalBestScore = score;
				GlobalBest = ParticlePosition;
			}
		}
	}
	return GlobalBest;
}

std::vector<float> PSO::ParticleSwarmOptimizationSeederSwarms(float c1, float c2, float w, int iterations, int ParticleNumber, int dimensions, int SeederSwarms, callback_function GetFitness,std::vector<void*> fitnessFunctionParameters)
{
	std::vector<float> AllParticlePositions(dimensions*SeederSwarms,0);
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int numThreads = omp_get_num_threads();
		for(int i = id; i < SeederSwarms; i += numThreads)
		{
			std::vector<float> GbestPosition = ParticleSwarmOptimizationGbest(c1,c2,w,iterations,ParticleNumber,dimensions,GetFitness,fitnessFunctionParameters);
			std::copy(GbestPosition.begin(), GbestPosition.end(), AllParticlePositions.begin() + (i*dimensions));
		}
	}
	return ParticleSwarmOptimiationInjectPositions(c1,c2,w,iterations,ParticleNumber,dimensions,GetFitness,fitnessFunctionParameters,AllParticlePositions);
}
