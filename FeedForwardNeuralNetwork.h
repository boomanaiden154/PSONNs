/*
 * FeedForwardNeuralNetwork.h
 *
 *  Created on: Feb 4, 2018
 *      Author: aiden
 */
#include <vector>
#include "Dataset.h"

#ifndef FEEDFORWARDNEURALNETWORK_H_
#define FEEDFORWARDNEURALNETWORK_H_

class FeedForwardNeuralNetwork
{

public:
	FeedForwardNeuralNetwork(int,int,int);
	FeedForwardNeuralNetwork(int,int,int,std::vector<float>);
	std::vector<float> Activate(std::vector<float>);
	void ActivateNoReturn(std::vector<float>,std::vector<float>*,std::vector<float>*);
	float GetError(std::vector<float>, std::vector<float>);
	float GetFitness(Dataset*);
	static float GetFitnessPSO(const std::vector<float>&,const std::vector<void*>&);
	std::vector<float> GetArrayRepresentation();
	void InstallArray(std::vector<float>);
	void TrainPSO(Dataset,int,int);
private:
	std::vector<float> InputToHiddenWeights;
	std::vector<float> HiddenToOutputWeights;
	std::vector<float> HiddenBias;
	std::vector<float> OutputBias;
	int inputCount;
	int hiddenCount;
	int outputCount;
	inline float RELU(float);
};

#endif /* FEEDFORWARDNEURALNETWORK_H_ */
