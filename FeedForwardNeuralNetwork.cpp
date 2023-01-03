/*
 * FeedForwardNeuralNetwork.cpp
 *
 *  Created on: Feb 4, 2018
 *      Author: aiden
 */

#include "FeedForwardNeuralNetwork.h"
#include "PSO.h"
#include <algorithm>
#include <vector>
#include <cmath>

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(int Input, int Hidden, int Outputs)
{
	InputToHiddenWeights.resize(Input * Hidden,0.0f);
	HiddenToOutputWeights.resize(Hidden * Outputs, 0.0f);
	HiddenBias.resize(Hidden,0.0f);
	OutputBias.resize(Outputs,0.0f);
	inputCount = Input;
	hiddenCount= Hidden;
	outputCount = Outputs;
}

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(int Input, int Hidden, int Outputs, std::vector<float> weightsBias)
{
	InputToHiddenWeights.resize(Input * Hidden,0.0f);
	HiddenToOutputWeights.resize(Hidden * Outputs, 0.0f);
	HiddenBias.resize(Hidden,0.0f);
	OutputBias.resize(Outputs,0.0f);
	inputCount = Input;
	hiddenCount= Hidden;
	outputCount = Outputs;
	InstallArray(weightsBias);
}

float FeedForwardNeuralNetwork::RELU(float input)
{
	return std::max(0.0f,input);
}

/**
 * A function that calculates the outputs of a neural network based on a given set of inputs. Internally calls ActivateNoReturn.
 * @param inputs The inputs upon which the output of the neural network will be calculated
 * @return the output of the neural network
 * @see ActivateNoReturn()
 */
std::vector<float> FeedForwardNeuralNetwork::Activate(std::vector<float> inputs)
{
	std::vector<float> toreturn(outputCount,0.0f);
	std::vector<float> HiddenTemp(hiddenCount,0.0f);
	for(int x = 0; x < hiddenCount; x++)
	{
		for(int y = 0; y < inputCount; y++)
		{
			HiddenTemp[x] += inputs[y] * InputToHiddenWeights[x*inputCount+y];
		}
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		HiddenTemp[i] += HiddenBias[i];
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		HiddenTemp[i] = RELU(toreturn[i]);
	}
	for(int x = 0; x < outputCount; x++)
	{
		for(int y = 0;y < hiddenCount; y++)
		{
			toreturn[x] += HiddenTemp[y] * HiddenToOutputWeights[x*hiddenCount+y];
		}
	}
	for(int i = 0; i < outputCount; i++)
	{
		toreturn[i] += OutputBias[i];
	}
	return toreturn;
}

/**
 * An alternative version of the Activate function that mainly cuts down on memory allocations and deallocations
 * @param inputs The inputs on which the neural network should be activated.
 * @param hiddenTemp A pointer to a vectr which the function will use to calculate the result
 * @param outputs A pointer to a vector which will contain the outputs of the neural network.
 * @see Activate()
 */
void FeedForwardNeuralNetwork::ActivateNoReturn(std::vector<float> inputs, std::vector<float>* hiddenTemp, std::vector<float>* outputs)
{
	for(int x = 0; x < hiddenCount; x++)
	{
		for(int y = 0; y < inputCount; y++)
		{
			(*hiddenTemp)[x] += inputs[y] * InputToHiddenWeights[x*inputCount+y];
		}
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		(*hiddenTemp)[i] += HiddenBias[i];
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		(*hiddenTemp)[i] = RELU((*hiddenTemp)[i]);
	}
	for(int x = 0; x < outputCount; x++)
	{
		for(int y = 0;y < hiddenCount; y++)
		{
			(*outputs)[x] += (*hiddenTemp)[y] * HiddenToOutputWeights[x*hiddenCount+y];
		}
	}
	for(int i = 0; i < outputCount; i++)
	{
		(*outputs)[i] += OutputBias[i];
	}
}

/**
 * Gets the error rate of a neural network for a given set of inputs and expected outputs. Uses Mean Square Error as a loss function
 * @param inputs The inputs on which the neural network should be activated
 * @param expectedOutpus The expected outputs that the neural network should provide in ideal circumstances
 * @return a float representing the MSE based on the training row provided
 */
float FeedForwardNeuralNetwork::GetError(std::vector<float> inputs, std::vector<float> expectedOutputs)
{
	float toreturn = 0.0f;
	std::vector<float> ActualOutputs = Activate(inputs);
	float sum = 0.0f;
	for(int i = 0; i < outputCount; i++)
	{
		sum += std::pow((expectedOutputs[i] - ActualOutputs[i]),2.0f);
	}
	toreturn = sum / outputCount;
	return toreturn;
}

/**
 * Gets the fitness of a neural network. It basically finds the average of the neural networks error rate over all rows in the dataset
 * @param dataset A pointer to a dataset which contains the inputs and outputs
 * @return a float representing the fitness of the neural network
 */
float FeedForwardNeuralNetwork::GetFitness(Dataset *dataset)
{
	float fitness = 0.0f;
	for(int i = 0; i < dataset->samples; i++)
	{
		std::vector<float> inputs(dataset->inputs.begin() + i * dataset->inputCount, dataset->inputs.begin() + (i + 1) * dataset->inputCount);
		std::vector<float> outputs(dataset->outputs.begin() + i * dataset->outputCount, dataset->outputs.begin() + (i + 1) * dataset->outputCount);
		fitness += GetError(inputs, outputs);
	}
	fitness = fitness / (float)dataset->samples;
	return fitness;
}

/**
 * A function that calls GetFitness internally. It is used while training the network using Particle Swarm Optimization.
 * @param NeuralNetwork A vector of floats representing the neural network
 * @param parameters A vector of void pointers which point to the dataset needed to evaluate the fitness, and an integer which contains the number of hidden nodes.
 * @see GetFitness()
 */
float FeedForwardNeuralNetwork::GetFitnessPSO(const std::vector<float> &NeuralNetwork,const  std::vector<void*> &parameters)
{
	Dataset* thedataset = static_cast<Dataset*>(parameters[0]);
	int hiddenNodes = *static_cast<int*>(parameters[1]);
	FeedForwardNeuralNetwork network(thedataset->inputCount,hiddenNodes,thedataset->outputCount, NeuralNetwork);
	return -network.GetFitness(thedataset);
}

/**
 * A function that returns the neural network in a one dimensional vector format
 * @return a vector representing the neural network
 */
std::vector<float> FeedForwardNeuralNetwork::GetArrayRepresentation()
{
	std::vector<float> toreturn((inputCount*hiddenCount)+(hiddenCount*outputCount)+hiddenCount+outputCount);
	int index = 0;
	for(int x = 0; x < hiddenCount; x++)
	{
		for(int y = 0; y < inputCount; y++)
		{
			toreturn[index] = InputToHiddenWeights[x*inputCount+y];
			index++;
		}
	}
	for(int x = 0; x < outputCount; x++)
	{
		for(int y = 0; y < hiddenCount; y++)
		{
			toreturn[index] = HiddenToOutputWeights[x*hiddenCount+y];
			index++;
		}
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		toreturn[index] = HiddenBias[i];
		index++;
	}
	for(int i = 0; i < outputCount; i++)
	{
		toreturn[index] = OutputBias[i];
		index++;
	}
	return toreturn;
}

/**
 * A function that takes an array and installs it into the neural network. It only changes weights and biases.
 * @param array The array to install into the neural network
 */
void FeedForwardNeuralNetwork::InstallArray(std::vector<float> array)
{
	int index = 0;
	for(int x = 0; x < hiddenCount; x++)
	{
		for(int y = 0; y < inputCount; y++)
		{
			InputToHiddenWeights[x*inputCount+y] = array[index];
			index++;
		}
	}
	for(int x = 0; x < outputCount; x++)
	{
		for(int y = 0; y < hiddenCount; y++)
		{
			HiddenToOutputWeights[x*hiddenCount+y] = array[index];
			index++;
		}
	}
	for(int i = 0; i < hiddenCount; i++)
	{
		HiddenBias[i] = array[index];
		index++;
	}
	for(int i = 0; i < outputCount; i++)
	{
		OutputBias[i] = array[index];
		index++;
	}
}

/**
 * A helper function that trains a neural network using Particle Swarm Optimization
 * @param dataset The dataset which the Neural Network should be trained on
 * @param iterations The number of iterations the neural network should be trained
 * @param particles The number of particles that should be used in the PSO algorithm
 */
void FeedForwardNeuralNetwork::TrainPSO(Dataset dataset, int iterations, int particles)
{
	int* hiddenNodePoiner = &hiddenCount;
	std::vector<void*> parameters(2,NULL);
	parameters[0] = &dataset;
	parameters[1] = hiddenNodePoiner;
	int size = (inputCount*hiddenCount)+(hiddenCount*outputCount)+hiddenCount+outputCount;
    float (*foo)(const std::vector<float>&,const std::vector<void*>&);
    foo = &FeedForwardNeuralNetwork::GetFitnessPSO;
	std::vector<float> BestNetworkConfiguration = PSO::ParticleSwarmOptimizationGbest(2.1f,2.1f,0.777f,iterations,particles,size,foo,parameters);
	InstallArray(BestNetworkConfiguration);
}



