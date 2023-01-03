#include <iostream>
#include "PSO.h"
#include "FeedForwardNeuralNetwork.h"
#include "Dataset.h"
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>

int main()
{
    srand(time(NULL));
    Dataset data(1,1,1);
    std::vector<bool> inputs = {false,true,true,true,true,false};
    std::vector<bool> outputs = {false,false,false,false,false,true};
    data.LoadFromCSV("Iris.csv",inputs,outputs,150);
    FeedForwardNeuralNetwork ffnn(4,5,1);
    ffnn.TrainPSO(data,100,20);
    std::cout << ffnn.GetFitness(&data) << "\n";
    /*float stepSize = 0.25;
    float stepsPerUnit = 1 / stepSize + 1;
    Dataset data(9,1,pow(stepsPerUnit,9.0f) + stepsPerUnit);
    Dataset XOR(2,1,4);
    XOR.inputs = {0,0,0,1,1,0,1,1};
    XOR.outputs = {0,1,1,0};
    std::vector<float> WeightBiasArray(9,0.0f);
    FeedForwardNeuralNetwork ffnn(2,2,1);
    for(int i = 0; i < data.samples; i++)
    {
    	ffnn.InstallArray(WeightBiasArray);
    	data.outputs[i] = ffnn.GetFitness(&XOR);
    	for(int x = 0; x < data.inputCount; x++)
    	{
    		data.inputs[i * data.inputCount + x] = WeightBiasArray[x];
    	}
    	WeightBiasArray[0] += stepSize;
    	for(int x = 0; x < WeightBiasArray.size(); x++)
    	{
    		if(WeightBiasArray[x] > 1.0f)
    		{
    			WeightBiasArray[x] = 0.0f;
    			if(i != WeightBiasArray.size() - 1)
    			{
    				WeightBiasArray[x + 1] += stepSize;
    			}
    		}
    	}
    }
    data.WriteToCSV("test.csv");*/
    return 0;
}
