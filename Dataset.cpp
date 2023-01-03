/*
 * Dataset.cpp
 *
 *  Created on: Feb 4, 2018
 *      Author: aiden
 */
#include "Dataset.h"
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

Dataset::Dataset(int inputs_, int outputs_, int samples_)
{
	inputCount = inputs_;
	outputCount = outputs_;
	samples = samples_;
	inputs.resize(inputs_*samples_,0);
	outputs.resize(outputs_*samples_,0);
}

void Dataset::LoadFromCSV(std::string fileName,std::vector<bool> whatInputs, std::vector<bool> whatOutputs, int examples)
{
	SetInputs(whatInputs,whatOutputs);
	inputs.resize(inputCount * examples,0);
	outputs.resize(outputCount * examples,0);
	samples = examples;
	std::ifstream CSVfile(fileName);
	std::string line;
	int column = 0;
	int outputIndex = 0;
	int inputIndex = 0;
	while(std::getline(CSVfile,line))
	{
		std::stringstream lineStream(line);
		std::string cell;
		while(std::getline(lineStream,cell,','))
		{
			double value = std::stod(cell);
			if(whatInputs[column] == true)
			{
				inputs[inputIndex] = value;
				inputIndex++;
			}
			if(whatOutputs[column] == true)
			{
				outputs[outputIndex] = value;
				outputIndex++;
			}
			column++;
		}
		column = 0;
	}
}

void Dataset::SetInputs(std::vector<bool> whatInputs, std::vector<bool> whatOutputs)
{
	inputCount = 0;
	outputCount = 0;
	for(int i = 0; i < whatInputs.size(); i++)
	{
		if(whatInputs[i] == true)
		{
			inputCount++;
		}
	}
	for(int i = 0; i < whatOutputs.size(); i++)
	{
		if(whatOutputs[i] == true)
		{
			outputCount++;
		}
	}
}

void Dataset::WriteToCSV(std::string fileName)
{
	std::ofstream outputFile(fileName);
	for(int i = 0; i < samples; i++)
	{
		std::string toWrite("");
		for(int x = 0; x < inputCount; x++)
		{
			toWrite += std::to_string(inputs[i * inputCount + x]);
			toWrite += ",";
		}
		for(int x = 0; x < outputCount; x++)
		{
			toWrite += std::to_string(outputs[i * outputCount + x]);
			toWrite += ",";
		}
		toWrite += "\n";
		outputFile << toWrite;
	}
}
