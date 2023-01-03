/*
 * Dataset.h
 *
 *  Created on: Feb 4, 2018
 *      Author: aiden
 */
#include<vector>
#include<string>

#ifndef DATASET_H_
#define DATASET_H_

class Dataset
{
public:
	Dataset(int,int,int);
	std::vector<float> inputs;
	std::vector<float> outputs;
	int inputCount;
	int outputCount;
	int samples;
	void LoadFromCSV(std::string,std::vector<bool>,std::vector<bool>,int);
	void WriteToCSV(std::string);
private:
	void SetInputs(std::vector<bool>,std::vector<bool>);
};

#endif /* DATASET_H_ */
