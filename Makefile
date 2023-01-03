flags := -g 

all: Dataset.o FeedForwardNeuralNetwork.o PSO.o main.o
	g++ Dataset.o FeedForwardNeuralNetwork.o PSO.o main.o -o main -O3 -fopenmp -std=c++11 $(flags)
Dataset.o : Dataset.cpp Dataset.h
	g++ -c -O3 Dataset.cpp -std=c++11 $(flags)
FeedForwardNeuralNetwork.o : FeedForwardNeuralNetwork.cpp FeedForwardNeuralNetwork.h
	g++ -c FeedForwardNeuralNetwork.cpp -fopenmp -O3 -std=c++11 $(flags)
PSO.o : PSO.cpp PSO.h
	g++ -c PSO.cpp -fopenmp -O3 -std=c++11 $(flags)
main.o : main.cpp
	g++ -c main.cpp -O3 $(flags)

clean : 
	rm ./Dataset.o
	rm ./FeedForwardNeuralNetwork.o
	rm ./PSO.o
	rm ./main.o
