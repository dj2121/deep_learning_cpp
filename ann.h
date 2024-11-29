#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <fstream>
#include "matrix.h"

/* 
    Author: Divyarajsinh Jhala
    IIT Kgp
*/



/* Standard activation functions used in machine learning */
double sigmoid(double input){
    return 1 / (1 + exp(-input));
}

double tanh(double input){
    return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
}

double reLU(double input){
    if(input >= 0){
        return input;
    }
    else{
        return 0;
    }
}

double leakyReLU(double input){
    if(input >= 0){
        return input;
    }
    else{
        return (0.01 * input);
    }
}

double step(double input){
    if(input >= 0){
        return 1;
    }
    else{
        return 0;
    }
}

double linear(double input){
    return input;
}



/* Derivatives of standard activation functions */
double derivSigmoid(double input){
    return exp(-input)/ (pow(1 + exp(-input), 2));
}

double derivTanh(double input){
    return 1 - (tanh(input) * tanh(input));
}

double derivReLU(double input){
    if(input >= 0){
        return 1;
    }
    else{
        return 0;
    }
}

double derivLeakyReLU(double input){
    if(input >= 0){
        return 1;
    }
    else{
        return 0.01;
    }
}


double derivLinear(double input){    
    return 1;
}


double random(double input){
    return (double)(rand() % 1000 + 50) / 10000.0;
}


std::vector<std::vector<double> > getTrainingData(std::string filename){

    std::vector<std::vector<double> > output;
    
    std::ifstream myfile (filename);
    std::string line;
    int i = 0;
    if (myfile.is_open()){
        while (getline (myfile, line)){
            if(i % 2 == 0){
                std::vector<double> tempData;
                std::stringstream s(line);
                double data;
                while (s >> data){
                    tempData.push_back(data);
                }
                output.push_back(tempData);
            }
            i++;
        }
        myfile.close();
        return output;
    }

    else{
        std::cout << "Unable to load training data from file"; 
        exit(1);
    }
}

std::vector<std::vector<double> > getTestData(std::string filename){

    std::vector<std::vector<double> > output;
    
    std::ifstream myfile (filename);
    std::string line;
    int i = 0;
    if (myfile.is_open()){
        while (getline (myfile, line)){
            if(i % 2 == 1){
                std::vector<double> tempData;
                std::stringstream s(line);
                double data;
                while (s >> data){
                    tempData.push_back(data);
                }
                output.push_back(tempData);
            }
            i++;
        }
        myfile.close();
        return output;
    }

    else{
        std::cout << "Unable to load training data from file"; 
        exit(1);
    }
}

void normalizeData(std::vector<std::vector<double> > &vec1, std::vector<std::vector<double> > &vec2){
    double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();

    //Finding Min and Max
    for(int i = 0; i < vec1.size(); i++){
        for(int j = 0; j < vec1[0].size(); j++){
            if(vec1[i][j] < min){
                min = vec1[i][j];
            }
            if(vec1[i][j] > max){
                max = vec1[i][j];
            }
        }
    }
    for(int i = 0; i < vec2.size(); i++){
        for(int j = 0; j < vec2[0].size(); j++){
            if(vec2[i][j] < min){
                min = vec2[i][j];
            }
            if(vec2[i][j] > max){
                max = vec2[i][j];
            }
        }
    }


    for(int i = 0; i < vec1.size(); i++){
        for(int j = 0; j < vec1[0].size(); j++){
            vec1[i][j] = (vec1[i][j] - min) / (max - min);
        }
    }

    for(int i = 0; i < vec2.size(); i++){
        for(int j = 0; j < vec2[0].size(); j++){
            vec2[i][j] = (vec2[i][j] - min) / (max - min);
        }
    }

}


class Layer {

    public:
    int id;
    int size;
    int type;                       //type = 0 for input layer, type = 1 for hidden layer, type = 2 for output layer
    int activationType = 0;         //0 for Tanh, 1 for sigmoid, 2 for ReLU, 3 for Leaky ReLU, 4 for Linear            
    Matrix X, W, Wdif, B, Bdif;     //Layer output values X and weights connecting to previous layer, W and Layer bias B


    Layer(int layerId, int layerType, int activationType, int inputSize, int outputSize){
        this->id = layerId;
        setActivation(activationType);
        setSize(inputSize, outputSize);
        setType(layerType);
        initMat();
    }

    void setActivation(int type){
        if(type >= 0 && type <= 4){
            this->activationType = type;
        }
        else{
            std::cout << "Error: Invalid Activation type for layer " << id + 1 << "." << std::endl;
            exit(1);
        }
    }

    void setType(int type){
        if(type == 0){
            this->type = 0;
        }
        else if(type == 1){
            this->type = 1;
        }
        else if(type == 2){
            this->type = 2;
        }
        else{
            std::cout << "Error: Invalid Layer type for layer " << id + 1 << "." << std::endl;
            exit(1);
        }
    }

    void setSize(int inputSize, int outputSize){
        this->X = Matrix(1, outputSize);
        this->W = Matrix(inputSize, outputSize);
        this->B = Matrix(1, outputSize);
        this->size = outputSize;
    }

    void initMat(){
        this->W = this->W.applyFunction(random);
        this->B = this->B.applyFunction(random);
    }
};


class Ann {

    public:
    Matrix forwardPass(std::vector<double> &inputVec)
    {
        if(inputVec.size() != this->inputSize){
            std::cout << "Input Size in Dataset(" << inputVec.size() << ") does not match Input-Layer Size("<< this->inputSize << ")." << std::endl;
            exit(1);
        }

        std::vector< std::vector<double> > inputData;
        inputData.push_back(inputVec);

        nnet[0].X = Matrix(inputData);

        if(this->inputSize > 0 && this->outputSize > 0){
            for(int i = 0; i < this->numLayers; i++){
                if(nnet[i+1].activationType == 0){
                    nnet[i+1].X = ((nnet[i].X * nnet[i+1].W) + nnet[i+1].B).applyFunction(tanh);
                }
                else if(nnet[i+1].activationType == 1){
                    nnet[i+1].X = ((nnet[i].X * nnet[i+1].W) + nnet[i+1].B).applyFunction(sigmoid);
                }
                else if(nnet[i+1].activationType == 2){
                    nnet[i+1].X = ((nnet[i].X * nnet[i+1].W) + nnet[i+1].B).applyFunction(reLU);
                }
                else if(nnet[i+1].activationType == 3){
                    nnet[i+1].X = ((nnet[i].X * nnet[i+1].W) + nnet[i+1].B).applyFunction(leakyReLU);
                }
                else if(nnet[i+1].activationType == 4){
                    nnet[i+1].X = ((nnet[i].X * nnet[i+1].W) + nnet[i+1].B).applyFunction(linear);
                }
            }

            return nnet[numLayers].X;

        }

        else{
            std::cout << "Input and Output layers not defined correctly" << std::endl;
            exit(1);
        }
    }

    void addLayer(int outputSize, int layerType, int activationType){
        if(layerType == 0){
            if(outputSize <= 0){
                std::cout << "Input size must be at least 1" << std::endl;
                exit(1);
            }
            this->inputSize = outputSize;
            nnet.emplace_back(0, 0, activationType, 1, outputSize);
        }
        else{
            if(this->inputSize == 0){
                std::cout << "Please add input layer first" << std::endl;
                exit(1);
            }
            int inputSize = nnet[numLayers].size;
            nnet.emplace_back(this->numLayers + 1, layerType, activationType, inputSize, outputSize);
            this->numLayers++;

            this->outputSize = outputSize;
        }
    }

    void trainingPass(std::vector<double> &expectedVec){

        if(expectedVec.size() != this->outputSize){
            std::cout << "Output size mismatch" << std::endl;
            exit(1);
        }

        std::vector< std::vector<double> > outputData;
        outputData.push_back(expectedVec);

        Matrix Y = Matrix(outputData);

        //Compute Gradients
        for(int i = this->numLayers; i > 0; i--){

            if(i == numLayers){ //We are processing output layer
                if(nnet[i].activationType == 0){
                    nnet[i].Bdif = (nnet[i].X - Y) & (((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivTanh)); 
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 1){
                    nnet[i].Bdif = (nnet[i].X - Y) & (((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivSigmoid)); 
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 2){
                    nnet[i].Bdif = (nnet[i].X - Y) & (((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivReLU)); 
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 3){
                    nnet[i].Bdif = (nnet[i].X - Y) & (((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivLeakyReLU)); 
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 4){
                    nnet[i].Bdif = (nnet[i].X - Y) & (((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivLinear)); 
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
            }

            else{
                if(nnet[i].activationType == 0){
                    Matrix temp = ((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivTanh);
                    nnet[i].Bdif =  nnet[i+1].Bdif * nnet[i+1].W.transpose() & temp;
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 1){
                    Matrix temp = ((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivSigmoid);
                    nnet[i].Bdif =  nnet[i+1].Bdif * nnet[i+1].W.transpose() & temp;
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 2){
                    Matrix temp = ((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivReLU);
                    nnet[i].Bdif =  nnet[i+1].Bdif * nnet[i+1].W.transpose() & temp;
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 3){
                    Matrix temp = ((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivLeakyReLU);
                    nnet[i].Bdif =  nnet[i+1].Bdif * nnet[i+1].W.transpose() & temp;
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
                else if(nnet[i].activationType == 4){
                    Matrix temp = ((nnet[i-1].X * nnet[i].W) + nnet[i].B).applyFunction(derivLinear);
                    nnet[i].Bdif =  nnet[i+1].Bdif * nnet[i+1].W.transpose() & temp;
                    nnet[i].Wdif = nnet[i-1].X.transpose() * nnet[i].Bdif;
                }
            }

        }

        //Update weights
        for(int i = this->numLayers; i > 0; i--){
            nnet[i].W = nnet[i].W - (nnet[i].Wdif * this->learningRate);
            nnet[i].B = nnet[i].B - (nnet[i].Bdif * this->learningRate);
        }

    }

    double calculateError(std::vector<std::vector<double> > &input, std::vector<std::vector<double> > &expectedOutput, double errorTolerence, int log){

        double error = 0;

        for(unsigned i = 0; i < input.size(); i++){
            Matrix output = forwardPass(input[i]);

            double eSum = 0;
            
            for(unsigned j = 0; j < expectedOutput[i].size(); j++){
                if(log){
                    std::cout << "MO: " << output.data[0][j] << ", EO: " << expectedOutput[i][j] << ".\t";
                }
                eSum += (output.data[0][j] - expectedOutput[i][j])*(output.data[0][j] - expectedOutput[i][j]);
            }

            eSum = eSum / expectedOutput[0].size();
            error = error + eSum;
        }

        error = error / input.size();
        return error;

    }

    void modelTrain(int epoch, double learningRate, double decay, double errorTolerence, std::vector<std::vector<double> > input, std::vector<std::vector<double> > expectedOutput, int log){

        this->learningRate = learningRate;

        std::cout << "Starting to train network..." << std::endl;
        std::cout << "MO: Model Output, EO: Expected output" << std::endl << std::endl << std::endl;

        for(int i = 0; i < epoch; i++){
            this->learningRate = this->learningRate - decay;
            double error = calculateError(input, expectedOutput, errorTolerence, log);
            std::cout << "Epoch " << i+1 << ": Error: " << error << std::endl << std::endl;
            for(int j = 0; j < input.size(); j++){
                Matrix output = forwardPass(input[j]);
                trainingPass(expectedOutput[j]);
            }
        }

    }

    void modelTest(std::vector<std::vector<double> > input, std::vector<std::vector<double> > expectedOutput){
        double error = calculateError(input, expectedOutput, 0, 1);
        std::cout << "Test Error: " << error << std::endl << std::endl;
    }

    Ann(){
        inputSize = 0;
        outputSize = 0;
        numLayers = 0;
        std::srand(std::time(nullptr));
    }

    private:
    int inputSize = 0;       //inputSize > 0 whenever an input layer is defined for the ANN
    int outputSize = 0;      //outputSize > 0 whenever an output layer is defined for the ANN
    int numLayers = 0;          //Stores the number of layers
    double learningRate = 0.01;

    std::vector<Layer> nnet;

};

/*
For compilation purposes.
int main(){
    return 0;
}
*/