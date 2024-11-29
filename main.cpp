#include "ann.h"

using namespace std;

int main(){

    vector< vector<double> > trainingData;
    vector< vector<double> > expectedOutput;
    vector<double> temp;

    string train = "train.txt";

    trainingData = getTrainingData(train);
    expectedOutput = getTestData(train);   

    Ann XORnet;
    XORnet.addLayer(32, 0, 2);
    XORnet.addLayer(100, 1, 0);
    XORnet.addLayer(100, 1, 0);
    XORnet.addLayer(100, 1, 0);
    XORnet.addLayer(1, 2, 2);

    XORnet.modelTrain(100, 0.01, 0.001, 5, trainingData, expectedOutput, 1);

    return 0;
}