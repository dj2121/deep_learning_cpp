#ifndef DEF_MATRIX
#define DEF_MATRIX
 
#include <vector>
#include <cassert>
#include <iostream>
 
class Matrix
{
public:
    Matrix();
    Matrix(std::vector<std::vector<double> > const &input);
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double key);
    Matrix(const Matrix &);
    ~Matrix();

    //Matrix operations
    Matrix operator+(Matrix const &);
    Matrix operator-(Matrix const &);
    Matrix operator*(Matrix const &);
    Matrix operator&(Matrix const &);
    Matrix transpose();


    //Scalar matrix operations
    Matrix operator+(double const);
    Matrix operator-(double const);
    Matrix operator*(double const);
    Matrix operator/(double const);

    //Apply a function to every element of the matrix
    Matrix applyFunction(double (*function)(double)) const;
 
    //Other Methods
    double& operator()(unsigned &, unsigned &) const;
    void print() const;
    unsigned getRows() const;
    unsigned getCols() const;
 
    std::vector<std::vector<double> > data;
    int rowSize;
    int colSize;
};


//Returns specific element of a Matrix
double& Matrix::operator() (unsigned &rowNo, unsigned &colNo) const{
    return (double &) data[rowNo][colNo];
}

//Returns number of rows
unsigned Matrix::getRows() const{
    return this->rowSize;
}

//Returns number of columns
unsigned Matrix::getCols() const{
    return this->colSize;
}

// Take any given matrices transpose and returns another matrix
Matrix Matrix::transpose() {
    Matrix result(colSize, rowSize, 0.0);
    for (unsigned i = 0; i < colSize; i++)
    {
        for (unsigned j = 0; j < rowSize; j++) {
            result(i,j) = this->data[j][i];
        }
    }
    return result;
}

//Default constructor
Matrix::Matrix(){}

// Constructor for Matrix initialized with 0s
Matrix::Matrix(int rowS, int colS){
    rowSize = rowS;
    colSize = colS;
    data.resize(rowSize);
    for (unsigned i = 0; i < data.size(); i++)
    {
        data[i].resize(colSize, 0);
    }
}

//Vector based constructor
Matrix::Matrix(std::vector<std::vector<double> > const &input){
    assert(input.size()!=0);
    this->rowSize = input.size();
    this->colSize = input[0].size();
    this->data = input;
}

// Constructor for Matrix based on key value (initializer)
Matrix::Matrix(int rowS, int colS, double key){
    rowSize = rowS;
    colSize = colS;
    data.resize(rowSize);
    for (unsigned i = 0; i < data.size(); i++)
    {
        data[i].resize(colSize, key);
    }
}

// Copy Constructor
Matrix::Matrix(const Matrix &B) {
    this->colSize = B.getCols();
    this->rowSize = B.getRows();
    this->data = B.data;
    
}

//Destructor
Matrix::~Matrix(){}


//Apply a specific function to each of the elements. Function must return double, and take a double as an argument
Matrix Matrix::applyFunction(double (*function)(double)) const{
    Matrix result(rowSize, colSize);
    int i,j;
 
    for (i=0 ; i<rowSize ; i++)
    {
        for (j=0 ; j<colSize ; j++){
            result.data[i][j] = (*function)(data[i][j]);
        }
    }
 
    return result;
}

//Addition of Matrices
Matrix Matrix::operator+(Matrix const &B){
    Matrix sum(rowSize, colSize, 0.0);
    unsigned i,j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            sum(i,j) = this->data[i][j] + B(i,j);
        }
    }
    return sum;
}

// Subtraction of Matrices
Matrix Matrix::operator-(Matrix const &B){
    Matrix result(rowSize, colSize, 0.0);
    unsigned i,j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            result(i,j) = this->data[i][j] - B(i,j);
        }
    }    
    return result;
}

// Multiplication of Matrices
Matrix Matrix::operator*(Matrix const &B){
    Matrix result(rowSize, B.getCols(), 0.0);

    if(colSize == B.getRows())
    {
        unsigned i, j, k;
        double temp = 0.0;
        for (i = 0; i < rowSize; i++)
        {
            for (j = 0; j < B.getCols(); j++)
            {
                temp = 0.0;
                for (k = 0; k < colSize; k++)
                {
                    temp += data[i][k] * B(k,j);
                }
                result(i,j) = temp;
            }
        }
        return result;
    }
    else
    {
        std::cout << "Incorrect dimensions of Matrices" << std::endl;
    }
}

// Element wise Multiplication of Matrices
Matrix Matrix::operator&(Matrix const &B){
    Matrix result(rowSize, colSize, 0.0);

    if(colSize == B.getCols() && rowSize == B.getRows())
    {
        unsigned i, j, k;
        double temp = 0.0;
        for (i = 0; i < rowSize; i++)
        {
            for (j = 0; j < colSize; j++)
            {
                result(i,j) = data[i][j] * B(i, j);
            }
        }
        return result;
    }
    else
    {
        std::cout << "Incorrect dimensions of Matrices" << std::endl;
    }
}

// Scalar Addition
Matrix Matrix::operator+(double const scalar){
    Matrix result(rowSize, colSize, 0.0);
    unsigned i,j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            result(i,j) = this->data[i][j] + scalar;
        }
    }
    return result;
}

// Scalar Subraction
Matrix Matrix::operator-(double const scalar){
    Matrix result(rowSize, colSize, 0.0);
    unsigned i,j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            result(i,j) = this->data[i][j] - scalar;
        }
    }
    return result;
}

// Scalar Multiplication
Matrix Matrix::operator*(double const scalar){
    Matrix result(rowSize, colSize, 0.0);
    unsigned i,j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            result(i,j) = this->data[i][j] * scalar;
        }
    }
    return result;
}

// Scalar Division
Matrix Matrix::operator/(double const scalar){
    Matrix result(rowSize,colSize,0.0);
    unsigned i, j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            result(i,j) = this->data[i][j] / scalar;
        }
    }
    return result;
}

// Prints the matrix matlab way
void Matrix::print() const {
    std::cout << "Matrix: " << std::endl;
    for (unsigned i = 0; i < rowSize; i++) {
        for (unsigned j = 0; j < colSize; j++) {
            std::cout << "[" << data[i][j] << "] ";
        }
        std::cout << std::endl;
    }
}
 
#endif