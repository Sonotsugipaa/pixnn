#include "nn/nn.hpp"



namespace nn {

	Perceptron::Perceptron(size_t inputs):
			input_size (inputs),
			weights (new double[inputs+1])
	{
		for(size_t i=0; i <= input_size; ++i) {
			weights[i] = nn::random();
		}
	}

	Perceptron::Perceptron(const Perceptron& cpy):
			input_size (cpy.input_size),
			weights (new double[input_size+1])
	{
		for(size_t i=0; i <= input_size; ++i)
			weights[i] = cpy.weights[i];
	}

	Perceptron::Perceptron(Perceptron&& mov):
			input_size (std::move(mov.input_size)),
			weights (std::move(mov.weights))
	{
		mov.weights = nullptr;
	}

	Perceptron::~Perceptron() {
		if(weights != nullptr) {
			delete[] weights;  weights = nullptr;
		}
	}

	Perceptron& Perceptron::operator = (const Perceptron& cpy) {
		this->~Perceptron();
		new (this) Perceptron(cpy);
		return *this;
	}

	Perceptron& Perceptron::operator = (Perceptron&& mov) {
		this->~Perceptron();
		new (this) Perceptron(std::move(mov));
		return *this;
	}


	double Perceptron::guess(activation_func act, double* in) const {
		double sum = 0.0;
		for(size_t i=0; i < input_size; ++i)
			sum += weights[i] * in[i];
		sum += weights[input_size];
		return act(sum);
	}

	void Perceptron::learn(
			activation_func act, double* inputs,
			double error, double rate
	) {
		error = rate * error;
		for(size_t i=0; i < input_size; ++i) {
			weights[i] -= error * inputs[i];
		}
		weights[input_size] -= error;
	}

	void Perceptron::train(activation_func act, DataSet& data, double rate) {
		for(DataRow& row : data) {
			double* inputs = row.inputs.data();
			double guessed = guess(act, inputs);
			learn(act, inputs, nn::error(row.outputs[0], guessed), rate);
		}
	}

	void Perceptron::randomize() {
		for(size_t i=0; i <= input_size; ++i)
			weights[i] = nn::random();
	}

}
