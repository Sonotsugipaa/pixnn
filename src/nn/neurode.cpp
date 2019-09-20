#include "nn/nn.hpp"



namespace nn {

	Neurode::Neurode(size_t inputs, size_t outputs):
			input_size (inputs),
			output_size (outputs)
	{
		perceptrons.reserve(outputs);
		for(size_t i=0; i < output_size; ++i) {
			perceptrons.push_back(Perceptron(inputs));
		}
	}

	Neurode::Neurode(const Neurode& cpy):
			input_size (cpy.input_size),
			output_size (cpy.output_size),
			perceptrons (cpy.perceptrons)
	{
		for(size_t i=0; i < output_size; ++i)
			perceptrons[i] = cpy.perceptrons[i];
	}

	Neurode::Neurode(Neurode&& mov):
			input_size (std::move(mov.input_size)),
			output_size (std::move(mov.output_size)),
			perceptrons (std::move(mov.perceptrons))
	{ }

	Neurode::~Neurode() { }

	Neurode& Neurode::operator = (const Neurode& cpy) {
		this->~Neurode();
		new (this) Neurode(cpy);
		return *this;
	}

	Neurode& Neurode::operator = (Neurode&& mov) {
		this->~Neurode();
		new (this) Neurode(std::move(mov));
		return *this;
	}


	void Neurode::guess(activation_func act, double* in, double* out) const {
		for(size_t i=0; i < output_size; ++i) {
			out[i] = perceptrons[i].guess(act, in);
			//out[i] = act(perceptrons[i].guess(act, in));
		}
	}

	void Neurode::learn(
			activation_func act,
			double* in,
			double* errors, double rate
	) {
		for(size_t i=0; i < output_size; ++i) {
			perceptrons[i].learn(act, in, errors[i], rate);
		}
	}

	void Neurode::train(activation_func act, DataSet& data, double rate) {
		for(DataRow& row : data) {
			for(size_t i=0; i < output_size; ++i) {
				double error = nn::error(
						row.outputs[i],
						perceptrons[i].guess(act, row.inputs.data())
				);
				perceptrons[i].learn(act, row.inputs.data(), error, rate);
			}
		}
	}

	void Neurode::randomize() {
		for(Perceptron& p : perceptrons)
			p.randomize();
		// bias = nn::random();
	}

}
