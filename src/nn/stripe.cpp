#include "nn/nn.hpp"



namespace nn {

	Stripe::Stripe(
			size_t inputs,
			std::vector<size_t> layer_sizes,
			size_t outputs
	):
			input_size (inputs),
			output_size (outputs),
			biggest_neurode ((inputs > outputs)? inputs : outputs),
			neurodes_count (layer_sizes.size() + 1),
			neurodes (),
			_forward (new double*[neurodes_count+1]),
			_backward (new double*[neurodes_count+1])
	{
		neurodes.reserve(neurodes_count);
		if(neurodes_count > 1) {
			size_t i;
			neurodes.push_back(Neurode(inputs, layer_sizes[0]));
			if(layer_sizes[0] > biggest_neurode)
				biggest_neurode = layer_sizes[0];
			for(i=1; i < neurodes_count-1; ++i) {
				if(layer_sizes[i] > biggest_neurode)
					biggest_neurode = layer_sizes[i];
				neurodes.push_back(Neurode(layer_sizes[i-1], layer_sizes[i]));
				_forward[i] = new double[neurodes[i].inputSize()];
				_backward[i] = new double[neurodes[i].inputSize()];
			}
			neurodes.push_back(Neurode(layer_sizes[i-1], output_size));
			_forward[i] = new double[neurodes[i].inputSize()];
			_backward[i] = new double[neurodes[i].inputSize()];
		} else {
			neurodes.push_back(Neurode(input_size, output_size));
		};
		_forward[0] = new double[neurodes[0].inputSize()];
		_backward[0] = new double[neurodes[0].inputSize()];
		_forward[neurodes_count] = new double[output_size];
		_backward[neurodes_count] = new double[output_size];
	}

	Stripe::Stripe(const Stripe& cpy):
			input_size (cpy.input_size),
			output_size (cpy.output_size),
			biggest_neurode (cpy.biggest_neurode),
			neurodes_count (cpy.neurodes_count),
			neurodes (cpy.neurodes),
			_forward (new double*[neurodes_count+1]),
			_backward (new double*[neurodes_count+1])
	{
		size_t i;
		for(i=0; i < neurodes_count; ++i) {
			_forward[i]  = new double[neurodes[i].inputSize()];
			_backward[i] = new double[neurodes[i].inputSize()];
		}
		_forward[i]  = new double[output_size];
		_backward[i] = new double[output_size];
	}

	Stripe::Stripe(Stripe&& mov):
			input_size (std::move(mov.input_size)),
			output_size (std::move(mov.output_size)),
			biggest_neurode (std::move(mov.biggest_neurode)),
			neurodes_count (std::move(mov.neurodes_count)),
			neurodes (std::move(mov.neurodes)),
			_forward (std::move(mov._forward)),
			_backward (std::move(mov._backward))
	{
		_forward = nullptr;
	}

	Stripe::~Stripe() {
		if(_forward != nullptr) {
			for(size_t i=0; i <= neurodes_count; ++i) {
				delete[] _forward[i];
				delete[] _backward[i];
			}
			delete[] _forward;
			delete[] _backward;
			_forward  = nullptr;
		}
	}

	Stripe& Stripe::operator = (const Stripe& cpy) {
		this->~Stripe();
		new (this) Stripe(cpy);
		return *this;
	}

	Stripe& Stripe::operator = (Stripe&& mov) {
		this->~Stripe();
		new (this) Stripe(std::move(mov));
		return *this;
	}


	void Stripe::guess(activation_func act, double* in, double* out) const {
		size_t last_n = neurodes_count - 1;

		neurodes[0].guess(act, in, _forward[1]);
		for(size_t i=1; i < last_n; ++i)
			neurodes[i].guess(act, _forward[i], _forward[i+1]);
		neurodes[last_n].guess(act, _forward[last_n], out);
	}

	double Stripe::train(
			activation_func act,
			activation_func_deriv derive,
			double* in, double* expect,
			double rate
	) {
		/* _backward: contains values derived from the inputs (derivative * error),
		 *            and has as many columns as the outputs of the [i-1]th neurode
		 *                                 or as the inputs of the [i]th neurode;
		 * _forward:  contains the inputs,
 		 *            and has as many columns as the inputs of the [i]th neurode
		 *                    or (except [0]) as the outputs of the [i-1]th neurode;
		 * _forward[2] is as big as _backward[1] */
		size_t last_n = neurodes_count - 1;

		double avg_error = 0.0;

		// in -> forward[0]
		for(size_t i=0; i < input_size; ++i)
			_forward[0][i] = in[i];

		// Make all guesses
		for(size_t i=0; i < neurodes_count; ++i) {
			neurodes[i].guess(act, _forward[i], _forward[i+1]);
		}

		/* Compute top-layer errors to output[N-1],
		 * then learn */
		double d_output_size = output_size;
		for(size_t i=0; i < output_size; ++i) {
			double error = nn::error(expect[i], _forward[neurodes_count][i]);
			_backward[neurodes_count][i] = error;
			if(error < 0.0)  error = -error;
			avg_error += error / d_output_size;
		}

		// Back-propagate down to the second layer
		for(size_t neurode = last_n; neurode > 0; --neurode) {
			size_t neurode_next = neurode + 1;
			double accum = 0.0;
			for(size_t i=0; i < neurodes[neurode].outputSize(); ++i)
				accum += _backward[neurode_next][i];
			for(size_t i=0; i < neurodes[neurode].inputSize(); ++i)
				_backward[neurode][i] = accum * derive(_forward[neurode][i]);
			neurodes[neurode].learn(act, _forward[neurode], _backward[neurode_next], rate);
		}

		// Final iteration for the first layer
		double accum = 0.0;
		for(size_t i=0; i < neurodes[0].outputSize(); ++i)
			accum += _backward[1][i];
		/*for(size_t i=0; i < neurodes[0].inputSize(); ++i)
			_backward[0][i] = accum * derive(_forward[0][i]);*/
		neurodes[0].learn(act, _forward[0], _backward[1], rate);

		// Finally: hope
		return avg_error;
	}

	double Stripe::train(
		activation_func act, activation_func_deriv derive,
		DataSet& data, long long int which, double rate
	) {
		double avg_error = 0.0;
		double runs = data.size();
		if(which < 0) {
			for(DataRow& row : data) {
				++runs;
				avg_error += train(act, derive, row.inputs.data(), row.outputs.data(), rate) / runs;
			}
		} else if(! data.empty()) {
			DataRow& row = data[which % data.size()];
			avg_error = train(act, derive, row.inputs.data(), row.outputs.data(), rate);
		}
		return avg_error;
	}

	void Stripe::randomize() {
		for(size_t i=0; i < neurodes_count; ++i)
			neurodes[i].randomize();
		// bias = nn::random();
	}

}
