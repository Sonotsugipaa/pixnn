#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <string>



inline namespace nn {

	using activation_func       = double (*)(double);
	using activation_func_deriv = double (*)(double);


	double random(double lower_bound = -1.0, double upper_bound = 1.0);

	double error(double expected, double guessed);


	struct DataRow {
		std::vector<double> inputs;
		std::vector<double> outputs;
	};

	using DataSet = std::vector<DataRow>;


	class Perceptron {
	protected:
		size_t input_size;
		double* weights;

	public:
		Perceptron(size_t input_size);
		Perceptron(const Perceptron&);
		Perceptron(Perceptron&&);
		~Perceptron();

		Perceptron& operator = (const Perceptron&);
		Perceptron& operator = (Perceptron&&);

		double guess(activation_func, double* inputs) const;
		void learn(activation_func_deriv, double* inputs, double error, double rate);
		void train(activation_func_deriv, DataSet& data, double rate);

		void randomize();

		inline std::vector<double> getWeights() const {
			std::vector<double> r;  r.reserve(input_size);
			for(size_t i=0; i < input_size; ++i)
				r.push_back(weights[i]);
			return r;
		}

		inline double getBias() const {
			return weights[input_size]; }

		inline void setBias(double value) {
			weights[input_size] = value; }
	};


	class Neurode {
	protected:
		size_t  input_size;
		size_t output_size;
		std::vector<Perceptron> perceptrons;

	public:
		Neurode(size_t inputs, size_t outputs);
		Neurode(const Neurode&);
		Neurode(Neurode&&);
		~Neurode();

		Neurode& operator = (const Neurode&);
		Neurode& operator = (Neurode&&);

		void guess(activation_func, double* inputs, double* outputs) const;
		void learn(activation_func, double* inputs, double* errors, double rate);
		void train(activation_func, DataSet& data, double rate);

		void randomize();

		constexpr size_t  inputSize() const { return  input_size; }
		constexpr size_t outputSize() const { return output_size; }

		inline       Perceptron& operator [] (unsigned i)       { return perceptrons[i]; }
		inline const Perceptron& operator [] (unsigned i) const { return perceptrons[i]; }
	};


	class Stripe {
	protected:
		size_t input_size;
		size_t output_size;
		size_t biggest_neurode; // Only needed for optimization, i.e. chain-buffering
		size_t neurodes_count;
		std::vector<Neurode> neurodes;

	private:
		/* _forward and _backward are internal buffers
		 * used for the back-propagation algorithm;
		 * additionally, _forward is checked to be ==nullptr
		 * to see if the stripe has been moved */
		mutable double** _forward;
		mutable double** _backward;

	public:
		Stripe(
				size_t inputs,
				std::vector<size_t> hidden_layer_sizes,
				size_t outputs
		);
		Stripe(const Stripe&);
		Stripe(Stripe&&);
		~Stripe();

		Stripe& operator = (const Stripe&);
		Stripe& operator = (Stripe&&);

		void guess(activation_func, double* inputs, double* outputs) const;
		double train(activation_func, activation_func_deriv, double* inputs, double* outputs, double rate);
		double train(activation_func, activation_func_deriv, DataSet& data, long long int which, double rate);

		void randomize();

		constexpr size_t  inputSize() const { return  input_size; }
		constexpr size_t outputSize() const { return output_size; }

		inline       Neurode& operator [] (unsigned i)       { return neurodes[i]; }
		inline const Neurode& operator [] (unsigned i) const { return neurodes[i]; }
	};


	class NeuralException {
	protected:
		std::string description;

	public:
		NeuralException(std::string description);

		inline const char* what() const noexcept {
			return description.c_str(); }

		inline const std::string& getDescription() const {
			return description; }
	};

}

#endif
