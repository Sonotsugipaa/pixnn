#include "nn/nn.hpp"

#include <cstdlib> // rand()



namespace nn {

	double random(double lower_bound, double upper_bound) {
		return ((rand() / (double) RAND_MAX) * (upper_bound - lower_bound)) + lower_bound;
	}

	double error(double exp, double got) {
		return got - exp;
	}

}
