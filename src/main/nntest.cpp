#include <iostream>



template<typename num>
constexpr num range(
		num value,
		num from_lo, num from_hi,
		num to_lo,   num to_hi
) {
	value *= (to_hi - to_lo);
	value /= (from_hi - from_lo);
	value += to_lo - from_lo;
	return value;
}


int main(int argn, char** args) {
	std::cout << "\033[1;93m";
	for(double i=0; i<=1; i+=0.125)
		std::cout << range<double>(i, 0.0, 1.0, -4.0, 4.0) << '\n';
	std::cout << "\033[m";

	return EXIT_FAILURE;
}
