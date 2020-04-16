#ifndef NN_NNCLI_ERROR_TYPES_HPP
#define NN_NNCLI_ERROR_TYPES_HPP

/* This header only contains a collection of classes to be
 * used for exception handling, ideally to avoid the bloat
 * of STL headers that may significantly increase compilation
 * times for RTTI features that are not really needed OR used.
 *
 * As its name suggests, this file "belongs" to src/main/nncli.cpp. */

#include <string>



namespace nn {

	using errcode_t = unsigned;


	enum errcode_e : errcode_t {
		ERRCODE_NO_ERROR = 0,
		ERRCODE_NOT_A_NUMBER,
		ERRCODE_INVALID_NUMBER,
		ERRCODE_INVALID_OPTION
	};


	struct Fault {
		errcode_t code;
		std::string message;

		inline Fault(errcode_t code, std::string message):
				code (code),
				message (message)
		{ }
	};

	struct Exception : public Fault {
		inline Exception(errcode_t code, std::string message):
				Fault (std::move(code), std::move(message))
		{ }
	};

	struct InvalidNumberException : public Exception {
		template<typename N>
		inline InvalidNumberException(N number):
				Exception (
					ERRCODE_INVALID_NUMBER,
					"invalid number \"" + std::to_string(number) + '"')
		{ }

		inline InvalidNumberException(std::string message):
				Exception (ERRCODE_INVALID_NUMBER, message)
		{ }
	};

	struct NotANumberException : public Exception {
		inline NotANumberException(std::string rep):
				Exception (
					ERRCODE_NOT_A_NUMBER,
					'"' + rep + "\" is not a number")
		{ }
	};

	struct InvalidOptionException : public Exception {
		inline InvalidOptionException(std::string option):
				Exception (
					ERRCODE_INVALID_OPTION,
					"invalid option \"" + std::move(option) + '"')
		{ }
	};

}

#endif
