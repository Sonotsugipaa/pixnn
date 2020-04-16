#include "nn/nn.hpp"
#include "pix/pix.hpp"

#include "nn/nncli_error_types.hpp"

#include <iostream>
#include <chrono>
#include <queue>
#include <limits>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <cmath> // ::exp(...), ::tanh(...)



constexpr int BOX_SIZE = 128;
constexpr int WIN_SIZE = 384;
constexpr int TRAINING_SIZE = (BOX_SIZE < 128)? BOX_SIZE : 128;
constexpr unsigned THROTTLE_MAX = 10;
constexpr unsigned TRAIN_BATCH_SIZE = 6;
constexpr double ERROR_AVERAGE_EXP_ALPHA = 0.05 / TRAIN_BATCH_SIZE;
constexpr double CLICK_REPEAT_S = 0.1;
constexpr double GRANULARITY = 4;
constexpr double FRAME_INTERVAL_S = 1.0 / 60.0;
constexpr double PRINT_INTERVAL_S = 0.5;
constexpr double PERF_ERROR_TOLERANCE = 0.1;
constexpr double DEF_LEARNING_RATE = 0.0001;
constexpr const char* LAYER_LIST = "32,16";

constexpr const double THROTTLE_TABLE_MS[THROTTLE_MAX+1] {
	200.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 4.0, 2.0, 1.0, 0.0
};
constexpr const double THROTTLE_TABLE_RATE[THROTTLE_MAX+1] {
	0.03125 / 32, 0.03125 / 16, 0.03125 / 8, 0.03125 / 4, 0.03125 / 2,
	0.03125, 0.0625, 0.125, 0.25, 0.5,
	1.0
};



namespace {

	constexpr double act_sign(double x) {
		return (x > 0.0)? 1.0 : -1.0;
	}

	constexpr double act_logistic(double x) {
		return (1.0 / (1.0 + ::exp(-x))) -1.0;
	}

	constexpr double act_logistic_deriv(double x) {
		return act_logistic(x) * (1.0 - act_logistic(x));
	}

	constexpr double act_relu(double x) {
		return (x > 0.0)? x : 0.0;
	}

	constexpr double act_relu_deriv(double x) {
		return (x > 0.0)? 1.0 : 0.0;
	}

	constexpr double act_tanh(double x) {
		return ::tanh(x);
	}

	constexpr double act_tanh_deriv(double x) {
		x = act_tanh(x);
		return 1.0 - (x*x);
	}


	class Trainer {
	protected:
		Stripe* n;
		DataSet* ds;
		volatile unsigned train_batch_size;
		volatile double rate;
		volatile double last_error = 0;
		volatile unsigned throttle; // 0 to THROTTLE_MAX
		volatile bool die;
		std::thread worker;
		std::mutex mutex;

		static void worker_func(
				Stripe** n, DataSet** ds,
				activation_func act,
				activation_func_deriv deriv,
				volatile const unsigned* train_batch_size,
				volatile const double* rate, volatile const unsigned* throttle,
				volatile double* last_error,
				volatile bool* die_monitor,
				std::mutex* mutex
		) {
			while(! *die_monitor) {
				unsigned throttle_actual;
				{
					unsigned throttle_error =
						*throttle * (((double) *throttle) * (*last_error / PERF_ERROR_TOLERANCE));
					throttle_actual = *throttle;
					if(throttle_error < throttle_actual) {
						throttle_actual = throttle_error; }
					if(throttle_error > THROTTLE_MAX) {
						throttle_error = THROTTLE_MAX; }
					double last_error_local = 0.0;
					unsigned train_batch_size_local = *train_batch_size;
					for(unsigned i=0; i < train_batch_size_local; ++i) {
						unsigned random = (unsigned) nn::random(0, (*ds)->size());
						auto lock = std::unique_lock<std::mutex>(*mutex);
						last_error_local += (*n)->train(
							act, deriv, **ds, random,
							*rate * THROTTLE_TABLE_RATE[throttle_error]);
					}
					last_error_local /= train_batch_size_local;
					*last_error = last_error_local =
						(*last_error * (1.0 - ERROR_AVERAGE_EXP_ALPHA)) +
						(last_error_local * ERROR_AVERAGE_EXP_ALPHA);
				}
				if(
						throttle_actual <= THROTTLE_MAX &&
						THROTTLE_TABLE_MS[throttle_actual] > 0.0
				) {
					std::this_thread::yield();
						std::this_thread::sleep_for(
							std::chrono::milliseconds((int) THROTTLE_TABLE_MS[throttle_actual]));
				}
			}
		}

	public:
		Trainer(
				Stripe* neurode, DataSet* dataset,
				activation_func activate, activation_func_deriv derivate,
				unsigned training_batch_size,
				double learning_rate, unsigned throttle
		):
				n (neurode),
				ds (dataset),
				train_batch_size (training_batch_size),
				rate (learning_rate),
				throttle (throttle < THROTTLE_MAX? throttle : THROTTLE_MAX),
				die (false),
				// Watch out below: do not accidentally pass pointers to parameters
				worker (
					worker_func, &n, &ds,
					activate, derivate,
					&train_batch_size,
					&rate, &throttle, &last_error,
					&die, &mutex)
		{ }

		~Trainer() { stop(); }

		std::unique_lock<std::mutex> acquireLock() {
			return std::unique_lock<std::mutex>(mutex);
		}

		inline double getLearningRate() const { return rate; }
		inline void setLearningRate(double value) { rate = value; }

		inline unsigned getBatchSize() const { return train_batch_size; }
		inline void setBatchSize(unsigned value) { train_batch_size = value; }

		inline double getThrottle() const { return throttle; }
		inline void setThrottle(unsigned value) {
			if(value > THROTTLE_MAX) {
				value = THROTTLE_MAX; }
			throttle = value;
		}

		inline double getError() const { return last_error; }

		void stop() {
			die = true;
			if(worker.joinable()) {
				worker.join();
			}
		}
	};


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


	enum class Action {
		NONE, RESET, REGEN, RATE_UP, RATE_DOWN,
		GRAN_UP, GRAN_DOWN, QUIT, SHOW_TRAINING,
		THROTTLE_UP, THROTTLE_DOWN, SHOW_DERIVS,
		PRINT_ERROR, UNDO
	};

	struct Click {
		GLFWwindow* window = nullptr;
		double x, y;
		int button, action, mods;
	};

	bool mouse_pressed;
	double mouse_pressed_last;
	Click last_click;

	Action stored_action = Action::NONE;
	std::queue<Click> clicks;

	void key_callback(GLFWwindow* w, int k, int c, int action, int mods) {
		if(action == GLFW_RELEASE) {
			switch(k) {
				case GLFW_KEY_R:            stored_action = Action::RESET;          break;
				case GLFW_KEY_Q:            stored_action = Action::QUIT;           break;
				case GLFW_KEY_G:            stored_action = Action::REGEN;          break;
				case GLFW_KEY_T:            stored_action = Action::SHOW_TRAINING;  break;
				case GLFW_KEY_D:            stored_action = Action::SHOW_DERIVS;    break;
				case GLFW_KEY_P:            stored_action = Action::PRINT_ERROR;    break;
				case GLFW_KEY_PAGE_UP:      stored_action = Action::RATE_UP;        break;
				case GLFW_KEY_PAGE_DOWN:    stored_action = Action::RATE_DOWN;      break;
				case GLFW_KEY_HOME:         stored_action = Action::GRAN_UP;        break;
				case GLFW_KEY_END:          stored_action = Action::GRAN_DOWN;      break;
				case GLFW_KEY_ESCAPE:       stored_action = Action::QUIT;           break;
				case GLFW_KEY_DELETE:       stored_action = Action::UNDO;           break;
				case GLFW_KEY_KP_ADD:       stored_action = Action::THROTTLE_UP;    break;
				case GLFW_KEY_KP_SUBTRACT:  stored_action = Action::THROTTLE_DOWN;  break;
			}
		}
	}

	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		clicks.push(Click{ window, x, y, button, action, mods });
	}

}



void poll_nn(Stripe n, pix::AsyncBox& box, activation_func act, size_t pix_throughput) {
	struct pack {
		Stripe& n;
		activation_func act;
		size_t pix_throughput;
		pix::AsyncBox& box;
	} packed = pack { n, act, pix_throughput, box };
	box.computePixels(&packed, [] (void* packed, unsigned x, unsigned y) -> glm::vec4 {
		double inputs[2] = {
			(static_cast<double>(x) - (BOX_SIZE/2)) / (BOX_SIZE/2),
			(static_cast<double>(y) - (BOX_SIZE/2)) / (BOX_SIZE/2)
		};
		pack* unpacked = reinterpret_cast<pack*>(packed);
		if(
				(unpacked->pix_throughput == 0) ||
				(::rand() % (unpacked->pix_throughput) == 0)
		) {
			double dguess;
			unpacked->n.guess(unpacked->act, inputs, &dguess);
			float guess = dguess;
			if(guess >  1.0f)  guess =  1.0f;
			else
			if(guess < -1.0f)  guess = -1.0f;
			if(guess > 0.0f) {
				return glm::vec4(1.0f, 0.6f, 0.0f, guess);
			} else {
				return glm::vec4(0.0f, 0.6f, 1.0f, -guess);
			}
		} else {
			return unpacked->box.getPixel(x, y);
		}
	});
}


void add_point(
		double x, double y, int button, int mod,
		Trainer& trainer, DataSet& dataset,
		double win_width, double win_height
) {
	if(
			(x >= 0.0) && (x < win_width) &&
			(y >= 0.0) && (y < win_height)
	) {
		double put_value = 0.0;
		x = range<double>(x, 0, win_width,  -1.0,  1.0);
		y = range<double>(y, 0, win_height,  1.0, -1.0);
		switch(button) {
			case GLFW_MOUSE_BUTTON_LEFT:    put_value =  1.0;  break;
			case GLFW_MOUSE_BUTTON_MIDDLE:  put_value =  0.0;  break;
			case GLFW_MOUSE_BUTTON_RIGHT:   put_value = -1.0;  break;
		}
		if(0 != (mod & GLFW_MOD_SHIFT))  put_value /= 2.0;
		if((button == GLFW_MOUSE_BUTTON_MIDDLE) || (put_value != 0.0)) {
			auto t_lock = trainer.acquireLock();
			dataset.push_back({{ x, y }, { put_value }});
		}
	}
}

void process_clicks(
		Trainer& trainer, DataSet& dataset,
		double win_width, double win_height,
		double current_time
) {
	while(! clicks.empty()) {
		Click c = clicks.front();
		clicks.pop();
		if(c.action == GLFW_PRESS) {
			add_point(c.x, c.y, c.button, c.mods, trainer, dataset, win_width, win_height);
			mouse_pressed = true;
			mouse_pressed_last = current_time + CLICK_REPEAT_S;
			last_click = c;
		} else {
			mouse_pressed = false;
		}
	}
}


template<typename N, N (*strton)(const char*, char**, int)>
N parse_number(
		const char* str,
		const char* allowed_endings = "",
		const char** end_saveptr = nullptr
) {
	const char* str_save;
	/* The following const_cast, as far as I could find from an
	 * unsatisfying search online, is valid and needed because of a flaw in
	 * the C implementation of strton-like functions. */
	N buffer_n = strton(str, const_cast<char**>(&str_save), 0);
	if(end_saveptr != nullptr) {
		*end_saveptr = str_save; }
	do {
		if(*str_save == *allowed_endings) {
			return buffer_n; }
		++allowed_endings;
	} while(*(allowed_endings-1) != '\0');
	if(str == str_save) {
		return 0; }
	if(buffer_n == std::numeric_limits<N>::max()) {
		return buffer_n; }
	throw nn::NotANumberException(str_save);
}


std::vector<size_t> parse_uint_list(
		const char* list, const char* allowed_endings,
		size_t min = 0, size_t max = std::numeric_limits<size_t>::max()
) {
	std::vector<size_t> r;  r.reserve(4);
	char ending = *list;
	const char* list_save = list;
	while(ending != '\0') {
		size_t n = parse_number<unsigned long, std::strtoul>(list, allowed_endings, &list);
		if(n < min) {
			throw nn::InvalidNumberException("number must be greater than " + std::to_string(n)); }
		if(n > max) {
			throw nn::InvalidNumberException("number must be lower than " + std::to_string(n)); }
		if(list != list_save) {
			r.push_back(n); }
		ending = *list;
		++list;
	}
	return r;
}


struct Options {
	std::vector<size_t> layer_list = parse_uint_list(LAYER_LIST, ",", 1);
	unsigned win_width = WIN_SIZE;
	unsigned win_height = WIN_SIZE;
	unsigned train_batch_size = TRAIN_BATCH_SIZE;

	/* changing the following fields is currently unsupported,
	 * and *will* create bugs before it is */
	unsigned res_x = BOX_SIZE;
	unsigned res_y = BOX_SIZE;
};

Options parse_arguments(char** args) {
	Options opts;
	std::string arg;
	while(*args != nullptr) {
		arg = *args;
		// TODO: re-implement this if-ladder as a string-enum map lookup
		if(arg == "--size") {
			++args;
			if(*args == nullptr) {
				throw nn::InvalidOptionException(
					"option \"--size\" requires an argument (valid formats: "
					"\"<width>x<height>\", \"<width>x\", "
					"\"<square_size>\" or \"x<height>\",)"); }
			const char* end;
			decltype(opts.win_width) n =
				parse_number<unsigned long, std::strtoul>(*args, "x", &end);
			if(end != *args) {
				opts.win_width = n; }
			// no 'else' here, because the size strings "x250" and "250x" allowed and valid
			if(*end == '\0') {
				opts.win_height = opts.win_width; }
			else
			if(*end == 'x' || *end == 'X') {
				const char* end_beg = ++end; // store the position of the char after 'x'
				n = parse_number<long unsigned, std::strtoul>(end, "", &end);
				if(end != end_beg) {
					opts.win_height = n; }
			}
		} else
		if(arg == "--layers") {
			++args;
			if(*args == nullptr) {
				throw nn::InvalidOptionException(
					"option \"--layers\" requires an argument (valid formats: "
					"comma-separated list, colon-separated list or "
					"semicolon-separated list, where every item is an integer number "
					"greater than 0"); }
			opts.layer_list = parse_uint_list(*args, ",:;", 1);
		}
		++args;
	}
	return opts;
}


int main(int argn, char** args) {
	try {
		Options options = parse_arguments(args+1);

		Stripe n = Stripe(2, options.layer_list, 1);
		DataSet ds;

		pix::Runtime runtime;
		pix::Window* window = new pix::Window(options.win_width, options.win_height, "PixNN");
		gla::ShaderProgram& shader = pix::get_shader();
		pix::AsyncBox frame = pix::AsyncBox(shader, options.res_x, options.res_y);
		Trainer trainer = Trainer(
				&n, &ds, act_tanh, act_tanh_deriv,
				options.train_batch_size, DEF_LEARNING_RATE, THROTTLE_MAX);

		bool show_training = true;
		bool show_derivs = false;
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		double last_time = 0.0;
		double time = 0.0;
		size_t granularity = GRANULARITY;
		glfwSetTime(time);

		glfwSetKeyCallback(*window, key_callback);
		glfwSetMouseButtonCallback(*window, mouse_button_callback);

		poll_nn(n, frame, show_derivs? act_tanh_deriv : act_tanh, 0);

		while(! window->shouldClose()) {
			time = glfwGetTime();
			if(
					(time - last_time) <
					(
						FRAME_INTERVAL_S +
						(THROTTLE_TABLE_MS[(unsigned) trainer.getThrottle()] / 1000.0)
					)
			) {
				double sleep_ms = THROTTLE_TABLE_MS[(unsigned) trainer.getThrottle()];
				if(sleep_ms < (FRAME_INTERVAL_S * 1.02)) {
					sleep_ms = FRAME_INTERVAL_S * 1.02; }
				std::this_thread::sleep_for(std::chrono::milliseconds((int) sleep_ms));
			} else {
				last_time = time;

				int win_width, win_height;
				glfwGetFramebufferSize(*window, &win_width, &win_height);
				glViewport(0, 0, win_width, win_height);
				window->pollEvents();
				process_clicks(trainer, ds, win_width, win_height, time);

				if(mouse_pressed) {
					if(mouse_pressed_last + CLICK_REPEAT_S < time) {
						mouse_pressed_last = time;
						double x, y;
						glfwGetCursorPos(*window, &x, &y);
						add_point(
								x, y, last_click.button, last_click.mods,
								trainer, ds, win_width, win_height);
					}
				}

				switch(stored_action) {
					case Action::QUIT:           window->close();  break;
					case Action::SHOW_TRAINING:
						show_training = ! show_training;
						std::cout << "----- " << (show_training? "Show":"Hid")
						          << "ing training data -----\n";
						break;
					case Action::SHOW_DERIVS:
						show_derivs = ! show_derivs;
						std::cout << "----- " << (show_derivs? "En":"Dis")
						          << "abled derivative mode -----\n";
						break;
					case Action::RATE_UP: {
						trainer.setLearningRate(trainer.getLearningRate() * 2.0);
						std::cout << "Rate: " << trainer.getLearningRate() << '\n';
					} break;
					case Action::RATE_DOWN: {
						trainer.setLearningRate(trainer.getLearningRate() / 2.0);
						std::cout << "Rate: " << trainer.getLearningRate() << '\n';
					} break;
					case Action::GRAN_UP: {
						granularity *= 2;
						std::cout << "Granularity: " << granularity << '\n';
					} break;
					case Action::GRAN_DOWN: {
						granularity /= 2;  if(granularity < 1)  granularity = 1;
						std::cout << "Granularity: " << granularity << '\n';
					} break;
					case Action::THROTTLE_UP: {
						trainer.setThrottle(trainer.getThrottle() + 1);
						std::cout << "Performance Throttle: " << trainer.getThrottle() << '\n';
					} break;
					case Action::THROTTLE_DOWN: {
						trainer.setThrottle(trainer.getThrottle() - 1);
						std::cout << "Performance Throttle: " << trainer.getThrottle() << '\n';
					} break;
					case Action::RESET: {
						auto lock = trainer.acquireLock();
						n.randomize();
						poll_nn(n, frame, show_derivs? act_tanh_deriv : act_tanh, 0);
						std::cout << "-----  NN reset  -----" << '\n';
					} break;
					case Action::PRINT_ERROR: {
						std::cout << "Current error: " << trainer.getError() << '\n';
					} break;
					case Action::REGEN: {
						auto lock = trainer.acquireLock();
						ds.resize(0);
						std::cout << "-----  Canvas cleared  -----" << '\n';
					} break;
					case Action::UNDO: {
						auto lock = trainer.acquireLock();
						ds.pop_back();
						std::cout << "-----  Undo last point  -----" << '\n';
					} break;
					default:  break;
				}

				glClear(GL_COLOR_BUFFER_BIT);

				// Enable transparency
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glAlphaFunc(GL_GREATER, 0.0001f);
				glEnable(GL_BLEND);
				glEnable(GL_ALPHA_TEST);

				poll_nn(n, frame, show_derivs? act_tanh_deriv : act_tanh, granularity);

				if(show_training) {
					for(DataRow& r : ds) {
						frame.setPixel(
							((r.inputs[0] * (options.res_x / 2.0)) + (options.res_x / 2.0)),
							((r.inputs[1] * (options.res_y / 2.0)) + (options.res_y / 2.0)),
							glm::vec4(r.outputs[0], 0.0f, -r.outputs[0], 1.0f)
						);
					}
				}
				frame.draw();

				window->swapBuffers();
				stored_action = Action::NONE;
			}
		}

		trainer.stop();
		delete window;

		return EXIT_SUCCESS;
	} catch(nn::Fault& fault) {
		std::cerr << "An error has occurred: " << fault.message << '\n';
		return EXIT_FAILURE;
	}
}
