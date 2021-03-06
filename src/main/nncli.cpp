#include "nn/nn.hpp"
#include "pix/pix.hpp"

#include <iostream>
#include <chrono>

#include <queue>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <cmath> // ::exp(...), ::tanh(...)



constexpr int BOX_SIZE = 128;
constexpr int TRAINING_SIZE = (BOX_SIZE < 128)? BOX_SIZE : 128;
constexpr double CLICK_REPEAT_S = 0.125;
constexpr double GRANULARITY = 16;
constexpr double FRAME_INTERVAL_S = 1.0 / 60.0;
constexpr double PRINT_INTERVAL_S = 0.5;
constexpr double DEF_LEARNING_RATE = 0.00001;



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


	double f1(double x) { return  x * 2.0; }
	double f2(double x) { return  0.3; }
	double f3(double x) { x*=2.0; return (x*x)-0.3; }

	double expected_value(double x, double y) {
		//return act_sign((y - f3(x)));
		//return act_sign(x);
		//{ return act_sign(x) * act_sign(y); }
		return ((x < 0.0) && (y > f3(x)))? 1.0 : -1.0;
	}

	DataSet gen_data(size_t size) {
		DataSet ds;  ds.reserve(size);
		for(size_t i=0; i < size; ++i) {
			double inputs[2] = { nn::random(-1.0, 1.0), nn::random(-1.0, 1.0) };
			ds.push_back(DataRow {
				{ inputs[0], inputs[1] },
				{ expected_value(inputs[0], inputs[1]) }
			});
		}
		return ds;
	}


	class Trainer {
	protected:
		Stripe* n;
		DataSet* ds;
		double rate;
		std::thread worker;
		std::mutex mutex;

		static void worker_func(
				Stripe** n, DataSet** ds,
				activation_func act,
				activation_func_deriv deriv,
				double* rate,
				std::mutex* mutex
		) {
			bool die = false;
			do {
				auto lock = std::unique_lock<std::mutex>(*mutex);
				if(*n == nullptr) {
					die = true;
				} else {
					int random = std::rand();
					if(random < 0)  random = -random;
					(*n)->train(act, deriv, **ds, random, *rate);
				}
			} while(! die);
		}

	public:
		Trainer(
				Stripe* neurode, DataSet* dataset,
				activation_func activate, activation_func_deriv derivate,
				double learning_rate
		):
				n (neurode),
				ds (dataset),
				rate (learning_rate),
				worker (worker_func, &n, &ds, activate, derivate, &rate, &mutex)
		{ }

		~Trainer() { stop(); }

		std::unique_lock<std::mutex> acquireLock() {
			return std::unique_lock<std::mutex>(mutex);
		}

		inline double getLearningRate() const { return rate; }
		inline void setLearningRate(double value) { rate = value; }

		void stop() {
			if(n != nullptr) {
				n = nullptr;
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
		SHOW_DERIVS, UNDO
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
				case GLFW_KEY_R:          stored_action = Action::RESET;          break;
				case GLFW_KEY_Q:          stored_action = Action::QUIT;           break;
				case GLFW_KEY_G:          stored_action = Action::REGEN;          break;
				case GLFW_KEY_T:          stored_action = Action::SHOW_TRAINING;  break;
				case GLFW_KEY_D:          stored_action = Action::SHOW_DERIVS;    break;
				case GLFW_KEY_PAGE_UP:    stored_action = Action::RATE_UP;        break;
				case GLFW_KEY_PAGE_DOWN:  stored_action = Action::RATE_DOWN;      break;
				case GLFW_KEY_HOME:       stored_action = Action::GRAN_UP;        break;
				case GLFW_KEY_END:        stored_action = Action::GRAN_DOWN;      break;
				case GLFW_KEY_ESCAPE:     stored_action = Action::QUIT;           break;
				case GLFW_KEY_DELETE:     stored_action = Action::UNDO;           break;
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
	/*box.computePixels([] (unsigned x, unsigned y) {
		if((x == 0 && y == 0)
		|| (x == 3 && y == 3)
		|| (x == 4 && y == 4)) return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		else return glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
	});*/
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
			mouse_pressed_last = current_time;
			last_click = c;
		} else {
			mouse_pressed = false;
		}
	}
}


int main(int argn, char** args) {
	Stripe n = Stripe(2, { 32, 16 }, 1);
	//DataSet ds = gen_data(TRAINING_SIZE);
	DataSet ds;

	pix::Runtime runtime;
	pix::Window* window = new pix::Window(650, 650, "Pixnn");
	gla::ShaderProgram& shader = pix::get_shader();
	pix::AsyncBox frame = pix::AsyncBox(shader, BOX_SIZE, BOX_SIZE);
	Trainer trainer = Trainer(&n, &ds, act_tanh, act_tanh_deriv, DEF_LEARNING_RATE);

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
		if((time - last_time) > FRAME_INTERVAL_S) {
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
				case Action::RESET: {
					auto lock = trainer.acquireLock();
					n.randomize();
					poll_nn(n, frame, show_derivs? act_tanh_deriv : act_tanh, 0);
					std::cout << "-----  NN reset  -----" << '\n';
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

			if(show_training)
			for(DataRow& r : ds) {
				auto lock = trainer.acquireLock();
				frame.setPixel(
					((r.inputs[0] * (BOX_SIZE/2.0)) + (BOX_SIZE/2.0)),
					((r.inputs[1] * (BOX_SIZE/2.0)) + (BOX_SIZE/2.0)),
					glm::vec4(r.outputs[0], 0.0f, -r.outputs[0], 1.0f)
				);
			}
			frame.draw();

			window->swapBuffers();
			stored_action = Action::NONE;
		}
	}

	trainer.stop();
	delete window;

	return EXIT_FAILURE;
}
