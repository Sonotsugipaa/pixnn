#include "pix/pix.hpp"

#include <iostream>



constexpr const char* vertex_shader =
	"#version 130\n"
	"in vec2 in_position;\n"
	"in vec2 in_tex;\n"
	"uniform float z;\n"
	"out vec2 f_tex;\n"
	"void main(void) {\n"
		"gl_Position = vec4(in_position, 0.0, 1.0);\n"
		"f_tex = in_tex;\n"
	"}\n";

constexpr const char* fragment_shader =
	"#version 130\n"
	"in vec2 f_tex;\n"
	"uniform vec4 color;\n"
	"uniform sampler2D box_texture;\n"
	"out vec4 out_color;\n"
	"void main(void) {\n"
		"out_color = texture(box_texture, f_tex) * color;\n"
	"}\n";



namespace {

	void err(const char * msg) {
		std::cout << "Error: " << msg << '\n';
		throw EXIT_FAILURE;
	}


	void errorCallback(int err, const char* msg) {
		std::cerr << msg << " (" << err << ")" << std::endl;
	}

}



namespace pix {

	gla::ShaderProgram* shader;

	gla::ShaderProgram& get_shader() {
		if(shader == nullptr) {
			shader = new gla::ShaderProgram(vertex_shader, fragment_shader);
		}
		return *shader;
	}

}

namespace pix {

	Runtime::Runtime() {
		glfwSetErrorCallback(errorCallback);
		if(! glfwInit()) err("Could not initialize GLFW");
	}

	Runtime::~Runtime() {
		glfwTerminate();
	}

}

namespace pix {

	Window::Window(unsigned int width, unsigned int height, const char* title) {
		glfw_window = glfwCreateWindow(width, height, title, NULL, NULL);
		glfwMakeContextCurrent(glfw_window);
		if(! glfw_window)  err("Could not create window");
		glewExperimental = GL_TRUE;
		if(glewInit() != 0)  err("Could not initialize GLEW");
	}

	Window::~Window() {
		if(glfw_window != nullptr) {
			glfwDestroyWindow(glfw_window);
			glfw_window = nullptr;
		}
	}


	void Window::pollEvents() {
		glfwPollEvents();
	}

	void Window::swapBuffers() {
		glfwSwapBuffers(glfw_window);
	}

}
