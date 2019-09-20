#ifndef PIX_PIX_HPP
#define PIX_PIX_HPP

#define GL3_PROTOTYPES 1
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include "box_async.hpp"



inline namespace pix {

	gla::ShaderProgram& get_shader();


	class Runtime {
	public:
		Runtime();
		~Runtime();
	};


	class Window {
	private:
		GLFWwindow* glfw_window;

	public:
		Window(unsigned int width, unsigned int height, const char* title = "Unnamed Window");
		~Window();

		inline bool shouldClose() const { return glfwWindowShouldClose(glfw_window); }
		inline operator       GLFWwindow* ()       { return glfw_window; }
		inline operator const GLFWwindow* () const { return glfw_window; }

		inline void close() { glfwSetWindowShouldClose(glfw_window, GL_TRUE); }

		void pollEvents();
		void swapBuffers();
	};

}


#endif
