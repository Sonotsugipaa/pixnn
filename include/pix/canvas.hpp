#ifndef PIX_CANVAS_HPP
#define PIX_CANVAS_HPP

#include "globject.hpp"
#include "shader.hpp"

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>



inline namespace pix {

	using color_func_t =
		glm::vec4(*)(void* static_data, unsigned int x, unsigned int y);

	using static_color_func_t =
		glm::vec4(*)(unsigned int x, unsigned int y);


	class Canvas {
	protected:
		unsigned int width, height;
		GLfloat* data;

	public:
		Canvas(unsigned width, unsigned height);
		Canvas(const Canvas &) = delete;

		inline Canvas(unsigned side_length):
				Canvas::Canvas(side_length, side_length)
		{ }

		~Canvas();

		void setPixel(unsigned x, unsigned y, glm::vec4 color);
		glm::vec4 getPixel(unsigned x, unsigned y) const;

		void fill(glm::vec4 color);

		constexpr unsigned int getWidth() const { return width; }
		constexpr unsigned int getHeight() const { return height; }

		void computePixels(void* data, color_func_t);
		void computePixels(static_color_func_t);
	};


	class CachedCanvas : public Canvas {
		void setPixel(unsigned x, unsigned y, glm::vec4 color);
		glm::vec4 getPixel(unsigned x, unsigned y) const;
		void fill(glm::vec4 color);
		void computePixels(void* data, color_func_t);
		void computePixels(static_color_func_t);
	};

}

#endif
