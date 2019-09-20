#ifndef PIX_BOX_HPP
#define PIX_BOX_HPP

#include "pix/canvas.hpp"



inline namespace pix {

	class Box : public Canvas {
	private:
		mutable bool cached = false;

	protected:
		gla::ShaderProgram& shader;
		gla::VertexArray va;
		gla::VertexBuffer vb;
		GLfloat z;
		GLfloat vertices[4][4];
		glm::vec4 color;

		GLuint texture_id;

	public:
		Box(
				gla::ShaderProgram& sp,
				unsigned int width,
				unsigned int height,
				glm::vec4 color = glm::vec4(1.0f),
				glm::vec2 top_left = glm::vec2(1.0, -1.0),
				glm::vec2 bottom_right = glm::vec2(-1.0, 1.0),
				GLfloat depth = 0.0f
		);

		void draw();
		void updateTexture();

		/* Below: state-altering functions that trigger the cache */

		inline void setPixel(unsigned x, unsigned y, glm::vec4 c) {
			cached = false;  Canvas::setPixel(x, y, c); }

		inline void fill(glm::vec4 c) {
			cached = false;  Canvas::fill(c); }

		inline void computePixels(void* d, color_func_t f) {
			cached = false;  Canvas::computePixels(d, f); }

		inline void computePixels(static_color_func_t f) {
			cached = false;  Canvas::computePixels(f); }

		/* Above: state-altering functions that trigger the cache */
	};

}

#endif
