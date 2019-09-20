#ifndef PIX_BOX_ASYNC_HPP
#define PIX_BOX_ASYNC_HPP

#include "pix/box.hpp"

#include <mutex>



inline namespace pix {

	class AsyncBox : public Box {
	protected:
		std::mutex mutex;

	public:
		AsyncBox(
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

		inline       std::mutex& getMutex()       { return mutex; }
		inline const std::mutex& getMutex() const { return mutex; }
	};

}

#endif
