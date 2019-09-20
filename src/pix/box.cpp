#include "pix/box_async.hpp"

#include <mutex>



namespace pix {

	Box::Box(
			gla::ShaderProgram& sp,
			unsigned int w,
			unsigned int h,
			glm::vec4 color,
			glm::vec2 top_left,
			glm::vec2 bottom_right,
			GLfloat depth
	):
			Canvas::Canvas(w, h),
			shader (sp),
			vb (GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW),
			z (depth),
			vertices {
				{ top_left[1],     bottom_right[0], 0.0f, 0.0f },
				{ top_left[1],     top_left[0],     0.0f, 1.0f },
				{ bottom_right[1], top_left[0],     1.0f, 1.0f },
				{ bottom_right[1], bottom_right[0], 1.0f, 0.0f }
			},
			color (color)
	{
		vb.bufferData(vertices, 4 * 4 * sizeof(GLfloat));
		va.assignVertexBuffer(
			vb, sp.getAttrib("in_position"),
			2, GL_FLOAT, GL_FALSE,
			4 * sizeof(GLfloat), (GLfloat*) (0 * sizeof(GLfloat)) );
		va.assignVertexBuffer(
			vb, sp.getAttrib("in_tex"),
			2, GL_FLOAT, GL_FALSE,
			4 * sizeof(GLfloat), (GLfloat*) (2 * sizeof(GLfloat)) );

		glGenTextures(1, &texture_id);
		glBindTexture(GL_TEXTURE_2D, texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}


	void Box::updateTexture() {
		if(! cached) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture_id);
			glTexImage2D(
					GL_TEXTURE_2D, 0, GL_RGBA,
					Canvas::width, Canvas::height, 0,
					GL_RGBA, GL_FLOAT, Canvas::data
			);
			glGenerateMipmap(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);
			cached = true;
		}
	}


	void Box::draw() {
		va.bind();
		updateTexture();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture_id);
		glUniform1i(shader.getUniform("box_texture"), 0);
		glUniform1f(shader.getUniform("z"), 0.0f);
		glUniform4fv(shader.getUniform("color"), 1, &(color[0]));
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

}

namespace pix {

	AsyncBox::AsyncBox(
			gla::ShaderProgram& sp,
			unsigned int w,
			unsigned int h,
			glm::vec4 c,
			glm::vec2 tl,
			glm::vec2 br,
			GLfloat d
	):
			Box::Box (sp, w, h, c, tl, br, d),
			mutex ()
	{ }


	void AsyncBox::draw() {
		auto lock = std::unique_lock<std::mutex>();
		Box::draw();
	}

	void AsyncBox::updateTexture() {
		auto lock = std::unique_lock<std::mutex>();
		Box::updateTexture();
	}

}
