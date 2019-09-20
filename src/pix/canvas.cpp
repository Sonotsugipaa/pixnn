#include "pix/canvas.hpp"



namespace {
	constexpr unsigned int coords_to_index(unsigned x, unsigned y, unsigned width) {
		return 4 * (x + (y * width));
	}
}


Canvas::Canvas(unsigned w, unsigned h):
		width(w),  height(h),
		data (new GLfloat[4 * w * h])
{ }

Canvas::~Canvas() {
	if(data != nullptr) {
		delete[] data;  data = nullptr;
	}
}


void Canvas::setPixel(unsigned x, unsigned y, glm::vec4 color) {
	GLfloat* ptr = &data[coords_to_index(x, y, Canvas::width)];
	ptr[0] = color[0];
	ptr[1] = color[1];
	ptr[2] = color[2];
	ptr[3] = color[3];
}

glm::vec4 Canvas::getPixel(unsigned x, unsigned y) const {
	GLfloat* ptr = &data[coords_to_index(x, y, Canvas::width)];
	return glm::vec4(ptr[0], ptr[1], ptr[2], ptr[3]);
}


void Canvas::computePixels(static_color_func_t computeColor) {
	for(unsigned y=0; y < height; ++y) {
		for(unsigned x=0; x < width; ++x) {
			setPixel(x, y, computeColor(x, y));
		}
	}
}

void Canvas::computePixels(void* data, color_func_t computeColor) {
	for(unsigned y=0; y < height; ++y) {
		for(unsigned x=0; x < width; ++x) {
			setPixel(x, y, computeColor(data, x, y));
		}
	}
}

void Canvas::fill(glm::vec4 color) {
	GLfloat* ptr;
	for(unsigned y=0; y < height; ++y) {
		for(unsigned x=0; x < width; ++x) {
			/* Same as Canvas::stPixel(x, y, color), but potentially
			 * marginally optimized */
			ptr = &data[coords_to_index(x, y, Canvas::width)];
			ptr[0] = color[0];
			ptr[1] = color[1];
			ptr[2] = color[2];
			ptr[3] = color[3];
		}
	}
}
