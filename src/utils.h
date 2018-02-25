#ifndef _UTILS_
#define _UTILS_

#include <vector>
#include <string>

#include <opencv2/core.hpp>

void load_script(
	const std::string & data_path,
	std::vector<std::string> & image_paths_vec,
	std::vector<int> & labels_vec,
	bool train = true
);

void load_data(
	const std::string & data_path,
	cv::Mat & images,
	cv::Mat & labels,
	bool train = true
);

#endif // !_UTILS_
