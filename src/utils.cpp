#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void load_script(
	const string & data_path,
	vector<string> & image_paths_vec,
	vector<int> & labels_vec,
	bool train=true
)
{
	// 1. Open file
	ifstream ifs;
	ifs.open(data_path + (train ? "/train.txt" : "/test.txt"));
	if (!ifs.is_open())
	{
		cerr << "Cannot load script file" << endl;
		exit(-1);
	}

	// 2. Parsing
	string path;
	int label;
	while (!ifs.eof())
	{
		ifs >> path >> label;
		image_paths_vec.push_back(data_path+"/"+path);
		labels_vec.push_back(label);
	}

}

void load_data(
	const string & data_path, 
	Mat & images, 
	Mat & labels, 
	bool train=true
)
{
	// 1. Load from script file
	vector<string> image_paths_vec;
	vector<int> labels_vec;
	load_script(data_path, image_paths_vec, labels_vec, train);

	// 2. Initilaze images/labels by num_examples and num_dims
	int num_examples = image_paths_vec.size();
	Mat img = imread(image_paths_vec[0], IMREAD_GRAYSCALE);
	if (img.empty()) { cerr << "Failed to read image : " << image_paths_vec[0] << endl; exit(-1); }
	int num_dims = img.size().area();
	images = Mat(num_examples, num_dims, CV_32FC1);
	labels = Mat(num_examples, 1, CV_32SC1);

	// 3. Flatten images and stacking
	for (size_t i = 0; i < num_examples; i++)
	{
		img = imread(image_paths_vec[i], IMREAD_GRAYSCALE);
		if (img.empty()) { cerr << "Failed to read image : " << image_paths_vec[0] << endl; exit(-1); }

		for (size_t c = 0; c < img.cols; c++)
		{
			for (size_t r = 0; r < img.rows; r++)
			{
				images.ptr<float>(i)[r*img.cols + c] = static_cast<float>(img.ptr<unsigned char>(r)[c]);
			}
		}

		labels.ptr<int>(i)[0] = labels_vec[i];
	}
}
