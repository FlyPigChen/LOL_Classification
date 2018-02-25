#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
using namespace cv;

vector<string> read_filelist(const string file_path)
{
	ifstream ifs;
	ifs.open(file_path+"/filelist.txt", std::ios::in);
	
	if (!ifs.is_open())
	{
		cerr << "Cannot open file_path" << endl;
		exit(-1);
	}

	vector<string> filelist;
	string item;
	while (!ifs.eof())
	{
		ifs >> item;
		filelist.push_back(file_path+"/"+item);
	}

	ifs.close();
	return filelist;
}

void get_dataset(
	const vector<string> & Anny_filelist,
	const vector<string> & Leesin_filelist,
	vector<Mat> & images,
	vector<int> & lables,
	Size & avg_size)
{
	double avg_width = 0.0;
	double avg_height = 0.0;
	
	// 1. Read data about Anny (label:0)
	for (string str : Anny_filelist)
	{
		Mat image = imread(str, IMREAD_GRAYSCALE);
		if (image.empty())
		{
			cerr << "Image " << str << " cannot open !" << endl;
			exit(-1);
		}
		images.push_back(image);
		lables.push_back(-1);
		avg_width += image.cols;
		avg_height += image.rows;
	}

	// 2. Read data about Leesin (label:1)
	for (string str : Leesin_filelist)
	{
		Mat image = imread(str, IMREAD_GRAYSCALE);
		if (image.empty())
		{
			cerr << "Image " << str << " cannot open !" << endl;
			exit(-1);
		}

		images.push_back(image);
		lables.push_back(1);
		avg_width += image.cols;
		avg_height += image.rows;
	}

	int total_num = Anny_filelist.size() + Leesin_filelist.size();
	avg_width /= total_num;
	avg_height /= total_num;

	avg_size.width = static_cast<int>(avg_width);
	avg_size.height = static_cast<int>(avg_height);
}

vector<Mat> preprocess(vector<Mat> & images, Size avg_size)
{
	vector<Mat> images_pre;

	for (Mat img : images)
	{
		Mat img_resize;
		resize(img, img_resize, avg_size);
		images_pre.push_back(img_resize);
	}

	return images_pre;
}

void deserilize(
	const string image_root_path, 
	const string script_root_path,
	vector<Mat> & images, 
	vector<int> & labels
)
{
	ofstream train_script_ofs;
	ofstream test_script_ofs;

	train_script_ofs.open(script_root_path + "/train.txt");
	test_script_ofs.open(script_root_path + "/test.txt");
	
	if (!train_script_ofs.is_open() || !test_script_ofs.is_open())
	{
		cerr << "Fail to write train/test script file" << endl;
		exit(-1);
	}

	for (size_t i=0; i < images.size(); i++)
	{

		stringstream path;
		stringstream fn;

		fn << labels[i] << "_" << i << ".png";
		path << image_root_path << "/" << fn.str();

		imwrite(path.str(), images[i]);

		if (i < 10)
		{
			test_script_ofs << fn.str() << " " << labels[i] << endl;
		}
		else
		{
			train_script_ofs << fn.str() << " " << labels[i] << endl;
		}
	}

	train_script_ofs.close();
	test_script_ofs.close();
}

int main(int argc, const char** argv)
{
	// Configuration
	const char* Anny_root_path = "../data/Anny/cropped";
	const char* Leesin_root_path = "../data/Leesin/cropped";
	const char* image_root_path = "../data/whole";
	const char* script_root_path = "../data/whole";

	// Read filelist
	cout << "Read filelist ..." << endl;
	vector<string> Anny_filelist = read_filelist(Anny_root_path);
	vector<string> Leesin_filelist = read_filelist(Leesin_root_path);

	// Get dataset
	cout << "Get dataset ..." << endl;
	vector<Mat> images;
	vector<int> labels;
	Size avg_size;
	get_dataset( Anny_filelist, Leesin_filelist, images, labels, avg_size);

	// Preprocess -- resize
	cout << "Preprocess ..." << endl;
	vector<Mat> images_pre = preprocess(images, avg_size);

	// Write to disk
	cout << "Deserilize ..." << endl;
	deserilize(image_root_path, script_root_path, images_pre, labels);

	cout << "Finish ..." << endl;
	return 0;
}