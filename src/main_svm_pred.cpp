#include <iostream>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include "utils.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, const char** argv)
{
	// Configuration
	const char* data_path = "../data/whole";
	const char* svm_model_path = "../model/svm.model";

	// 1. Set up traning data
	cout << "Load data ... " << endl;
	Mat testingDataMat;
	Mat labelsMat;
	load_data(data_path, testingDataMat, labelsMat, true);

	// 2. Load SVM model
	cout << "Load SVM model ... " << endl;
	Ptr<SVM> svm = SVM::load(svm_model_path);

	// 3. Evaluation or Test
	cout << "Performance ... " << endl;
	clock_t t_start, t_stop;
	int num_examples = testingDataMat.rows;
	int error_cnt = 0;

	t_start = clock();
	for (size_t i = 0; i < num_examples; i++)
	{
		Mat example = testingDataMat.row(i);
		float response = svm->predict(example);
		if (response != labelsMat.ptr<int>(i)[0]) error_cnt++;
	}
	t_stop = clock();
	cout << "   Elapsed ( per example ) : "
		<< static_cast<float>(t_stop - t_start) / CLOCKS_PER_SEC * 1000 / num_examples << " ms" << endl;

	float precision = 1 - static_cast<float>(error_cnt) / num_examples;
	cout << "   Precision : " << precision << endl;

	return 0;
}