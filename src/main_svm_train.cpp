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
	
	// 1. Set up traning data
	cout << "Load data ... " << endl;
	Mat trainingDataMat;
	Mat labelsMat;
	load_data(data_path, trainingDataMat, labelsMat, true);

	clock_t t_start, t_stop;

	// 2. Train the SVM
	cout << "Train SVM ..." << endl;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	t_start = clock();
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	t_stop = clock();
	cout << "   Elapsed : " 
		<< static_cast<float>(t_stop - t_start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

	// 3. Performance
	cout << "Performance ..." << endl;
	int num_examples = trainingDataMat.rows;
	int error_cnt = 0;

	t_start = clock();
	for (size_t i = 0; i < num_examples; i++)
	{
		Mat example = trainingDataMat.row(i);
		float response = svm->predict(example);
		if (response != labelsMat.ptr<int>(i)[0]) error_cnt++;
	}
	t_stop = clock();
	cout << "   Elapsed ( per example ) : "
		<< static_cast<float>(t_stop - t_start) / CLOCKS_PER_SEC * 1000 / num_examples << " ms" << endl;

	float precision = 1 - static_cast<float>(error_cnt) / num_examples;
	cout << "   Precision : " << precision << endl;

	// 4. Save model
	cout << "Save model ..." << endl;
	svm->save("../model/svm.model");

	return 0;
}