#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>  
#include <time.h>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <stdio.h>
#include<sys/time.h>
#define BAD_TRAIN_FILES_0704 6322
#define BAD_TRAIN_FILES_0711 5736
#define BAD_TRAIN_FILES_0715 28000
#define GOOD_TRAIN_FILES_0704 3161
#define GOOD_TRAIN_FILES_0711 2868
#define GOOD_TRAIN_FILES_0715 13000
#define GOOD_TRAIN_FILES_0303 19000
#define BAD_TRAIN_FILES_0303 39000
#define HARD_TRAIN_FILES 0
using namespace std;
using namespace cv;
using namespace ml;
#define UNTRAIN
void get_goodtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
  //  string dir = "../data/train/gooddata/";
  string dir = "../../../data/20190704/positive/";
    vector<string> files;
    // for (int i = 1; i < GOOD_TRAIN_FILES; i++)								// 取前400张数字0来训练
	// {
	// 	files.push_back(dir + to_string(i) + ".jpg");
	// }

	// for(int i=1;i<10;i++)
	// {
	// 	files.push_back(dir +"0000"+ to_string(i) + ".jpg");
	// }
	// for(int i=10;i<100;i++)
	// {
	// 	files.push_back(dir +"000"+ to_string(i) + ".jpg");
	// }
	// for(int i=100;i<1000;i++)
	// {
	// 	files.push_back(dir +"00"+ to_string(i) + ".jpg");
	// }
	// for(int i=1000;i<=GOOD_TRAIN_FILES;i++)
	// {
	// 	files.push_back(dir +"0"+ to_string(i) + ".jpg");
	// }
for(int i=0;i<GOOD_TRAIN_FILES_0704;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190711/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0711;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190715/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0715;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20200303/positive/";
for(int i=0;i<GOOD_TRAIN_FILES_0303;i++)
{
        files.push_back(dir + to_string(i) + ".jpg");
}


    //HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
 HOGDescriptor *hog = new HOGDescriptor(Size(48, 48), Size(8, 8), Size(4, 4), Size(4, 4), 9);
	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		//Mat train_data(64, 128, CV_32FC1);
		Mat train_data;
	
		//resize(SrcImage, train_data, Size(64, 128));
		resize(SrcImage, train_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(train_data, descriptor, Size(8, 8));		// 计算hog特征
		hog->compute(train_data, descriptor, Size(4, 4));	
		int feature_dim = descriptor.size();
		if (i == 0) {
			trainingFeatures = Mat::zeros(BAD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0711+BAD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+HARD_TRAIN_FILES+GOOD_TRAIN_FILES_0303+BAD_TRAIN_FILES_0303, feature_dim, CV_32FC1);
			trainingLabels = Mat::zeros(BAD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0711+BAD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+HARD_TRAIN_FILES+GOOD_TRAIN_FILES_0303+BAD_TRAIN_FILES_0303, 1, CV_32SC1);
		}
		float * featurePtr = trainingFeatures.ptr<float>(i);
		int * labelPtr = trainingLabels.ptr<int>(i);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		*labelPtr = 1;
		labelPtr++;
    }
}
void get_badtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
    string dir = "../../../data/20190704/negative_true/";
    vector<string> files;
	for(int i=0;i<BAD_TRAIN_FILES_0704;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190711/negative_true/";
for(int i=0;i<BAD_TRAIN_FILES_0711;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20190715/negative_true/";
for(int i=0;i<BAD_TRAIN_FILES_0715;i++)
{
	files.push_back(dir + to_string(i) + ".jpg");
}
dir="../../../data/20200303/negative/";
for(int i=0;i<BAD_TRAIN_FILES_0303;i++)
{
        files.push_back(dir + to_string(i) + ".jpg");
}

    // for (int i = 0; i < BAD_TRAIN_FILES; i++)								// 取前400张数字0来训练
	// {
	// 	files.push_back(dir + to_string(i) + ".jpg");
	// }
	// for(int i=1;i<10;i++)
	// {
	// 	files.push_back(dir +"0000"+ to_string(i) + ".jpg");
	// }
	// for(int i=10;i<100;i++)
	// {
	// 	files.push_back(dir +"000"+ to_string(i) + ".jpg");
	// }
	// for(int i=100;i<1000;i++)
	// {
	// 	files.push_back(dir +"00"+ to_string(i) + ".jpg");
	// }
	// for(int i=1000;i<10000;i++)
	// {
	// 	files.push_back(dir +"0"+ to_string(i) + ".jpg");
	// }
	// for(int i=10000;i<=BAD_TRAIN_FILES;i++)
	// {
	// 	files.push_back(dir + to_string(i) + ".jpg");
	// }
   // HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
 HOGDescriptor *hog = new HOGDescriptor(Size(48, 48), Size(8, 8), Size(4, 4), Size(4, 4), 9);
	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		//Mat train_data(64, 128, CV_32FC1);
		Mat train_data;
		//resize(SrcImage, train_data, Size(48, 128));
		resize(SrcImage, train_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(train_data, descriptor, Size(8, 8));		// 计算hog特征
		hog->compute(train_data, descriptor, Size(4, 4));	
		int feature_dim = descriptor.size();
		float * featurePtr = trainingFeatures.ptr<float>(i+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0303);
		int * labelPtr = trainingLabels.ptr<int>(i+GOOD_TRAIN_FILES_0704+GOOD_TRAIN_FILES_0711+GOOD_TRAIN_FILES_0715+GOOD_TRAIN_FILES_0303);
		for (int j = 0; j < feature_dim; j++)
         {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		*labelPtr = -1;
		labelPtr++;
    }
}
void get_hardtrain(Mat& trainingFeatures, Mat& trainingLabels)
{
    string dir = "../../../data/train/harddata/";
    vector<string> files;
    // for (int i = 0; i < BAD_TRAIN_FILES; i++)								// 取前400张数字0来训练
	// {
	// 	files.push_back(dir + to_string(i) + ".jpg");
	// }
	for(int i=1;i<10;i++)
	{
		files.push_back(dir +"0000"+ to_string(i) + ".jpg");
	}
	for(int i=10;i<=HARD_TRAIN_FILES;i++)
	{
		files.push_back(dir +"000"+ to_string(i) + ".jpg");
	}
	
    //HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
  HOGDescriptor *hog = new HOGDescriptor(Size(48, 48), Size(8, 8), Size(4, 4), Size(4, 4), 9);
	for (int i = 0; i < files.size(); i++)
	{
		Mat SrcImage = imread(files[i].c_str());
		//Mat train_data(64, 128, CV_32FC1);
		Mat train_data;
		//resize(SrcImage, train_data, Size(64, 128));
		resize(SrcImage, train_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(train_data, descriptor, Size(8, 8));		// 计算hog特征
		hog->compute(train_data, descriptor, Size(4, 4));
		int feature_dim = descriptor.size();
		float * featurePtr = trainingFeatures.ptr<float>(i+GOOD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0704);
		int * labelPtr = trainingLabels.ptr<int>(i+GOOD_TRAIN_FILES_0704+BAD_TRAIN_FILES_0704);
		for (int j = 0; j < feature_dim; j++)
         {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		*labelPtr = -1;
		labelPtr++;
    }
}
int main()
{
	struct timeval start,end;
    //获取训练数据
	string model_path = "SVC_RBF_1000_4848_all.xml";
	string model_save="SVC_RBF_1000_4848_all.xml";
#ifdef TRAIN
    Mat trainingFeatures, trainingLabels;
	cout<<"加载训练图"<<endl;
	get_goodtrain(trainingFeatures, trainingLabels);
	cout<<"正样本加载完成"<<endl;
    get_badtrain(trainingFeatures, trainingLabels);
	cout<<"负样本加载完成"<<endl;
	//get_hardtrain(trainingFeatures, trainingLabels);
	//cout<<"难样本加载完成"<<endl;
    Ptr<TrainData> data = TrainData::create(trainingFeatures, ROW_SAMPLE, trainingLabels);

    //设置SVM训练器参数并训练
	Ptr<SVM> model = SVM::create();
	model->setDegree(0);
	model->setGamma(1);
	model->setCoef0(1.0);
	model->setC(10);
	model->setNu(0.5);
	model ->setP(1.0);
	model->setType(SVM::C_SVC);														// svm类型
	model->setKernel(SVM::LINEAR);														// kernel类型，核函数，线性核
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,1000, FLT_EPSILON));
model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS , 1000, FLT_EPSILON));
//model->setTermCriteria(TermCriteria(TermCriteria::EPS, 200, FLT_EPSILON));
    cout<<"开始训练!"<<endl;
    
//    model->trainAuto(data);	
    //保存模型

//	model->save(model_path);

	cout << "训练完毕！开始测试：" << endl;


    model->load(model_path);
model->trainAuto(data);
model->save(model_save);
#endif
#ifdef UNTRAIN
    Ptr<SVM>model=SVM::load(model_save);//直接获取数据进行检测
#endif

	cout<<"开始检测"<<endl;
    int count0 = 0;//成功检测到人的个数
	string dir = "./../../../data/20200303/positive/";
	//HOGDescriptor *hog = new HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
        HOGDescriptor *hog = new HOGDescriptor(Size(48, 48), Size(8, 8), Size(4, 4), Size(4, 4), 9);
        
	for (int i =19000; i < 24700; i++)								// 取后100张数字0来测试
	{
		Mat testData, testFeatures;
		Mat test = imread(dir + to_string(i) + ".jpg");
		//Mat test_data(64, 128, CV_32FC1);
		Mat test_data;
		//resize(test, test_data, Size(64, 128));
		resize(test, test_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(test_data, descriptor, Size(8, 8));
		hog->compute(test_data, descriptor, Size(4, 4));
		int feature_dim = descriptor.size();
		testFeatures = Mat::zeros(1, feature_dim, CV_32FC1);
		float * featurePtr = testFeatures.ptr<float>(0);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		int k = model->predict(testFeatures);
		if (k == 1)
			count0++;
	}

    int count1 = 0;//成功检测到不是人的个数
    dir = "./../../../data/20200303/negative/";
	gettimeofday(&start,NULL);
	for (int i =39000; i < 49000; i++)								// 取后100张数字0来测试
	{
		Mat testData, testFeatures;
		Mat test = imread(dir + to_string(i) + ".jpg");
		//Mat test_data(64, 128, CV_32FC1);
		Mat test_data;
		//resize(test, test_data, Size(64, 128));
		resize(test, test_data, Size(48, 48));
		vector<float>descriptor;								// hog特征描述子
		//hog->compute(test_data, descriptor, Size(8, 8));
		hog->compute(test_data, descriptor, Size(4, 4));
		int feature_dim = descriptor.size();
		testFeatures = Mat::zeros(1, feature_dim, CV_32FC1);
		float * featurePtr = testFeatures.ptr<float>(0);
		for (int j = 0; j < feature_dim; j++) {
			*featurePtr = descriptor[j];
			featurePtr++;
		}
		int k = model->predict(testFeatures);
		if (k == -1)
			count1++;
	}
	gettimeofday(&end,NULL);
	long timeuse=1000000*(end.tv_sec-start.tv_sec) + end.tv_usec -start.tv_usec;
	cout<<"用时："<<timeuse<<"微秒"<<endl;
double b1=double(count0)/double(5700);
double b2=double(count1)/double(10000);
cout<<count0<<"/5700"<<endl;
    cout << "检测正确比例：" << b1 << endl;
cout<<count1<<"/10000"<<endl;
		cout << "检测非人正确比例：" << b2<< endl;
	cout << "测试完毕." << endl;
	getchar();
	return 0;
}
