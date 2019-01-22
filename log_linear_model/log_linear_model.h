#pragma once
#include<iostream>
#include<unordered_map>
#include<vector>
#include<cmath>//自然对数的指数。
#include<map>
#include"dataset.h"
#include"windows.h"
#include"numeric"
using namespace std;
class log_linear_model
{
public:
	void create_feature_space();
	void SGD_online_training(bool shuffle, int iterator, int exitor);
	log_linear_model(const string &train_,const string &dev_,const string &test_);
	//存储。
	void save_file(int i);
	~log_linear_model();
private:
	//基础数据集。
	dataset train;
	dataset dev;
	dataset test;
	unordered_map<string, int> model;//特征空间。
	map<string, int> tag;//词性
	vector<string> vector_tag;
	vector<double> w;
	vector<string> value;
	unordered_map<int, double> g;
	//创建特征空间。
	vector<string> create_feature(const sentence &sentence, int pos);
	//在线算法
	void update_w(int eta);
	void update_g(sentence &sen, int pos, string &tag);
	string maxscore_tag(sentence &sen, int pos);
	vector<int> get_id(vector<string> &f);
	int count_score(int offset, vector<int> &fv);
//	double max(double x1, double x2);
	//评价。
	double evaluate(dataset&);
};

#pragma once
