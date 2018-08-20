#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//自然对数的指数。
#include"dataset.h"
using namespace std;
class log_linear_model
{
public:
	void create_feature_space();
	void SGD_online_training();
	log_linear_model();
	//存储。
	void save_file(int i);
	~log_linear_model();
private:
	//基础数据集。
	dataset train;
	dataset dev;
	map<string, int> model;//特征空间。
	map<string, int> tag;//词性
	vector<double> w;
	map<int, double> g;
	vector<string> value;
	//创建特征空间。
	vector<string> create_feature(sentence sentence, int pos);
	//在线算法
	void update_w(int eta);
	void update_g(sentence sen, int pos, string tag);
	string maxscore_tag(sentence sen, int pos);
	vector<int> get_id(vector<string> f);
	int count_score(int offset, vector<int> fv);
	double max(double x1, double x2);
	//评价。
	double evaluate(dataset);
};

#pragma once
