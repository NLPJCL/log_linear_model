#pragma once
#include<iostream>
#include<map>
#include<vector>
#include<cmath>//��Ȼ������ָ����
#include"dataset.h"
using namespace std;
class log_linear_model
{
public:
	void create_feature_space();
	void SGD_online_training();
	log_linear_model();
	//�洢��
	void save_file(int i);
	~log_linear_model();
private:
	//�������ݼ���
	dataset train;
	dataset dev;
	map<string, int> model;//�����ռ䡣
	map<string, int> tag;//����
	vector<double> w;
	map<int, double> g;
	vector<string> value;
	//���������ռ䡣
	vector<string> create_feature(sentence sentence, int pos);
	//�����㷨
	void update_w(int eta);
	void update_g(sentence sen, int pos, string tag);
	string maxscore_tag(sentence sen, int pos);
	vector<int> get_id(vector<string> f);
	int count_score(int offset, vector<int> fv);
	double max(double x1, double x2);
	//���ۡ�
	double evaluate(dataset);
};

#pragma once
