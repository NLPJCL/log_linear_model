#include<iostream>
#include"dataset.h"
#include"log_linear_model.h"
using namespace std;
int main()
{
	log_linear_model b;
	b.create_feature_space();
	b.SGD_online_training();
	system("pause");
	return 0;
}