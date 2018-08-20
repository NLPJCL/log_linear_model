#include "log_linear_model.h"
//实例化特征
vector<string> log_linear_model::create_feature(sentence sentence, int pos)
{
	string word = sentence.word[pos];//当前词。
	string word_char_first = sentence.word_char[pos][0];//当前词的第一个元素。
	string word_char_last = sentence.word_char[pos][sentence.word_char[pos].size() - 1];//当前词的最后一个元素。
	string word_m1;
	string word_char_m1m1;
	string word_p1;
	string word_char_p1_first;
	int word_count = sentence.word.size();//当前句的总词数。
	if (pos == 0)
	{
		word_m1 = "$$";
		word_char_m1m1 = "$";
	}
	else
	{
		word_m1 = sentence.word[pos - 1];
		word_char_m1m1 = sentence.word_char[pos - 1][(sentence.word_char[pos - 1].size() - 1)];
	}
	if (pos == word_count - 1)
	{
		word_p1 = "##";
		word_char_p1_first = "#";
	}
	else
	{
		word_p1 = sentence.word[pos + 1];
		word_char_p1_first = sentence.word_char[pos + 1][0];
	}
	vector<string> f;
	f.push_back("02:*" + word);
	f.push_back("03:*" + word_m1);
	f.push_back("04:*" + word_p1);
	f.push_back("05:*" + word + "*" + word_char_m1m1);
	f.push_back("06:*" + word + "*" + word_char_p1_first);
	f.push_back("07:*" + word_char_first);
	f.push_back("08:*" + word_char_last);
	int pos_word_len = sentence.word_char[pos].size();
	for (int k = 0; k < pos_word_len - 1; k++)
	{
		string cik = sentence.word_char[pos][k];
		f.push_back("09:*" + cik);
		f.push_back("10:*" + word_char_first + "*" + cik);
		f.push_back("11:*" + word_char_last + "*" + cik);
		string cikp1 = sentence.word_char[pos][k + 1];
		if (cik == cikp1)
		{
			f.push_back("13:*" + cik + "*" + "consecutive");
		}
	}
	if (pos_word_len == 1)
	{
		f.push_back("12:*" + word + "*" + word_char_m1m1 + "*" + word_char_p1_first);
	}
	for (int k = 0; k <pos_word_len; k++)
	{
		if (k >= 4)break;
		string prefix, suffix;
		//获取前缀
		for (int n = 0; n <= k; n++)
		{
			prefix = prefix + sentence.word_char[pos][n];
		}
		//获取后缀。
		for (int n = pos_word_len - k - 1; n <= pos_word_len - 1; n++)
		{
			suffix = suffix + sentence.word_char[pos][n];
		}
		f.push_back("14:*" + prefix);
		f.push_back("15:*" + suffix);
	}
	return f;
}
//创建特征空间
void log_linear_model::create_feature_space()
{
	int count_feature = 0, count_tag = 0;
	for (auto z = train.sentences.begin(); z != train.sentences.end(); z++)
	{
		for (int i = 0; i < z->word.size(); i++)
		{
			vector <string> f;
			f = create_feature(*z, i);
			for (auto q = f.begin(); q != f.end(); q++)
			{
				if (model.find(*q) == model.end())//如果不在词性里面。
				{
					model[*q] = count_feature;
					value.push_back(*q);
					count_feature++;
				}
			}
			if (tag.find(z->tag[i]) == tag.end())
			{
				tag[z->tag[i]] = count_tag;
				count_tag++;
			}
		}
	}
	w.reserve(tag.size()*model.size());
	for (int i = 0; i < tag.size()*model.size(); i++)
	{
		w.push_back(0);
	}
	//cout << w.size() << endl;
	cout << "the total number of features is " << model.size() << endl;
	cout << "the total number of tags is " << tag.size() << endl;
}
vector<int> log_linear_model::get_id(vector<string> f)
{
	vector<int> fv;
	for (auto q = f.begin(); q != f.end(); q++)
	{
		auto t = model.find(*q);
		if (t != model.end())
		{
			fv.push_back(t->second);
		}
	}
	return fv;
}
int log_linear_model::count_score(int offset, vector<int> fv)
{
	double score = 0;
	for (auto z0 = fv.begin(); z0 != fv.end(); z0++)
	{
		score = score + w[offset + *z0];
	}
	return score;
}
double log_linear_model::max(double x1, double x2)
{
	if (x1 > x2)
	{
		return x1;
	}
	else
	{
		return x2;
	}
}
void log_linear_model::update_w(int eta)
{

	for (auto z = g.begin(); z != g.end(); z++)
	{
		w[z->first] += z->second;
	}
}
void log_linear_model::update_g(sentence sen, int pos, string correct_tag)
{
	vector<string> f = create_feature(sen, pos);
	vector<int> fv = get_id(f);
	//加上正确的词性。
	int offset = tag[correct_tag] * model.size();
	for (auto i = fv.begin(); i != fv.end(); i++)
	{
		int index = offset + *i;
		g[index]++;
	}
	//减去当前的概率。
	//求分母。
	double	denominator = 0.0;
	for (auto t = tag.begin(); t != tag.end(); t++)
	{
		int offset = t->second*model.size();
		int score = count_score(offset, fv);
		denominator += exp(score);
	}
	//求分式。	
	for (auto t0 = tag.begin(); t0 != tag.end(); t0++)
	{
		int offset = t0->second*model.size();
		int score = count_score(offset, fv);
		for (auto z = fv.begin(); z != fv.end(); z++)
		{
			int index = offset + *z;
			g[index] = g[index] - (exp(score) / denominator);
		}
	}
}
string log_linear_model::maxscore_tag(sentence  sen, int pos)
{
	double max_num = -1e10, score;
	string max_tag;
	vector<string> f = create_feature(sen, pos);
	vector<int> fv = get_id(f);
	for (auto z = tag.begin(); z != tag.end(); z++)//遍历词性
	{
		int offset = z->second*model.size();
		score = count_score(offset, fv);
		if (score > max_num + 1e-10)
		{
			max_num = score;
			max_tag = z->first;
		}
	}
	return max_tag;
}
double log_linear_model::evaluate(dataset data)
{
	int c = 0, total = 0;
	for (auto z = data.sentences.begin(); z != data.sentences.end(); z++)
	{
		for (int z0 = 0; z0 < z->word.size(); z0++)
		{
			total++;
			string max_tag = maxscore_tag(*z, z0);
			string correct_tag = z->tag[z0];
			if (max_tag == correct_tag)
			{
				c++;
			}
		}
	}
	return (c / double(total));
}
void log_linear_model::SGD_online_training()
{
	double max_train_precision = 0;
	double max_dev_precision = 0;
	int B = 50, b = 0, eta = 0.01;//初试步长。
	for (int i = 0; i < 20; i++)
	{
		cout << "iterator " << i << endl;
		for (auto sen = train.sentences.begin(); sen != train.sentences.end(); sen++)
		{
			for (int pos = 0; pos < sen->word.size(); pos++)
			{
				update_g(*sen, pos, sen->tag[pos]);
				b = b + 1;
				if (B == b)
				{
					update_w(eta);
					eta = max(0.999*eta, 0.00001);
					b = 0;
					g.clear();
				}
			}
		}
		if (b != 0)
		{
			update_w(eta);
			b = 0;
			g.clear();
		}
		save_file(i);
		cout << w.size() << endl;
		double train_precision = evaluate(train);
		cout << train.name << "=" << train_precision << endl;
		double dev_precision = evaluate(dev);
		cout << dev.name << "=" << dev_precision << endl;
		if (train_precision > max_train_precision)
		{
			max_train_precision = train_precision;
		}
		if (dev_precision > max_train_precision)
		{
			max_dev_precision = dev_precision;
		}
	}
	cout << train.name << "=" << max_train_precision << endl;
	cout << dev.name << "=" << max_dev_precision << endl;
}

log_linear_model::log_linear_model()
{
	train.read_data("train");
	dev.read_data("dev");
}


void log_linear_model::save_file(int i)
{
	string file_name = "feature" + to_string(i);
	ofstream feature_file(file_name);
	if (!feature_file)
	{
		cout << "don't open feature file" << endl;
	}
	for (auto z = value.begin(); z != value.end(); z++)
	{
		int i = (*z).find(":");
		string left = (*z).substr(0, i);
		string right = (*z).substr(i + 1);
		int f = model[*z];
		for (auto z = tag.begin(); z != tag.end(); z++)
		{
			int offset = model.size()*z->second;
			int index = offset + f;
			if (w[index] != 0)
			{
				string feature = left + z->first + right + "\t" + to_string(w[index]);
				feature_file << feature << endl;
			}
		}
	}

}

log_linear_model::~log_linear_model()
{
}
