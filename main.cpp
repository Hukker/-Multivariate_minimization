#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <memory>
#include <iterator>
#include <utility>
#include <cstdio>
#include <iomanip>

using std::vector;
using std::pair;

#define ERROR -1
#define BREAK -2

//-e^(-x^2-y^2)-cos(x)
double func(const vector<double>& X)
{
	//double y1 = cos(X[0]);
	double g = -pow(X[0], 2) - pow(X[1], 2);
	double y2 = -exp(g);
	return y2 + 1;

}


class PartitionGradient
{
	//стартовая точка
	vector<double> X{ 0.3,0.4 };

	double r = 1;

	//точка известного оптимума
	vector<double> X_opt{ 0,0 };

public:
	PartitionGradient(const vector<double> &X_, double r_) :
		X(X_), r(r_) {}

	

	//Проверка находится ли точка в окрестности
	void check_nh()
	{
		size_t n = X.size();
		double sum = 0;

		for (int i = 0; i < n; i++) {
			sum += pow(X[i] - X_opt[i], 2);
		}
		double norm = sqrt(sum);

		if (norm < this->r) {
			std::cout << "point is correct" << std::endl;
		}
		else {
			std::cerr << "Point is out of neighborhood" << std::endl;
			exit(ERROR);
		}
	}

	//вычисление частной производной j-й компоненты
	double partial_derivative(const vector<double> &X_, int j, double delta_x = 0.00001)
	{

		if (j > X.size()-1) {
			std::cerr << "j'th component out of vector's size" << std::endl;
			return 1;
		}
		vector<double> X(X_);
		X[j] += delta_x;
		double y1 = func(X);
		double y2 = func(X_);
		double delta_y = y1 - y2;
		return delta_y / delta_x;
	}

	//вычисление градиента функции в точке 
	vector<double> gradient(const vector<double>& X_, double eps = 0.00001)
	{
		vector<double> grad;

		for (int i = 0; i < X.size(); i++) {
			double dy = partial_derivative(X_, i, eps);
			grad.push_back(dy);
		}

		return grad;
	}

	//вторая норма вектора
	double norm2(const vector<double> X_)
	{
		vector<double> X(X_);

		size_t n = X.size();
		double sum = 0;

		for (int i = 0; i < n; i++) {
			sum += pow(X[i], 2);
		}
		double norm = sqrt(sum);
		return norm;
	}


	//вектор на вектор скалярно
	double dot(const vector<double>& x_, const vector<double>& y_)
	{
		if (x_.size() != y_.size()) {
			std::cout << "dim(x) != dim(y)" << std::endl;
		}
		else {
			double sum = 0;
			for (int i = 0; i < x_.size(); i++){
				sum += x_[i] * y_[i];

			}
			return sum;
		}
	}

	//число на вектор
	vector<double> num_on_vec(double num, const vector<double>& x_)
	{
		vector<double> X(x_);
		size_t n = x_.size();
		for (int i = 0; i < n; i++) {
			X[i] *= num;
		}
		return X;
	}

	//вычестание векторов
	vector<double> substraction(const vector<double>& x_, const vector<double>& y_)
	{
		vector<double> sub;
		for (int i = 0; i < x_.size(); i++) {
			sub.push_back(x_[i] - y_[i]);
		}
		return sub;
	}

	//градиентный метод первого порядка
	pair<vector<double>, double> gradient_method(const vector<double>& X_,double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		//int num = std::static_cast<int> (1.0/tol);

		vector<double> X(X_);
		vector<double> grad;
		double func_value = func(X);
		int iter = 0;

		while (iter < max_iter)
		{
			grad = gradient(X, delta);

			if (norm2(grad) < tol) {
				
				return { X, iter };
			}

			X = substraction(X, num_on_vec(alpha, grad));

			double new_func_value = func(X);

			while (new_func_value > func_value && alpha > 1e-3) {
				alpha *= 0.5;  
				for (int i = 0; i < X.size(); i++) {
					X[i] += alpha * grad[i];  
					X[i] -= alpha * grad[i]; 
				}
				new_func_value = func(X);
			}

			func_value = new_func_value;
			iter++;
		}

		if (iter >= max_iter) {
			exit(BREAK);
		}

		

		return { X, func_value };
	}


	//метод ньютона
	pair<vector<double>, int> newthon(const vector<double>& X_, double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X = X_;
		int iter = 0;

		while (iter < max_iter)
		{
			// Вычисляем градиент
			auto grad = gradient(X, delta);
			double grad_norm = norm2(grad);

			// Проверяем условие остановки
			if (grad_norm < tol) {
				break;
			}

			// Вычисляем гессиан
			vector<vector<double>> hessian(2, vector<double>(2));

			// Более точное вычисление вторых производных
			vector<double> X_plus_dx = X, X_minus_dx = X;
			X_plus_dx[0] += delta; X_minus_dx[0] -= delta;

			vector<double> X_plus_dy = X, X_minus_dy = X;
			X_plus_dy[1] += delta; X_minus_dy[1] -= delta;

			// ??f/?x?
			hessian[0][0] = (partial_derivative(X_plus_dx, 0, delta) -
				partial_derivative(X_minus_dx, 0, delta)) / (2 * delta);

			// ??f/?y?
			hessian[1][1] = (partial_derivative(X_plus_dy, 1, delta) -
				partial_derivative(X_minus_dy, 1, delta)) / (2 * delta);

			// ??f/?x?y
			vector<double> X_plus_dx_for_mixed = X; X_plus_dx_for_mixed[0] += delta;
			vector<double> X_minus_dx_for_mixed = X; X_minus_dx_for_mixed[0] -= delta;
			hessian[0][1] = (partial_derivative(X_plus_dx_for_mixed, 1, delta) -
				partial_derivative(X_minus_dx_for_mixed, 1, delta)) / (2 * delta);

			hessian[1][0] = hessian[0][1]; // Симметрия

			// Решаем систему H*d = -g
			double det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];

			//if (fabs(det) < 1e-10) {
			//	// Регуляризация гессиана
			//	hessian[0][0] += 0.1;
			//	hessian[1][1] += 0.1;
			//	det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
			//}

			// Вычисляем d = H??*(-g)
			vector<double> d(2);
			d[0] = (-grad[0] * hessian[1][1] + grad[1] * hessian[0][1]) / det;
			d[1] = (grad[0] * hessian[1][0] - grad[1] * hessian[0][0]) / det;

			// Делаем шаг с регулировкой размера
			double step_size = alpha;
			double current_val = func(X);
			vector<double> new_X = { X[0] + step_size * d[0], X[1] + step_size * d[1] };
			double new_val = func(new_X);

			// Поиск оптимального размера шага
			while (new_val > current_val && step_size > 1e-6) {
				step_size *= 0.5;
				new_X = { X[0] + step_size * d[0], X[1] + step_size * d[1] };
				new_val = func(new_X);
			}

			X = new_X;
			iter++;
		}

		return { X, iter };
	}


	//метод хука дживса
	pair<vector<double>, int> hook_jeeves(const vector<double>& X_,double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X = X_;
		vector<double> X_base = X;
		vector<double> delta_step(X.size(), alpha); // Шаги по каждой координате
		double func_value = func(X);
		int iter = 0;
		bool improved;

		while (iter < max_iter)
		{
			improved = false;
			vector<double> X_new = X_base;

			// 1. Исследующий поиск (Exploratory move)
			for (size_t i = 0; i < X.size(); ++i)
			{
				// Пробуем шаг в положительном направлении
				vector<double> X_plus = X_new;
				X_plus[i] += delta_step[i];
				double f_plus = func(X_plus);

				// Пробуем шаг в отрицательном направлении
				vector<double> X_minus = X_new;
				X_minus[i] -= delta_step[i];
				double f_minus = func(X_minus);

				// Выбираем наилучшее направление
				if (f_plus < func(X_new)) {
					X_new = X_plus;
				}
				if (f_minus < func(X_new)) {
					X_new = X_minus;
				}
			}

			// Проверяем, было ли улучшение
			double new_func_value = func(X_new);
			if (new_func_value < func_value - tol)
			{
				// 2. Поиск по образцу (Pattern move)
				vector<double> X_pattern(X_new.size());
				for (size_t i = 0; i < X_new.size(); ++i) {
					X_pattern[i] = X_new[i] + (X_new[i] - X_base[i]);
				}

				double f_pattern = func(X_pattern);

				// Если шаг по образцу успешен
				if (f_pattern < new_func_value) {
					X_base = X_new;
					X_new = X_pattern;
					new_func_value = f_pattern;
					improved = true;
				}
				else {
					improved = true;
				}

				X_base = X_new;
				func_value = new_func_value;
			}

			// 3. Уменьшаем шаг, если улучшения нет
			if (!improved)
			{
				for (size_t i = 0; i < delta_step.size(); ++i) {
					delta_step[i] *= 0.5;
				}

				// Проверка условия остановки
				if (norm2(delta_step) < tol) {
					break;
				}
			}

			iter++;
		}

		return { X_base, iter };
	}

	pair<vector<double>, int> gradient_newthon(const vector<double>& X_, double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X(X_);
		auto step1 = gradient_method(X, max_iter = 2);
		auto step2 = newthon(step1.first);

		return step2;
	}
	

	pair<vector<double>, int> jeeves_newthon(const vector<double>& X_, double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X(X_);
		auto step1 = hook_jeeves(X, max_iter = 2);
		auto step2 = newthon(step1.first);

		return step2;
	}

};


int main()
{
	vector<double> x{ 0.4, 0.9 };
	double r = 1;
	PartitionGradient p(x, r);
	p.check_nh();
	
	

	double alpha = 0.1;
	size_t n = x.size();


	auto res1 = p.gradient_method(x, alpha); //Вызов Градиентного метода
	std::cout << res1.second << std::endl;

	auto res2 = p.newthon(x, alpha);  // Вызов метода Ньютона
	std::cout << res2.second << std::endl;

	auto res3 = p.hook_jeeves(x, alpha); // Вызов метода Хука-Дживса
	std::cout  << res3.second << std::endl;
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << res1.first[i] << " ";
	}
	std::cout << "\n";
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << res2.first[i] << " ";
	}
	std::cout << "\n";
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << res3.first[i] << " ";
	}


	return 0;
}
