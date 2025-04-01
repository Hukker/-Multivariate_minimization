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
	double y1 = cos(X[0]);
	double g = -pow(X[0], 2) - pow(X[1], 2);
	double y2 = -exp(g) - y1;
	return y2;

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
				std::cout << "Converged after " << iter << " iterations" << std::endl;
				std::cout << "Calculated optimal point is (";
				for (const auto& el : X) {
					std::cout << el << " ";
				}
				std::cout << ")" << "and function value is "<< std::setprecision(4) <<func_value << std::endl;
				return { X, func_value };
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
			std::cout << "Reached maximum iterations (" << max_iter << ")" << std::endl;
			exit(BREAK);
		}

		/*std::cout << "Optimal point: (";
		for (size_t i = 0; i < X.size(); i++) {
			std::cout << X[i];
			if (i != X.size() - 1) std::cout << ", ";
		}
		std::cout << ")" << std::endl;

		std::cout << "Function value at optimal point: " << func_value << std::endl;

		return { X, func_value };*/
	}


	//метод ньютона
	pair<vector<double>, double> newthon(const vector<double>& X_, double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X = X_;
		double func_value = func(X);
		int iter = 0;

		std::cout << "Starting Newton's method..." << std::endl;
		std::cout << "Initial point: (" << X[0] << ", " << X[1] << ")" << std::endl;
		std::cout << "Initial function value: " << func_value << std::endl << std::endl;

		while (iter < max_iter)
		{
			// Вычисляем градиент
			auto grad = gradient(X, delta);
			double grad_norm = norm2(grad);

			// Выводим информацию каждые 10 итераций
			if (iter % 10 == 0) {
				std::cout << "Iteration " << iter
					<< ": f(x) = " << func_value
					<< ", ||grad|| = " << grad_norm << std::endl;
			}

			
			if (grad_norm < tol) {
				std::cout << "\nConverged after " << iter << " iterations" << std::endl;
				break;
			}

			vector<vector<double>> hessian(2, vector<double>(2));

			hessian[0][0] = (partial_derivative(X, 0, delta) -
				partial_derivative(substraction(X, { delta, 0 }), 0, delta)) / delta;

			hessian[0][1] = hessian[1][0] =
				(partial_derivative(X, 1, delta) -
					partial_derivative(substraction(X, { delta, 0 }), 1, delta)) / delta;

			
			hessian[1][1] =
				(partial_derivative(X, 1, delta) -
					partial_derivative(substraction(X, { 0, delta }), 1, delta)) / delta;

			// Решаем систему H*d = -g (методом Крамера для 2x2)
			double det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];

			if (fabs(det) < 1e-10) {
				std::cerr << "Hessian is singular, switching to gradient descent" << std::endl;
				vector<double> step = num_on_vec(0.01, grad); // маленький шаг по градиенту
				X = substraction(X, step);
			}
			else {
				// Вычисляем d = H^-1*(-g)
				vector<double> d(2);
				d[0] = (-grad[0] * hessian[1][1] + grad[1] * hessian[0][1]) / det;
				d[1] = (grad[0] * hessian[1][0] - grad[1] * hessian[0][0]) / det;

				X[0] += d[0];
				X[1] += d[1];
			}

			double new_func_value = func(X);

			if (new_func_value > func_value) {
				std::cout << "Function increased, reducing step" << std::endl;
				X = substraction(X, num_on_vec(0.5, grad)); 
				new_func_value = func(X);
			}

			func_value = new_func_value;
			iter++;
		}

		if (iter == max_iter) {
			std::cout << "\nReached maximum iterations (" << max_iter << ")" << std::endl;
		}

		// Выводим результаты
		std::cout << "\nOptimization results:" << std::endl;
		std::cout << std::fixed << std::setprecision(8);
		std::cout << "Optimal point: (" << X[0] << ", " << X[1] << ")" << std::endl;
		std::cout << "Function value: " << func_value << std::endl;

		return { X, func_value };
	}


	//метод хука дживса
	pair<vector<double>, double> hook_jeeves(const vector<double>& X_, double alpha = 1, double tol = 0.1, double delta = 0.00001, int max_iter = 10000)
	{
		vector<double> X = X_;
		vector<double> X_base = X;
		vector<double> delta_step(X.size(), alpha); // Шаги по каждой координате
		double func_value = func(X);
		int iter = 0;
		bool improved;

		std::cout << "Starting Hooke-Jeeves method..." << std::endl;
		std::cout << "Initial point: (" << X[0] << ", " << X[1] << ")" << std::endl;
		std::cout << "Initial function value: " << func_value << std::endl << std::endl;

		while (iter < max_iter)
		{
			improved = false;

			// 1. Исследующий поиск (Exploratory move)
			for (size_t i = 0; i < X.size(); ++i)
			{
				// Пробуем шаг в положительном направлении
				vector<double> X_plus = X_base;
				X_plus[i] += delta_step[i];
				double f_plus = func(X_plus);

				// Пробуем шаг в отрицательном направлении
				vector<double> X_minus = X_base;
				X_minus[i] -= delta_step[i];
				double f_minus = func(X_minus);

				// Выбираем наилучшее направление
				if (f_plus < func_value)
				{
					X_base = X_plus;
					func_value = f_plus;
					improved = true;
				}
				else if (f_minus < func_value)
				{
					X_base = X_minus;
					func_value = f_minus;
					improved = true;
				}
			}

			// 2. Проверка на улучшение
			if (improved)
			{
				// 3. Поиск по образцу (Pattern move)
				vector<double> X_pattern = X_base;
				for (size_t i = 0; i < X.size(); ++i)
				{
					X_pattern[i] += (X_base[i] - X[i]);
				}

				// Проверяем новую точку
				double f_pattern = func(X_pattern);
				if (f_pattern < func_value)
				{
					X = X_base;
					X_base = X_pattern;
					func_value = f_pattern;
				}
				else
				{
					X = X_base;
				}
			}
			else
			{
				// 4. Уменьшаем шаг, если улучшения нет
				for (size_t i = 0; i < delta_step.size(); ++i)
				{
					delta_step[i] *= 0.5;
				}
			}

			// Выводим информацию каждые 100 итераций
			/*if (iter % 100 == 0)
			{
				std::cout << "Iteration " << iter << ": f(x) = " << func_value
					<< ", step size = " << delta_step[0] << std::endl;
			}*/

			// Проверка условия остановки
			bool stop = true;
			for (const auto& step : delta_step)
			{
				if (step > tol)
				{
					stop = false;
					break;
				}
			}
			if (stop)
			{
				std::cout << "\nConverged after " << iter << " iterations" << std::endl;
				break;
			}

			iter++;
		}

		if (iter == max_iter)
		{
			std::cout << "\nReached maximum iterations (" << max_iter << ")" << std::endl;
		}

		// Выводим результаты
		std::cout << "\nOptimization results:" << std::endl;
		std::cout << std::fixed << std::setprecision(8);
		std::cout << "Optimal point: (" << X_base[0] << ", " << X_base[1] << ")" << std::endl;
		std::cout << "Function value: " << func_value << std::endl;

		return { X_base, func_value };
	}

};


int main()
{
	vector<double> x{ 0.4, 0.9 };
	double r = 1;
	PartitionGradient p(x, r);
	p.check_nh();
	
	/*vector<double> grad = p.gradient(x);
	for (auto el : grad) {
		std::cout << el << "\n";
	}*/

	//auto res = p.gradient_method(x);
	//std::cout << res.second;

	//auto res1 = p.newthon(x);  // Вызов метода Ньютона
	//std::cout << "Final result: " << res1.second << std::endl;

	auto res = p.hook_jeeves(x, 0.5, 0.001); // alpha=0.5, tol=0.001
	std::cout << "Final result: " << res.second << std::endl;

	return 0;
}
