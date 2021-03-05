#define _CRT_SECURE_NO_WARNINGS
#define DEBUG_PRINT
//#undef DEBUG_PRINT
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_real.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>
#include "cmath"
#include <random>

//define constants:
constexpr double kB = 1, tau = 1, D = 1200, g = 7, Ea = 500, k = 0.1;

constexpr int N = 1000;
double x[N], r[N];
int spin[N];		// necesar? avem raza care e strict corelata cu spinul
FILE *fp;

const gsl_multimin_fdfminimizer_type *Typ = gsl_multimin_fdfminimizer_conjugate_fr;
gsl_multimin_fdfminimizer *s;
gsl_vector *poz;
gsl_multimin_function_fdf minex_func;

struct parameters {
	int N;
	double bond_len, elastic_k, A, r_diff;
};

double prob_HtoL(double T, double p)
{
	return (1 / tau) * exp((D - kB * T * g) / (2 * kB * T)) * exp(-(Ea - k * p) / (kB * T));
}

double prob_LtoH(double T, double p)
{
	return (1 / tau) * exp(-(D - kB * T * g) / (2 * kB * T)) * exp(-(Ea + k * p) / (kB * T));
}

//double V(float A, int N, double x[], float r_diff) {
  //  double  factor = (0.5)/r_diff, V_tot = 0;
	//for(int i = 0; i < N; i++)
	  //  V_tot = V_tot - A*abs(sin(factor*(x[i] + r_diff)*M_PI));
	//return V_tot;
//}
//double bond_energy(int N, double bond_len, double x[], double r[], double elastic_k){
//    double E = 0;
  //  for(int i = 0; i< N-1; i++){
	//    double xi = x[i], ri = r[i], x_next  = x[i+1], r_next = r[i+1], e_bond;
	  //  e_bond = (elastic_k/2.0)*pow((bond_len - (x_next - xi - ri - r_next)), 2);
	//    E += e_bond;
  //  
 //   return E;
//}

double total_energy(const gsl_vector q[], void *para) {
	parameters *params = (parameters *)para;
	double  bond_len = params->bond_len, elastic_k = params->elastic_k, A = params->A, r_diff = params->r_diff;
	int N = params->N;
	double  factor = (0.5) / r_diff, V_tot = 0;
	double E = 0, xi, ri, x_next, r_next, e_bond;
	for (int i = 0; i < N - 1; i++) {
		xi = gsl_vector_get(q, i);
		ri = r[i];
		x_next = gsl_vector_get(q, i + 1);
		r_next = r[i + 1];

		e_bond = (elastic_k / 2.0) * pow((bond_len - (x_next - xi - ri - r_next)), 2);
		E += e_bond;

		V_tot = V_tot - A * abs(sin(factor * (xi + r_diff) * M_PI));
	}

	V_tot = V_tot - A * abs(sin(factor * (gsl_vector_get(q, N - 1) + r_diff) * M_PI));

	return E + V_tot;
}

void d_total_energy(const gsl_vector q[], void *para, gsl_vector *d_energy)
{
	parameters *params = (parameters *)para;
	int N = params->N;
	double bond_len = params->bond_len, elastic_k = params->elastic_k, A = params->A, r_diff = params->r_diff;
	
	double  factor = (0.5) / r_diff, V = 0;
	double xi, ri, x_next, r_next, x_prev, r_prev, e_bond = 0, de_dxi;

	xi = gsl_vector_get(q, 0);
	ri = r[0];
	x_next = gsl_vector_get(q, 1);
	r_next = r[1];
	V = (-A * factor * M_PI * cos(factor * (xi + r_diff) * M_PI));
	e_bond = elastic_k * (bond_len - x_next + xi + ri + r_next);
	de_dxi = V + e_bond;
	gsl_vector_set(d_energy, 0, de_dxi);

	for (int i = 1; i < N - 1; i++) {
		x_prev = xi;
		xi = x_next;
		x_next = gsl_vector_get(q, i + 1);
		r_prev = ri;
		ri = r_next;
		r_next = r[i + 1];

		V = -A * factor * M_PI * cos(factor * (xi + r_diff) * M_PI);
		e_bond = elastic_k * (-x_next - x_prev + 2 * xi + -r_prev + r_next);
		de_dxi = V + e_bond;
		gsl_vector_set(d_energy, i, de_dxi);
	}
	//last particle in the array:
	x_prev = xi;
	xi = x_next;
	r_prev = ri;
	ri = r_next;
	V = (-A * factor * M_PI * cos(factor * (xi + r_diff) * M_PI));
	e_bond = elastic_k * (-bond_len - x_prev + xi - ri - r_prev);
	de_dxi = V + e_bond;
	gsl_vector_set(d_energy, N - 1, de_dxi);
}

void fdfn1(const gsl_vector q[], void *para, double *f, gsl_vector *df)
{
	*f = total_energy(q, para);
	d_total_energy(q, para, df);
}

void minimize_energy(double A, double bond_length, double r_low, double r_high, double elastic_k)
{
	size_t iter = 0;
	int status;
	//double size;

	/* Starting point */
	poz = gsl_vector_alloc(N);
	for (int i = 0; i < N; i++)
	{
		gsl_vector_set(poz, i, x[i]);
	}

	parameters params;
	params.N = N;
	params.A = A;
	params.bond_len = bond_length;
	params.elastic_k = elastic_k;
	params.r_diff = r_high - r_low;

	/* Initialize method and iterate */
	minex_func.n = N;
	minex_func.f = total_energy;
	minex_func.df = d_total_energy;
	minex_func.fdf = fdfn1;
	minex_func.params = &params;

	s = gsl_multimin_fdfminimizer_alloc(Typ, N);
	//status = gsl_multimin_fdfminimizer_restart(s);
	gsl_multimin_fdfminimizer_set(s, &minex_func, poz, 0.01, 1.0e-4);

	do
	{
		iter++;

		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient(s->gradient, 1.0e-3);

	} while (status == GSL_CONTINUE);

	for (int i = 0; i < N; i++)
	{
		x[i] = gsl_vector_get(s->x, i);
		gsl_vector_set(poz, i, x[i]);
	}
#ifdef DEBUG_PRINT
	printf("minimized!!!!! \n");
#endif
}

int change_spin(int i, double p_HtoL, double p_LtoH, double R_low, double R_high) {

	double rand_prob;
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> random_para(0, 1);
	rand_prob = random_para(mt);
#ifdef DEBUG_PRINT
	printf("%d \t %lf \t %lf \t %lf \t L_to_H  %lf \t L_to_H  %lf \n", i, x[i], r[i], rand_prob, p_LtoH, p_HtoL);
#endif

	int changed = 0;
	if (spin[i] == 1 && rand_prob < p_HtoL) {
		spin[i] = 0;
		r[i] = R_low;
		changed = 1;
#ifdef DEBUG_PRINT
		printf("Changed high to low \n");
#endif
	}

	if (changed == 0 && spin[i] == 0 && rand_prob < p_LtoH) {
		spin[i] = 1;
		r[i] = R_high;
		changed = 1;
#ifdef DEBUG_PRINT
		printf("Changed low to high \n");
#endif
	}

	return changed;
}

double elastic_force(int i, double elastic_k, double bond_len) {
	double xi = x[i], ri = r[i], elastic_f = 0, prev_particle_x, prev_particle_r, next_particle_x, next_particle_r;
	if (i > 0) {
		prev_particle_x = x[i - 1];
		prev_particle_r = r[i - 1];
		elastic_f += elastic_k * (bond_len - (xi - prev_particle_x - ri - prev_particle_r));
	}

	if (i < N - 1) {
		next_particle_x = x[i + 1];
		next_particle_r = r[i + 1];
		elastic_f += elastic_k * (bond_len - (next_particle_x - xi - ri - next_particle_r));
	}

	return elastic_f;
}

void create_system(int number_of_particles, int starting_spin, double r_low, double r_high, double bond_length) {

	double current_x = 0.4, starting_radius;

	if (starting_spin == 1)
		starting_radius = r_high;
	else
		starting_radius = r_low;

	for (int i = 0; i < number_of_particles; i++) {

		x[i] = current_x;
		r[i] = starting_radius;
		spin[i] = starting_spin;
		current_x += (bond_length + 2 * starting_radius);
	}
}

void savesys(void)
{
	fp = fopen("rez.dat", "a");
	for (int i = 0; i < N; i++)
	{
		fprintf(fp, "%5.4lf ", x[i]);
	}
	fprintf(fp, "\n");
	for (int i = 0; i < N; i++)
	{
		fprintf(fp, "%d ", spin[i]);
	}
	fprintf(fp, "\n");
	for (int i = 0; i < N; i++)
	{
		fprintf(fp, "%3.2lf ", r[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
}

int main()
{
	// starting spin, r_low, r_high, bond_length, elastic_k for bond, A(mplitude of potential well):
	int starting_spin = 0, chosen_particle = 0, ch;
	double r_low = 0.25, r_high = 0.30, bond_length = 0.4, elastic_k = 1.0, A = 0.001;

	// for Monte-Carlo steps:
	double T0 = 50, Tf = 300, T_step = 1, n_steps = N, HS, LS, n_HS;
	// for calculating probs and elastic pressure:
	double p_HtoL, p_LtoH, elastic_p;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(0, N - 1);

	parameters params;
	params.N = N;
	params.A = A;
	params.bond_len = bond_length;
	params.elastic_k = elastic_k;
	params.r_diff = r_high - r_low;

	minex_func.n = N;
	minex_func.f = total_energy;
	minex_func.df = d_total_energy;
	minex_func.fdf = fdfn1;
	minex_func.params = &params;

	s = gsl_multimin_fdfminimizer_alloc(Typ, N);

	create_system(N, starting_spin, r_low, r_high, bond_length);
	minimize_energy(A, bond_length, r_low, r_high, elastic_k);
	savesys();

	//for (double temp = T0; temp <= Tf; temp = temp + T_step)
	double temp = T0;
	{
		for (int step = 1; step <= n_steps; step++)
		{
			// printf("%d \n", dist(mt));
			chosen_particle = dist(mt);

			elastic_p = elastic_force(chosen_particle, elastic_k, bond_length);
			p_HtoL = prob_HtoL(temp, elastic_p);
			p_LtoH = prob_LtoH(temp, elastic_p);

			//ch = change_spin(chosen_particle, p_HtoL, p_LtoH, r_low, r_high);
			ch = 1;
			r[500] = r_high;

			if (ch == 1)
			{
				minimize_energy(A, bond_length, r_low, r_high, elastic_k);
				savesys();
				break;
			}
		}

		HS = 0, LS = 0;
		for (int j = 0; j < N; j++)
		{
			if (spin[j] == 0)
				LS++;
			else
				HS++;
		}

		n_HS = HS / N;
	}

	return 0;
}

