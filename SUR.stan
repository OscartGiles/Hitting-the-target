data {

	int<lower=1> K;  //Number outcomes
	int<lower=1> J;  //Number predictors
	int<lower=0> N;  //Number data points
	vector[J] x[N];
	vector[K] y[N];

	real<lower = 0> corr_prior;

}

parameters {

	matrix[K, J] beta;
	cholesky_factor_corr[K] L_Omega;
	vector<lower=0>[K] sigma;

}

model {
	vector[K] mu[N];
	matrix[K, K] L_Sigma;

	for (n in 1:N)
		mu[n] = beta * x[n];

	L_Sigma = diag_pre_multiply(sigma, L_Omega);

	to_vector(beta) ~ normal(0, 0.5);

	L_Omega ~ lkj_corr_cholesky(corr_prior);

	sigma ~ cauchy(0, 2.5);

	y ~ multi_normal_cholesky(mu, L_Sigma);
}

