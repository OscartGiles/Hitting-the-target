/**Ordered probit model with the simplex trick to get an ordered vector with fixed upper and lower bound.
Thanks to Bob Carpenter and Conor Goold for their help with the model
*/

data { 

  int<lower = 2> J;  //Number of outcome levels  
  int<lower = 1> Q; //Number of predictor variables (plus 1 for the intercept)  

  int<lower=0> N;  // data length  
  matrix[N, Q] X; // linear predictor

  int<lower=1,upper=J> y[N]; // response 
  
  
}
  
transformed data { 

  real L;
  real<lower = 1.5> U;
  real<lower = 0> diff; 
  int<lower = 1> K;

  K = J - 1; //Number of cutoff points 
  L = 1.5; //Lower cut off value
  U = J - 0.5; //Upper cut off value
  diff = U - L; 
}
  
parameters { 
  simplex[K - 1] cuts_raw; 
  
  vector[Q] beta;
  real<lower=0> sigma;
  
}

transformed parameters { 

  ordered[K] cuts; 
  cuts[1] = L; 
  cuts[2:K] = L + diff * cumulative_sum(cuts_raw); 
  
}

model{

    vector[J] theta;

    sigma ~ cauchy(0, 100);
    beta ~ normal(0, 500);

    for(n in 1:N) { 
        real eta;
        eta = X[n] * beta;
        theta[1] = normal_cdf( cuts[1] , eta, sigma);
        for (l in 2:K)
            theta[l] = normal_cdf(cuts[l], eta, sigma) - normal_cdf(cuts[l-1], eta , sigma);
    theta[J] = 1 - normal_cdf(cuts[K] , eta , sigma);
    y[n] ~ categorical(theta);
    }
}
