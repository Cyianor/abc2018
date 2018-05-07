p <- c(qnorm(1.61 / 2), qnorm((0.1 + 1) / 2), -0.05, log(0.41))

chol_inv <- function(X) {
  U <- chol(X)
  chol2inv(U)
}

gen_sample <- function(theta) {
  # theta is assumed to be a k x 4 matrix of unbounded parameter values
  alpha <- 2 * pnorm(theta[,1])
  beta <- 2 * pnorm(theta[,2]) - 1
  mu <-theta[,4]
  c <- exp(theta[,3])
  
  N <- dim(theta)[1]
  
  U <- (runif(N) - 0.5) * pi
  W <- -log(runif(N))
  
  zeta <- beta * tan((pi * alpha) / 2)
  chi <- atan(zeta)
  X <- (1 + zeta * zeta)^(1 / (2 * alpha)) *
    sin(alpha * U + chi) / cos(U)**(1 / alpha) *
    (cos(U * (1 - alpha) - chi) / W)**((1 - alpha) / alpha)
  
  c * X + mu
}

p <- rnorm(4 * 500000)
dim(p) <- c(500000, 4)
a <- gen_sample(p)
