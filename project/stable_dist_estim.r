library(tidyverse)
library(StableEstim)
library(stabledist)

zt <- read.csv('euro_in_sterling.csv')$zt
yt <- 100 * log(zt[2:length(zt)] / zt[1:(length(zt) - 1)])

# Fit stable distribution to data, using a MLE method
fitML <- Estim('ML', yt, ComputeCov = TRUE)
smpls <- rstable(1e7, fitML@par['alpha'], fitML@par['beta'], fitML@par['gamma'], fitML@par['delta'])

smpls <- smpls[(smpls > -2) & (smpls < 2)]

ggplot(data=tibble(it=1:length(yt), yt=yt), 
       aes(x=yt)) +
  geom_histogram(aes(y = ..density..), bins = 40, alpha=0.4) +
  geom_histogram(data = tibble(smpls), aes(x = smpls, y = ..density..), bins = 40, fill='red', alpha=0.2) + 
  scale_colour_discrete(labels = c("Data", "Simulated"))
  