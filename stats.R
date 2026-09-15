# install.packages(c("tidyverse", "lubridate", "lme4","readr","dplyr","tidyr","lmerTest"))
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(lme4)
library(lmerTest)   # optional, for p-values

df <- read_csv("C:/Users/paul.grube.ZI/Desktop/df.csv")
fit <- lmer(
  bprs ~ cent_c+clus_c+ids_pres+ time +(1 +time | code),
  data = df,
  na.action = na.omit
)

summary(fit)

# lm_within <- lm(clus_c ~cent_c+int_ratio_c+rel_int_c+ids_pres+female+age, data = df)
# summary(lm_within)$r.squared
# 27 % shared variance between clus and cent?
install.packages("r2glmm")
library(r2glmm)

r2_out <- r2beta(
  fit,
  partial = TRUE,
  method = "nsj"
)

print(r2_out)
install.packages("performance")
library(performance)
m123 <- lmer(bprs ~ clus_c+cent_c+ids_pres + time + (1 + time | code), data = df)
m12  <- lmer(bprs ~ clus_c+cent_c + time + (1 + time | code), data = df)
m13  <- lmer(bprs ~ clus_c + ids_pres + time + (1 + time | code), data = df)
m23  <- lmer(bprs ~ cent_c + ids_pres  + time + (1 + time | code), data = df)

m1   <- lmer(bprs ~ clus_c + time + (1 + time | code), data = df)
m2   <- lmer(bprs ~ cent_c + time + (1 + time | code), data = df)
m3   <- lmer(bprs ~ ids_pres + time + (1 + time | code), data = df)

m0   <- lmer(bprs ~ time + (1 + time | code), data = df)
R123 <- r2_nakagawa(m123)$R2_marginal
R12  <- r2_nakagawa(m12)$R2_marginal
R13  <- r2_nakagawa(m13)$R2_marginal
R23  <- r2_nakagawa(m23)$R2_marginal
R1   <- r2_nakagawa(m1)$R2_marginal
R2   <- r2_nakagawa(m2)$R2_marginal
R3   <- r2_nakagawa(m3)$R2_marginal
R0   <- r2_nakagawa(m0)$R2_marginal
U1 <- R123 - R23   # unique to x1
U2 <- R123 - R13   # unique to x2
U3 <- R123 - R12   # unique to x3

C12 <- R12 - R0 - U1 - U2
C13 <- R13 - R0 - U1 - U3
C23 <- R23 - R0 - U2 - U3

C123 <- R123 - R0 - U1 - U2 - U3 - C12 - C13 - C23

commonality <- data.frame(
  component = c("U1", "U2", "U3", "C12", "C13", "C23", "C123"),
  value = c(U1, U2, U3, C12, C13, C23, C123)
)

print(commonality)
sum(commonality$value)
print(R123 - R0)
