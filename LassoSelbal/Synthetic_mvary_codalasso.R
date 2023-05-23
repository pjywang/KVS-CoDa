library(pracma) # logspace, logseq function

# parallel computation
library(foreach)
library(doParallel)

setwd("/home/users/pjywang/Desktop/DimReduction/ConstrainedLASSO")

source(file = './CoDA-Penalized-Regression/R/functions_coda_penalized_regression.R')
source(file = './CoDA-Penalized-Regression/R/functions.R')
source(file = './selbal/R/Selbal_Functions.R')
source(file = './selbal/R/data.R')
source(file = './tools.R')

# To read *.pickle file in R
library(reticulate)
# use_python("your python.exe path")
pd <- import("pandas") # PANDAS VERSION: 1.5.2 (necessary)

# Where data are stored
dirname = "./Synthetic_safe/"

data_list <- list()
for (i in 1:50){
  filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
  pickle_data <- pd$read_pickle(filename)
  pickle_data$Y[pickle_data$Y < 0] <- 0
  
  # 0.5x_min replacement
  pickle_data$X[pickle_data$X == 0] <- 0.5
  
  data_list[[i]] = pickle_data 
}

len_lamseq = 50
lamseq = logseq(0.03, 0.19, 12 * len_lamseq)


# Parallel computation setting
# numCores <- detectCores() - 2
numCores <- 10
myCluster <- makeCluster(numCores)
registerDoParallel(myCluster)

start <- Sys.time()
# 3.3hours taken
all_results <- foreach(i = 1:50, 
                   .multicombine = TRUE) %:%
  foreach(lambda = lamseq) %dopar% {
    coda_logistic_lasso(data_list[[i]]$Y, data_list[[i]]$X, lambda)
  }
end = Sys.time()
cat(sprintf("\niter time = %f\n", end - start))

stopCluster(myCluster)
######################### Stop Cluster ###########################

save(all_results, file="./results/Synthetic_data/allmodels_0.5rep.RData")
load(file="./results/Synthetic_data/allmodels_0.5rep.RData")

# Find suitable lambdas from the all_results
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  # allocate the memory
  lambdas[[i]][[9]] = list()
  for (k in 1:960){
    betas = all_results[[i]][[k]]$betas
    betas = betas[2:length(betas)]
    numVar = as.integer(sum(betas != 0))
    if (numVar >= 5 && numVar < 45){
      j = floor(numVar / 5)
      lambdas[[i]][[j]] = c(lambdas[[i]][[j]], lamseq[k])
    }
  }
}

# Check if further search is required
# for (i in 1:50){ for (j in 1:8){ if (length(lambdas[[i]][[j]]) == 0){
#       cat(i, j, "\n")
#     }  } }

lamlen <- foreach(i = 1:50, .combine = 'rbind') %:% 
  foreach(j = 1:8, .combine = 'c') %do% {
    length(lambdas[[i]][[j]])
  }

# Check for the number of true positives

num_true_pos = zeros(50, 8)

for (i in 1:50){
  for (j in 1:8){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], lamseq)){
      sel_var = all_results[[i]][[lamidx]]$`name of selected variables`
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_lasso_0.5rep.RData")

# [1] 3.92 4.96 5.74 6.38 6.88 7.40 7.86 8.18
colMeans(num_true_pos)
# [1] 1.2751951 1.3546594 1.2906255 1.3231070 1.2229105 1.0879676 1.1250397 0.9833305
apply(num_true_pos, 2, sd)




################### Other Zero replacement strategies #####################

# GBM
data_list <- list()
for (i in 1:50){
  filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
  pickle_data <- pd$read_pickle(filename)
  pickle_data$Y[pickle_data$Y < 0] <- 0
  
  # GBM replacement, output is already compositional
  pickle_data$X <- cmultRepl2(pickle_data$X, zero.rep = "bayes")
  data_list[[i]] = pickle_data 
}

# Expecting 5/8 * 3.3 hours... but 5 cores were too small.
len_lamseq = 50
lamseq = logseq(0.03, 0.19, 12 * len_lamseq)

numCores <- 5
myCluster <- makeCluster(numCores)
registerDoParallel(myCluster)

start <- Sys.time()
# About 220MB memory usage, 3.3hours taken
all_results <- foreach(i = 1:50, 
                       .multicombine = TRUE) %:%
  foreach(lambda = lamseq) %dopar% {
    coda_logistic_lasso(data_list[[i]]$Y, data_list[[i]]$X, lambda)
  }
end = Sys.time()
cat(sprintf("\niter time = %f\n", end - start))

stopCluster(myCluster)
######################### Stop Cluster ###########################

save(all_results, file="./results/Synthetic_data/allmodels_gbm.RData")
load(file="./results/Synthetic_data/allmodels_gbm.RData")

# Find suitable lambdas from the all_results
lamseq = c(lamseq, newlamseq)
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  # allocate the memory
  lambdas[[i]][[9]] = list()
  for (k in 1:length(all_results[[i]])){
    betas = all_results[[i]][[k]]$betas
    betas = betas[2:length(betas)]
    numVar = as.integer(sum(betas != 0))
    if (numVar >= 5 && numVar < 45){
      j = floor(numVar / 5)
      lambdas[[i]][[j]] = c(lambdas[[i]][[j]], lamseq[k])
    }
  }
}

# Check if further search is required
need_further = c()
for (i in 1:50){ for (j in 1:8){ if (length(lambdas[[i]][[j]]) == 0){
      cat(i, j, "\n")
      need_further = c(need_further, i)
}  } }

# Further search..
newlamseq = logseq(0.19, 0.25, 50)
for (i in need_further){
  for (lam in newlamseq){
    model <- coda_logistic_lasso(data_list[[i]]$Y, data_list[[i]]$X, lam)
    all_results[[i]] = append(all_results[[i]], list(model))
  }
}

lamlen <- foreach(i = 1:50, .combine = 'rbind') %:% 
  foreach(j = 1:8, .combine = 'c') %do% {
    length(lambdas[[i]][[j]])
  }

num_true_pos = zeros(50, 8)

for (i in 1:50){
  for (j in 1:8){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], lamseq)){
      sel_var = all_results[[i]][[lamidx]]$`name of selected variables`
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_lasso_gbm.RData")

# [1] 3.08 3.84 4.42 4.90 5.58 6.08 6.44 6.98
colMeans(num_true_pos)
# [1] 1.352850 1.489555 1.310709 1.328648 1.162158 1.157760 1.145711 1.115567
apply(num_true_pos, 2, sd)


##################################################################
### 1sum replacement

data_list <- list()
for (i in 1:50){
  filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
  pickle_data <- pd$read_pickle(filename)
  pickle_data$Y[pickle_data$Y < 0] <- 0
  
  # 1sum replacement
  pickle_data$X <- pickle_data$X + 1
  data_list[[i]] = pickle_data 
}

# 
len_lamseq = 50
lamseq = logseq(0.03, 0.19, 12 * len_lamseq)

numCores <- 25
myCluster <- makeCluster(numCores)
registerDoParallel(myCluster)

start <- Sys.time()
# About 220MB memory usage, 3.3hours taken
all_results <- foreach(i = 1:50, 
                       .multicombine = TRUE) %:%
  foreach(lambda = lamseq) %dopar% {
    coda_logistic_lasso(data_list[[i]]$Y, data_list[[i]]$X, lambda)
  }
end = Sys.time()
cat(sprintf("\niter time = %f\n", end - start))

stopCluster(myCluster)
######################### Stop Cluster ###########################

save(all_results, file="./results/Synthetic_data/allmodels_1sum.RData")
load(file="./results/Synthetic_data/allmodels_1sum.RData")


# Find suitable lambdas from the all_results
# lamseq = c(lamseq, newlamseq)
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  # allocate the memory
  lambdas[[i]][[9]] = list()
  for (k in 1:length(all_results[[i]])){
    betas = all_results[[i]][[k]]$betas
    betas = betas[2:length(betas)]
    numVar = as.integer(sum(betas != 0))
    if (numVar >= 5 && numVar < 45){
      j = floor(numVar / 5)
      lambdas[[i]][[j]] = c(lambdas[[i]][[j]], lamseq[k])
    }
  }
}

# Check if further search is required
need_further = c()
for (i in 1:50){ for (j in 1:8){ if (length(lambdas[[i]][[j]]) == 0){
  cat(i, j, "\n")
  need_further = c(need_further, i)
}  } }

lamlen <- foreach(i = 1:50, .combine = 'rbind') %:% 
  foreach(j = 1:8, .combine = 'c') %do% {
    length(lambdas[[i]][[j]])
  }

num_true_pos = zeros(50, 8)

for (i in 1:50){
  for (j in 1:8){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], lamseq)){
      sel_var = all_results[[i]][[lamidx]]$`name of selected variables`
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_lasso_1sum.RData")

# [1] 4.04 5.00 5.76 6.38 7.10 7.42 7.92 8.10
colMeans(num_true_pos)
# [1] 1.194545 1.261680 1.333401 1.496799 1.388730 1.108225 1.103612 1.073807
apply(num_true_pos, 2, sd)
