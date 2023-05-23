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

# load data
system.time({
  data_list <- mclapply(1:50, function(i){
    filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
    pickle_data <- pd$read_pickle(filename)
    pickle_data$Y[pickle_data$Y < 0] <- 0
    
    # 0.5x_min replacement
    pickle_data$X <- zero_rep(pickle_data$X, "xmin", 0.5)
    return(pickle_data)
  }, mc.cores = 50)
})


#######################################################################
######################### Selbal execution ############################
#######################################################################

# mclapply (3.65 min)
system.time({
  selbal_results <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:8){
      model_selbal <- selbal(data_list[[i]]$X, as.factor(data_list[[i]]$Y),
                             maxV = 5 * j, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]]$X)
    }
    return(ith_data)
  }, mc.cores=50)
})

truepos_selbal = zeros(50, 8)
stopping_point = c()
for (i in 1:50){
  for (j in 1:8){
    truepos_selbal[i, j] = sum(selbal_results[[i]][[j]]$varSelect %in% as.character(data_list[[i]]$True))
  }
  stopping_point[i] = selbal_results[[i]][[j]]$numVarSelect
}

colMeans(truepos_selbal) # [1] 1.82 2.80 3.12 3.40 3.50 3.58 3.60 3.60

apply(truepos_selbal, 2, sd) / sqrt(50)
# [1] 0.1235198 0.1714286 0.1843688 0.1958758 0.2025350 0.2062097 0.2080031 0.2080031

save(selbal_results, file = "./results/Synthetic_data/selbalresults_0.5rep.RData")
save(truepos_selbal, file = "./results/Synthetic_data/Truepos_selbal_0.5rep.RData")



##################################
# 1sum replacement

system.time({
  data_list <- mclapply(1:50, function(i){
    filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
    pickle_data <- pd$read_pickle(filename)
    pickle_data$Y[pickle_data$Y < 0] <- 0
    
    # 0.5x_min replacement
    pickle_data$X <- pickle_data$X + 1
    return(pickle_data)
  }, mc.cores = 50)
})

# ncores 25 -> 3.7min // ncores 50 -> 3.67min
system.time({
  selbal_results <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:8){
      model_selbal <- selbal(data_list[[i]]$X, as.factor(data_list[[i]]$Y),
                             maxV = 5 * j, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]]$X)
    }
    return(ith_data)
  }, mc.cores=50)
})

truepos_selbal = zeros(50, 8)
stopping_point = c()
for (i in 1:50){
  for (j in 1:8){
    truepos_selbal[i, j] = sum(selbal_results[[i]][[j]]$varSelect %in% as.character(data_list[[i]]$True))
  }
  stopping_point[i] = selbal_results[[i]][[j]]$numVarSelect
}

colMeans(truepos_selbal) # [1] 1.80 2.92 3.22 3.50 3.62 3.74 3.74 3.74
apply(truepos_selbal, 2, sd) / sqrt(50)
# [1] 0.1069045 0.1848110 0.1813554 0.2025350 0.2018082 0.2076300 0.2076300 0.2076300

save(selbal_results, file = "./results/Synthetic_data/selbalresults_1sum.RData")
save(truepos_selbal, file = "./results/Synthetic_data/Truepos_selbal_1sum.RData")



##################################
# gbm replacement

system.time({
  data_list <- mclapply(1:50, function(i){
    filename <- paste0(dirname, "200_100_log5_", i - 1, ".pickle")
    pickle_data <- pd$read_pickle(filename)
    pickle_data$Y[pickle_data$Y < 0] <- 0
    
    # gbm
    pickle_data$X <- cmultRepl2(pickle_data$X, zero.rep = "bayes")
    return(pickle_data)
  }, mc.cores = 50)
})

# 5min
system.time({
  selbal_results <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:8){
      model_selbal <- selbal(data_list[[i]]$X, as.factor(data_list[[i]]$Y),
                             maxV = 5 * j, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]]$X)
    }
    return(ith_data)
  }, mc.cores=50)
})

truepos_selbal = zeros(50, 8)
stopping_point = c()
for (i in 1:50){
  for (j in 1:8){
    truepos_selbal[i, j] = sum(selbal_results[[i]][[j]]$varSelect %in% as.character(data_list[[i]]$True))
  }
  stopping_point[i] = selbal_results[[i]][[j]]$numVarSelect
}

colMeans(truepos_selbal) # [1] 1.12 1.56 1.90 2.16 2.26 2.32 2.36 2.36
apply(truepos_selbal, 2, sd) / sqrt(50)
# [1] 0.1329109 0.1542857 0.1835033 0.1986896 0.2115251 0.2165405 0.2263463 0.2263463

save(selbal_results, file = "./results/Synthetic_data/selbalresults_gbm.RData")
save(truepos_selbal, file = "./results/Synthetic_data/Truepos_selbal_gbm.RData")


