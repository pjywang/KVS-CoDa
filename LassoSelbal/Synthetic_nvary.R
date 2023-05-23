library(pracma) # logspace, logseq function

# parallel computation
library(foreach)
library(doParallel)

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
dirname = "./Synthetic_safe/nvary/"

# Read data (be sure to adjust mc.cores < cpu cores you have)
system.time({
  data_list <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      n <- seq(200, 1000, 200)[j]
      filename <- paste0(dirname, n, "_nvary_", i - 1, ".pickle")
      pickle_data <- pd$read_pickle(filename)
      pickle_data$Y[pickle_data$Y < 0] <- 0
      pickle_data$X <- zero_rep(pickle_data$X, method = "xmin", val = 0.5)
      ith_data[[j]] <- pickle_data
    }
    return(ith_data)
  }, mc.cores=50)
})


lamseq = logseq(0.04, 0.16, 90)

# 4 hours
system.time({
  all_results <- list()
  for (i in 1:50){
    ith_data <- list()
    for (j in 1:5){
      jth_data <-list()
      for (k in 1:length(lamseq)){
        jth_data[[k]] <-coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                            data_list[[i]][[j]]$X, lamseq[k])
      }
      ith_data[[j]] <- jth_data
    }
    all_results[[i]] <- ith_data
  }
})

save(all_results, file="./results/Synthetic_data/allmodels_nvary_0.5rep.RData")
load(file="./results/Synthetic_data/allmodels_nvary_0.5rep.RData")

for (i in 1:50){
  for (j in 1:5){
    cat(" ", length(all_results[[i]][[j]]))
  }
  cat("\n")
}


# Find suitable lambdas from the all_results
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  for (j in 1:5){
    lambdas[[i]][[6]] = list()
    for (k in 1:length(all_results[[i]][[j]])){
      betas = all_results[[i]][[j]][[k]]$betas
      betas = betas[2:length(betas)]
      numVar = as.integer(sum(betas != 0))
      if (numVar >= 10 && numVar < 15){
        lambdas[[i]][[j]] = c(lambdas[[i]][[j]],
                              c(lamseq, finer_lamseq)[k])
      }
    }
    if(length(lambdas[[i]][[j]]) == 0){
      cat(i, j, "\n")
    }
  }
}

# further search (17 min)
finer_lamseq = logseq(0.045, 0.17, 200)
system.time({
  for (i in 1:50){
    for (j in 1:5){
      if (length(lambdas[[i]][[j]]) == 0){
        for (lam in finer_lamseq){
          model <- coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                       data_list[[i]][[j]]$X, lam)
          all_results[[i]][[j]] = append(all_results[[i]][[j]], list(model))
        }
      }
    }
  }
})


mini = 3
maxi = 0
for (i in 1:50){
  for (j in 1:4){
    mini = min(mini, lambdas[[i]][[j]])
    maxi = max(maxi, lambdas[[i]][[j]])
  }
}

num_true_pos = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], c(lamseq, finer_lamseq))){
      sel_var = all_results[[i]][[j]][[lamidx]]$`name of selected variables`
      # cat(sel_var, '\n')
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]][[j]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_nvary_lasso_0.5rep.RData")

colMeans(num_true_pos) # [1] 4.58 5.88 6.14 6.40 6.64
apply(num_true_pos, 2, sd) / sqrt(50)
# [1] 0.1853623 0.1951242 0.1873772 0.2118914 0.1846342







##################################################################
######################### Selbal #################################
##################################################################

# took about 2.5min 
system.time({
  selbal_results_nvary <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      model_selbal <- selbal(data_list[[i]][[j]]$X, as.factor(data_list[[i]][[j]]$Y),
                             maxV = 10, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]][[j]]$X)
    }
    return(ith_data)
  }, mc.cores = 50)
})

save(selbal_results_nvary, 
     file="./results/Synthetic_data/selbalmodels_nvary_0.5rep.RData")
load(file="./results/Synthetic_data/selbalmodels_nvary_0.5rep.RData")


truepos_selbal_nvary = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    truepos_selbal_nvary[i, j] = sum(
      selbal_results_nvary[[i]][[j]]$varSelect %in% as.character(data_list[[i]][[j]]$True))
  }
}

colMeans(truepos_selbal_nvary) # [1] 2.80 3.16 3.82 3.84 3.82
apply(truepos_selbal_nvary, 2, sd) / sqrt(50)
# [1] 0.1714286 0.1698979 0.1448152 0.1224245 0.1330644

save(truepos_selbal_nvary, 
     file = "./results/Synthetic_data/Truepos_nvary_selbal_0.5rep.RData")





###################################################################
############ Other zero replacement methods #######################

####### 1sum #########
# Abou 0.8sec (50 cores)
system.time({
  data_list <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      n <- seq(200, 1000, 200)[j]
      filename <- paste0(dirname, n, "_nvary_", i - 1, ".pickle")
      pickle_data <- pd$read_pickle(filename)
      pickle_data$Y[pickle_data$Y < 0] <- 0
      pickle_data$X <- pickle_data$X + 1
      ith_data[[j]] <- pickle_data
    }
    return(ith_data)
  }, mc.cores=50)
})

# Selbal (1.2min)
system.time({
  selbal_results_nvary <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      model_selbal <- selbal(data_list[[i]][[j]]$X, as.factor(data_list[[i]][[j]]$Y),
                             maxV = 10, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]][[j]]$X)
    }
    return(ith_data)
  }, mc.cores = 50)
})

save(selbal_results_nvary, 
     file="./results/Synthetic_data/selbalmodels_nvary_1sum.RData")
load(file="./results/Synthetic_data/selbalmodels_nvary_1sum.RData")


truepos_selbal_nvary = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    truepos_selbal_nvary[i, j] = sum(
      selbal_results_nvary[[i]][[j]]$varSelect %in% as.character(data_list[[i]][[j]]$True))
  }
}

colMeans(truepos_selbal_nvary) # [1] 2.92 3.28 3.90 3.96 4.00
apply(truepos_selbal_nvary, 2, sd) / sqrt(50)
# [1] 0.1848110 0.1807919 0.1696335 0.1537557 0.1277753

save(truepos_selbal_nvary, 
     file = "./results/Synthetic_data/Truepos_nvary_selbal_1sum.RData")


#### Coda-lasso
lamseq = logseq(0.04, 0.16, 90)

# 90 lamseq: 200 min
system.time({
  all_results <- list()
  for (i in 1:50){
    ith_data <- list()
    for (j in 1:5){
      jth_data <-list()
      for (k in 1:length(lamseq)){
        jth_data[[k]] <-coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                            data_list[[i]][[j]]$X, lamseq[k])
      }
      ith_data[[j]] <- jth_data
    }
    all_results[[i]] <- ith_data
  }
})

save(all_results, file="./results/Synthetic_data/allmodels_nvary_1sum.RData")
load(file="./results/Synthetic_data/allmodels_nvary_1sum.RData")

for (i in 1:50){
  for (j in 1:5){
    cat(" ", length(all_results[[i]][[j]]))
  }
  cat("\n")
}


# Find suitable lambdas from the all_results
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  for (j in 1:5){
    lambdas[[i]][[6]] = list()
    for (k in 1:length(all_results[[i]][[j]])){
      betas = all_results[[i]][[j]][[k]]$betas
      betas = betas[2:length(betas)]
      numVar = as.integer(sum(betas != 0))
      if (numVar >= 10 && numVar < 15){
        lambdas[[i]][[j]] = c(lambdas[[i]][[j]],
                              c(lamseq, finer_lamseq)[k])
      }
    }
    if(length(lambdas[[i]][[j]]) == 0){
      cat(i, j, "\n")
    }
  }
}

# further search (6 min)
finer_lamseq = logseq(0.04, 0.17, 200)
system.time({
  for (i in 1:50){
    for (j in 1:5){
      if (length(lambdas[[i]][[j]]) == 0){
        for (lam in finer_lamseq){
          model <- coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                       data_list[[i]][[j]]$X, lam)
          all_results[[i]][[j]] = append(all_results[[i]][[j]], list(model))
        }
      }
    }
  }
})


mini = 3
maxi = 0
for (i in 1:50){
  for (j in 1:4){
    mini = min(mini, lambdas[[i]][[j]])
    maxi = max(maxi, lambdas[[i]][[j]])
  }
}

num_true_pos = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], c(lamseq, finer_lamseq))){
      sel_var = all_results[[i]][[j]][[lamidx]]$`name of selected variables`
      # cat(sel_var, '\n')
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]][[j]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_nvary_lasso_1sum.RData")

colMeans(num_true_pos) # [1] 4.66 6.22 6.32 6.64 7.08
apply(num_true_pos, 2, sd) / sqrt(50)
# [1] 0.1752200 0.2124493 0.2108099 0.2299246 0.1560743




##### gbm method ####

# 10sec
system.time({
  data_list <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      n <- seq(200, 1000, 200)[j]
      filename <- paste0(dirname, n, "_nvary_", i - 1, ".pickle")
      pickle_data <- pd$read_pickle(filename)
      pickle_data$Y[pickle_data$Y < 0] <- 0
      # gbm
      pickle_data$X <- cmultRepl2(pickle_data$X, zero.rep = "bayes")
      ith_data[[j]] <- pickle_data
    }
    return(ith_data)
  }, mc.cores=50)
})

# Selbal (70sec)
system.time({
  selbal_results_nvary <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:5){
      model_selbal <- selbal(data_list[[i]][[j]]$X, as.factor(data_list[[i]][[j]]$Y),
                             maxV = 10, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]][[j]]$X)
    }
    return(ith_data)
  }, mc.cores = 50)
})

save(selbal_results_nvary, 
     file="./results/Synthetic_data/selbalmodels_nvary_gbm.RData")
load(file="./results/Synthetic_data/selbalmodels_nvary_gbm.RData")


truepos_selbal_nvary = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    truepos_selbal_nvary[i, j] = sum(
      selbal_results_nvary[[i]][[j]]$varSelect %in% as.character(data_list[[i]][[j]]$True))
  }
}

colMeans(truepos_selbal_nvary) # [1] 1.56 2.22 2.44 2.78 2.58
apply(truepos_selbal_nvary, 2, sd) / sqrt(50)
# [1] 0.1542857 0.1673076 0.2003263 0.1835923 0.1809048

save(truepos_selbal_nvary, 
     file = "./results/Synthetic_data/Truepos_nvary_selbal_gbm.RData")



#### Coda-lasso
lamseq = logseq(0.04, 0.16, 90)

# 90 lamseq: 180 min
system.time({
  all_results <- list()
  for (i in 1:50){
    ith_data <- list()
    for (j in 1:5){
      jth_data <-list()
      for (k in 1:length(lamseq)){
        jth_data[[k]] <-coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                            data_list[[i]][[j]]$X, lamseq[k])
      }
      ith_data[[j]] <- jth_data
    }
    all_results[[i]] <- ith_data
  }
})

save(all_results, file="./results/Synthetic_data/allmodels_nvary_gbm.RData")
load(file="./results/Synthetic_data/allmodels_nvary_gbm.RData")

for (i in 1:50){
  for (j in 1:5){
    cat(" ", length(all_results[[i]][[j]]))
  }
  cat("\n")
}


# Find suitable lambdas from the all_results
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  for (j in 1:5){
    lambdas[[i]][[6]] = list()
    for (k in 1:length(all_results[[i]][[j]])){
      betas = all_results[[i]][[j]][[k]]$betas
      betas = betas[2:length(betas)]
      numVar = as.integer(sum(betas != 0))
      if (numVar >= 10 && numVar < 15){
        lambdas[[i]][[j]] = c(lambdas[[i]][[j]],
                              c(lamseq, finer_lamseq, further)[k])
      }
    }
    if(length(lambdas[[i]][[j]]) == 0){
      cat(i, j, "\n")
    }
  }
}

# further search (20 min)
finer_lamseq = logseq(0.06, 0.2, 150)
system.time({
  for (i in 1:50){
    for (j in 1:5){
      if (length(lambdas[[i]][[j]]) == 0){
        for (lam in finer_lamseq){
          model <- coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                       data_list[[i]][[j]]$X, lam)
          all_results[[i]][[j]] = append(all_results[[i]][[j]], list(model))
        }
      }
    }
  }
})

further = logseq(0.07, 0.18, 250)
system.time({
  for (i in 1:50){
    for (j in 1:5){
      if (length(lambdas[[i]][[j]]) == 0){
        for (lam in further){
          model <- coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                       data_list[[i]][[j]]$X, lam)
          all_results[[i]][[j]] = append(all_results[[i]][[j]], list(model))
        }
      }
    }
  }
})


mini = 3
maxi = 0
for (i in 1:50){
  for (j in 1:4){
    mini = min(mini, lambdas[[i]][[j]])
    maxi = max(maxi, lambdas[[i]][[j]])
  }
}

num_true_pos = zeros(50, 5)
for (i in 1:50){
  for (j in 1:5){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], c(lamseq, finer_lamseq))){
      sel_var = all_results[[i]][[j]][[lamidx]]$`name of selected variables`
      # cat(sel_var, '\n')
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]][[j]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_nvary_lasso_gbm.RData")

colMeans(num_true_pos) # [1] 3.66 4.28 4.54 4.74 4.34
apply(num_true_pos, 2, sd) / sqrt(50)
# [1] 0.2149703 0.2041208 0.2024543 0.1912579 0.2296581




