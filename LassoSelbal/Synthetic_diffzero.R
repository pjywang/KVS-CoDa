# Different zero ratios

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
dirname = "./Synthetic_safe/zerovary/"

# 10 sec for 50 cores
system.time({
  data_list <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:4){
      ratio <- seq(10, 70, 20)[j]
      filename <- paste0(dirname, "zero_", as.character(ratio), '_', i - 1, ".pickle")
      pickle_data <- pd$read_pickle(filename)
      pickle_data$Y[pickle_data$Y < 0] <- 0
      # pickle_data$X <- zero_rep(pickle_data$X, method = "xmin", val = 0.5)
      ith_data[[j]] <- pickle_data
    }
    return(ith_data)
  }, mc.cores = 50)
})



lamseq = logseq(0.04, 0.15, 80)

# length 80
start <- Sys.time()
all_results <- mclapply(1:50, function(i){
  ith_data <- list()
  for (j in 1:4){
    jth_data <- list()
    for (k in 1:length(lamseq)){
      jth_data[[k]] <- coda_logistic_lasso(data_list[[i]][[j]]$Y, 
                                           data_list[[i]][[j]]$X, lamseq[k])
    }
    ith_data[[j]] <- jth_data
  }
  return(ith_data)
}, mc.cores = 25)
end = Sys.time()
cat(sprintf("\niter time = %f\n", end - start))

              

save(all_results, file="./results/Synthetic_data/allmodels_diffzero_0.5rep.RData")
load(file="./results/Synthetic_data/allmodels_diffzero_0.5rep.RData")


for (i in 1:50){
  for (j in 1:4){
    cat(" ", length(all_results[[i]][[j]]))
  }
  cat("\n")
}


# Find suitable lambdas from the all_results
lambdas = list()
for (i in 1:50){
  lambdas[[i]] = list()
  for (j in 1:4){
    lambdas[[i]][[5]] = list()
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

# 16 min
finer_lamseq = logseq(0.04, 0.15, 200)
system.time({
  for (i in 1:50){
    for (j in 1:4){
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

num_true_pos = zeros(50, 4)

for (i in 1:50){
  for (j in 1:4){
    get_max = 0
    # Use match function to find the corresponding indices
    for (lamidx in match(lambdas[[i]][[j]], c(lamseq, finer_lamseq))){
      sel_var = all_results[[i]][[j]][[lamidx]]$`name of selected variables`
      cat(sel_var, '\n')
      get_max = max(get_max, sum(sel_var %in% as.character(data_list[[i]][[j]]$True)))
    }
    num_true_pos[i, j] = get_max
  }
}
save(num_true_pos, file = "./results/Synthetic_data/Truepos_diffzero_lasso_0.5rep.RData")

colMeans(num_true_pos) # [1] 2.32 3.70 5.76 6.30
apply(num_true_pos, 2, sd) # [1] 1.583589 1.265718 1.436549 1.373956

# Data with few zeros are more contaminated by zero-replacement + log-ratio




##################################################################
######################### Selbal #################################
##################################################################

# 2min
system.time({
  selbal_results_zerovary <- mclapply(1:50, function(i){
    ith_data <- list()
    for (j in 1:4){
      model_selbal <- selbal(data_list[[i]][[j]]$X, as.factor(data_list[[i]][[j]]$Y),
                             maxV = 10, draw = F)
      ith_data[[j]] <- selbal_wrapper(model_selbal, data_list[[i]][[j]]$X)
    }
    return(ith_data)
  }, mc.cores = 50)
})

save(selbal_results_zerovary, 
     file="./results/Synthetic_data/selbalmodels_zerovary_0.5rep.RData")
load(file="./results/Synthetic_data/selbalmodels_zerovary_0.5rep.RData")

truepos_selbal_zerovary = zeros(50, 4)
for (i in 1:50){
  for (j in 1:4){
    truepos_selbal_zerovary[i, j] = sum(
      selbal_results_zerovary[[i]][[j]]$varSelect %in% as.character(data_list[[i]][[j]]$True))
  }
}

colMeans(truepos_selbal_zerovary) # [1] 0.78 1.74 3.36 4.22
apply(truepos_selbal_zerovary, 2, sd) / sqrt(50)
# [1] 0.1318317 0.1533570 0.1584620 0.1697297

save(truepos_selbal_zerovary, 
     file="./results/Synthetic_data/Truepos_zerovary_selbal_0.5rep.RData")


