# Implement selbal for COMBO dataset (bmi dataset)

source(file = './selbal/R/Selbal_Functions.R')
source(file = './selbal/R/data.R')
source(file = './tools.R')


# To use the same cross-validation splits as used in python codes
library(reticulate)
# use_python("your python path")
pd <- import("pandas")
np <- import("numpy")
mod <- import("sklearn.model_selection")


# Load and preprocess
# filepath = "..parent folder../datasets/COMBO_data/"

count <- read.csv(paste0(filepath, "complete_data/GeneraCounts.csv"), header = F)
tax_table <- t(read.csv(paste0(filepath, "complete_data/GeneraPhylo.csv"), header = F))
y <- read.csv(paste0(filepath, "BMI.csv"), header = F)
y <- y$V1
x <- t(count)
colnames(x) <- rownames(count)

# Delete columns with only one nonzero value (to compare results of gbm zero replacement)
remove_idx <- c()
for (j in 1:87){
  if (sum(x[, j] > 0) == 1){
    remove_idx <- c(remove_idx, j)
  }
}
x_rem <- x[, -remove_idx]




####### Experiments #######


# 0.5min replacement
x <- zero_rep(x_rem, method="xmin", val=0.5)
selbal_bmi <- myselbal.cv(x = x, y = y, n.fold = 5, n.iter = 10,
                          covar = NULL, maxV = 10)

# worse than codalasso and our method
# note: deleting columns with only one positive values improved selbal performance
selbal_bmi[["accuracy.nvar"]][["data"]][["mean"]]
# [1] 31.09797 33.03377 32.57037 32.91240 32.90981 33.58666 33.76605 34.19465 34.63952

selbal_bmi[["accuracy.nvar"]][["data"]]$se
# [1] 1.716032 1.661020 1.766669 1.793564 1.794204 1.844242 1.882902 1.832050 1.845399




# 1sum replacement
x <- x_rem + 1
selbal_bmi <- myselbal.cv(x = x, y = y, n.fold = 5, n.iter = 10,
                          covar = NULL, maxV = 10)

selbal_bmi[["accuracy.nvar"]][["data"]][["mean"]]
# [1] 31.30017 33.46035 33.73769 33.92159 34.96790 34.53042 35.54682 35.75858 36.51980

selbal_bmi[["accuracy.nvar"]][["data"]]$se
# [1] 1.705231 1.729238 1.775242 1.848470 1.924958 1.894478 2.032145 2.005671 1.949277




# GBM zero replacement method (automatically applied in myselbal.cv)
selbal_bmi <- myselbal.cv(x = x_rem, y = y, n.fold = 5, n.iter = 10,
                          covar = NULL, maxV = 10)

selbal_bmi[["accuracy.nvar"]][["data"]][["mean"]]
# [1] 34.13090 32.90741 34.58820 37.05251 37.05684 38.99642 39.58636 40.08390 41.16069

selbal_bmi[["accuracy.nvar"]][["data"]]$se
# [1] 1.810673 2.004286 2.945553 3.574267 3.491945 4.013374 3.405337 3.743086 4.121520




#########################
# save gbm replaced data
x_rep <- cmultRepl2(x_rem)
x_rep_df <- data.frame(x_rep)
colnames(x_rep_df) <- colnames(x_rep)

write.csv(x_rep_df, file = paste0(filepath, "complete_data/GeneraCountsGBM.csv"))
