# Repeatedly used functions

# Library collision may occur
# Import this file at the end of the importing procedure.
library(MLmetrics)
library(pROC)
library(pracma)

library(foreach)
library(doParallel)

source(file = './CoDA-Penalized-Regression/R/functions_coda_penalized_regression.R')
source(file = './CoDA-Penalized-Regression/R/functions.R')
source(file = './selbal/R/Selbal_Functions.R')

# Dependence: the CMA library..
# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("CMA")


# only for classification
get.estimator <- function(estimator, x, y, lambda=0.2, maxV=20){
  if(estimator=='codalasso'){
    model <- coda_logistic_lasso(y = y, X = x, lambda = lambda)
  } else if (estimator=='selbal'){
    model <- selbal(x, as.factor(y), maxV = maxV, draw = F)
  }
  return(model)
}

predictor <- function(model, estimator='codalasso', x=NULL,
                      idx=NULL){
  if (estimator=='codalasso'){
    # will be rounded in scores function
    return(predict_codalasso(x, model = model, type = "response"))
  } else if (estimator=='selbal'){
    logCounts <- log(x)
    POS = model$numerator
    NEG = model$denominator
    k1 <- length(POS); k2 <- length(NEG)
    
    # Balance for prediction
    FINAL.BAL <- sqrt((k1*k2)/(k1+k2))*
      (rowM(logCounts[,POS])- rowM(logCounts[,NEG]))
    U <- data.frame(FINAL.BAL)
    colnames(U)[1] <- "V1"
    
    prob = predict.glm(model$glm, U, type = "response")
    
    if(!is.null(idx)){prob = prob[idx]}
    return(prob)
    }
}

# scorer(y_true, predictor(model, estimator, x), scoring, positive)
scorer <- function(y_true, y_pred, scoring='f1', positive=1){
  if(scoring=='f1'){
    return(F1_Score(y_true, round(y_pred), positive = positive))
  } else if (scoring=='auc'){
    return(as.numeric(auc(y_true, y_pred, quiet=T)))
  } else if (scoring=='accuracy'){
    return(sum(y_true==round(y_pred)) / length(y_true))
  } else {print("scoring name is wrong: use 'f1', 'auc', or 'accuracy'")}
}


# Iterated cv score (only for classification!)
cv.score <- function(x, y, group=NULL, estimator='codalasso',
                     lamseq = logseq(0.01, 1, 50),
                     outer_folds=5, inner_folds=5, N_TRIALS=10,
                     oneSE=FALSE,
                     scoring='f1', groupfold=FALSE, positive=1){
  # scoring: 'f1', 'accuracy', 'auc'
  # estimator: 'codalasso' or 'selbal'
  # positive: only used when scoring='f1'
  # *** need to change y as 0, 1 vector before apply ftn
  # groupfold: not yet developed.. group seperating in inner fold needed
  #            this will be easy! just take group[train_idx]
  # lamseq: will be an integer-sequence of maxV's when 'selbal'
  
  Outer_scores = NULL
  models = list()
  for (i in 1:N_TRIALS) {
    cat("\n", i,"th iteration\n")
    Outer_cv = mod$StratifiedKFold(n_splits = as.integer(outer_folds),
                                   shuffle = T, random_state=as.integer(i))
    if(groupfold){
      Outer_cv = mod$StratifiedGroupKFold(n_splits = as.integer(outer_folds),
                                          shuffle = T, random_state=as.integer(i))
    } 
    Outer_split = Outer_cv$split(x, y, groups = group)
    Outer_split <- iterate(Outer_split)
    models[[i]] = list()
    Outer_score = rep(0, outer_folds)
    # Outer CV loop for estimation of cv error
    for (k in 1:outer_folds){
      cat("\t", k, "th fold: ")
      train_idx = Outer_split[[k]][[1]] + 1
      test_idx = Outer_split[[k]][[2]] + 1
      
      # inner cv loop for parameter choice (model)
      inner_cv = mod$StratifiedKFold(n_splits=as.integer(inner_folds), shuffle=T,
                                     random_state = as.integer(i*10+k-1))
      inner_split = inner_cv$split(x[train_idx, ], y[train_idx])
      inner_split <- iterate(inner_split)
      inner.result <- matrix(0, length(lamseq), inner_folds)
      # inner cv loop
      for (l in 1:5){
        real_train <- train_idx[inner_split[[l]][[1]]]
        val_idx <- train_idx[inner_split[[l]][[2]]]
        # model learning for all lambdas
        for (m in 1:length(lamseq)){
          model <- get.estimator(estimator = estimator, x = x[real_train, ],
                                 y= y[real_train], lambda = lamseq[m], maxV = lamseq[m])

          inner.result[m, l] = scorer(y_true = y[val_idx],
                                      y_pred = predictor(model, estimator, x[val_idx, ]),
                                      scoring = scoring, positive = positive)
        }
      } # inner loop done
      inner.result <- rowMeans(inner.result)
      # cat("Inner cv scores", inner.result, "\n")  # for code check
      bestidx = which.max(inner.result)
      if(oneSE){
        se = sd(inner.result) / sqrt(length(lamseq))
        idxes = which(inner.result > (max(inner.result) - se))
        bestidx = idxes[length(idxes)]
      }
      best_lambda = lamseq[bestidx]
      cat("chosen lambda is", best_lambda, "\n")
      
      # Refit and Calculate test accuracy of kth outer fold
      models[[i]][[k]] = get.estimator(estimator, x = x[train_idx, ],
                                       y = y[train_idx], lambda = best_lambda,
                                       maxV = best_lambda)

      Outer_score[k] = scorer(y_true = y[test_idx],
                              y_pred = predictor(models[[i]][[k]], estimator = estimator,
                                                 x = x[test_idx, ]),
                              scoring = scoring, positive = positive)
      
      cat("\t\tscore here:", Outer_score[k], "\n")
      if(estimator == 'codalasso'){
        cat("\t\tselected indices:", models[[i]][[k]]$'indices of selected variables', "\n\n")
      } else if (estimator == 'selbal'){
        sel_idx = which(colnames(x) %in% models[[i]][[k]]$'balance')
        cat("\t\tselected indices:", sel_idx, "\n\n")
      }
    }
    Outer_scores = rbind(Outer_scores, Outer_score)
  }
  print(mean(Outer_scores))
  
  cv_results = list("scores" = Outer_scores,
                    "models" = models)
  return(cv_results)
}


# Zero replacement function
zero_rep <- function(X, method = "xmin", val = 0.5){
  # method: "count" or "xmin", replace by val, or "sum" by val
  
  # we often have a 0.5count-replaced data
  X[X == 0.5] = 0
  if (method == "count") {X[X == 0] = val}
  else if (method == "xmin"){
    for (i in 1:nrow(X)){
      xmin = min(X[i, as.vector(X[i, ] > 0)])
      X[i, as.vector(X[i, ] == 0)] = xmin * val
    }
  } else if (method == "sum") {X = X + val}
  
  return(X)
}


# To use f1score, type="terms" needed
predict_codalasso <- function(x, model, type = "terms"){
  z = log(x)
  z <- cbind(rep(1,nrow(z)),z)
  z <- as.matrix(z)
  
  logit = z %*% model$"betas"
  prob = exp(logit) / (1 + exp(logit))
  if (type == "response") return(prob)
  else if (type == "terms") return(round(prob))
}


myselbal.cv <- function(x, y, n.fold = 5, n.iter = 10,
                        covar = NULL, col = c("steelblue1", "tomato1"),
                        col2 = c("darkgreen", "steelblue4","tan1"),
                        logit.acc = "f1", positive=1,
                        maxV = 20, zero.rep = "bayes",
                        opt.cri = "1se", user_numVar = NULL){
  
  # Load package plyr
  suppressMessages(library(plyr))
  
  
  #------------------------------------------------------------------------------#
  
  #----------------------------------------------------------------------------#
  # STEP 0: build the necessary objects
  #----------------------------------------------------------------------------#
  
  #-----------------------------------------------#
  # 0.1: Build necessary objects for the function
  #-----------------------------------------------#
  
  # Class of the response
  classy <- class(y)
  # Family for the glm (default: gaussian)
  f.class <- "gaussian"
  # numy to be y and it will be modified if y is a factor
  numy <- y
  # If y is a factor, compute the number of levels of y
  if (classy == "factor"){
    ylev <- levels(y)
    numy <- as.numeric(y) - 1
    f.class <- "binomial"
  }
  
  # Names of x
  x.nam <- colnames(x)
  
  # ROB.TAB
  ROB.TAB <- matrix(0, nrow = ncol(x), ncol=3)
  colnames(ROB.TAB) <- c("Prop. Included", "Prop_Numerator",
                         "Prop_Denominator")
  row.names(ROB.TAB) <- x.nam
  
  # BAL.resume
  BAL.resume <- matrix(0,nrow =n.fold*n.iter , ncol = ncol(x))
  colnames(BAL.resume) <- colnames(x)
  
  # Build a table with the response variable and covariates for correction
  if (!is.null(covar)){ dat <- data.frame(cbind(numy, covar))
  } else { dat <-data.frame(numy)}
  
  # Message starting the algorithm
  cat(paste("\n\n###############################################################",
            "\n STARTING selbal.cv FUNCTION",
            "\n###############################################################"))
  # Log-transformed counts (with zero replacement
  cat(paste(
    "\n\n#-------------------------------------------------------------#",
    "\n# ZERO REPLACEMENT . . .\n\n"))
  
  # Define log-transformed data with the zero-replacement made
  logc <- log(cmultRepl2(x, zero.rep = zero.rep))
  
  
  cat(paste("\n, . . . FINISHED.",
            "\n#-------------------------------------------------------------#"))
  
  #---------------------------------------#
  # 0.2: Define cross - validation groups
  #---------------------------------------#
  
  # CROSS - VALIDATION groups
  CV.groups <- matrix(, nrow=n.iter * n.fold, ncol=nrow(x))
  for (i in 1:n.iter){
    Outer_cv = mod$StratifiedKFold(n_splits = as.integer(n.fold),
                                   shuffle = T, random_state=as.integer(i))
    if(classy=="numeric"){
      Outer_cv = mod$KFold(n_splits = as.integer(n.fold),
                           shuffle = T, random_state=as.integer(i))
    }
    sep = Outer_cv$split(x, y)
    sep <- iterate(sep)
    for (j in 1:n.fold){
      CV.groups[(i-1) * n.fold + j, 1:length(sep[[j]][[1]])] <- sep[[j]][[1]] + 1
      # CV.groups <- rbind(CV.groups, sep[[j]][[1]])
    }
  }
  # CV.groups <- CV.groups + 1
  
  suppressMessages(library(pROC))
  
  
  #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
  # Function for cross - validation
  #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
  
  cv.MSE <- function(k){
    
    #-------------------------------------#
    # Necessary matrices for the function
    #-------------------------------------#
    
    # Folds associated to a "complete FOLD"
    CV.group<-CV.groups[((k-1)*n.fold + 1):(k*n.fold),]
    
    # Build objects
    # Table with all the variables included
    Bal.List <- list()
    # Matrix for MSE values (or ACC "Accuracy")
    ACC.Mat <- matrix(0,nrow = maxV-1, ncol = nrow(CV.group))
    
    
    # For each fold
    for (i in 1:nrow(CV.group)){
      
      # Define training data.set
      train.idx<-CV.group[i,]
      train.idx<-train.idx[!is.na(train.idx)]
      # Training dataset (x, y and covar)
      x.train<-logc[train.idx,]
      x.test<-logc[-train.idx,]
      y.train<-y[train.idx]
      y.test<-y[-train.idx]
      covar.train<-covar[train.idx,]
      covar.test<-covar[-train.idx,]
      
      # Compute the balances for the training data set
      BAL <- selbal.aux(x.train, y.train, th.imp = 0, covar = covar.train,
                        logit.acc, logt=F, maxV = maxV)
      
      # Variables included in the balance (as NUMERATOR | DENOMINATOR)
      Bal.List[[i]]<-BAL
      
      # A matrix for predictions for Y
      PRED.y <- matrix(0, nrow = length(y.test), ncol = nrow(BAL) - 1)
      
      # For each number of variables (2:nrow(BAL))
      for (l in 2:min(maxV,nrow(BAL))){
        # Data frame for train data
        df <-data.frame(Y = y.train, B = bal.value(BAL[1:l,],x.train))
        
        # Data frame for test data
        df.test <- data.frame(Y = y.test, B = bal.value(BAL[1:l,],x.test))
        
        if(!is.null(covar)){
          df <- cbind(df,cov=covar.train)
          df.test <- cbind(df.test,cov=covar.test)
        }
        
        # Regression model for test data
        FIT <- glm(Y ~ ., data= df, family = f.class)
        # Predictions
        PRED.y[,l-1] <- predict(FIT,df.test, type="response")
      }
      
      #------------------------------------------------------------------------------#
      # FUNCTION: Measure the error value
      #------------------------------------------------------------------------------#
      
      ACC.eval <- function(y, pred, classy, logit.acc=NULL){
        
        if (classy == "numeric"){
          ACC <- apply(pred, 2, function(x) mean((y-x)^2))
        }else{
          if (logit.acc == "AUC"){
            # Load library
            library(pROC)
            ACC <- apply(pred, 2, function(x) auc(y,x, quiet=TRUE))
          } else if(logit.acc == "Rsq"){
            ACC <- apply(pred, 2, function(x) cor (y, x)^2)
          } else if (logit.acc == "Tjur"){
            ACC <- apply(pred, 2, function(x) mean(x[y==1]) - mean(x[y==0]))
          } else if (logit.acc == "Dev"){
            ACC<-apply(pred, 2, function(x) 1-(deviance(glm(y ~ x, data= df, family = binomial()))/glm(y~1, family=binomial())[[10]]) )  # proportion of explained deviance
          } else if (logit.acc == "f1"){
            ACC <- apply(pred, 2, function(x) F1_Score(y, round(x), positive = positive))
          } else if (logit.acc == "accuracy"){
            ACC <- apply(pred, 2, function(x) mean(y == round(x)))
          }
          
        }
        
        return(ACC)
      }
      
      #------------------------------------------------------------------------------#
      
      # Run ACC.eval function
      R <- ACC.eval(numy[-train.idx], PRED.y, classy=classy, logit.acc)
      # Add the information
      ACC.Mat[,i] <- c(R, rep(R[length(R)], maxV-length(R)-1))
    } # End of i
    
    return(list(Bal.List, ACC.Mat))
  }
  
  #------------------------------------------------------------------------------#
  
  ################################################################################
  
  cat(paste("\n\n#-------------------------------------------------------------#",
            "\n# Starting the cross - validation procedure . . ."))
  
  # Build a parallelization scenario
  suppressMessages(library(foreach))
  suppressMessages(library(doParallel))
  # Number of cores of the computer but one
  no_cores <- n.iter
  # Register the number of cores
  myCluster <- makeCluster(no_cores)
  registerDoParallel(myCluster)
  
  # Define the function comb
  # basically, we want to combine "lists of two elements"
  comb <- function(x, ...) {
    lapply(seq_along(x),
           function(i) c(x[[i]], 
                         lapply(list(...), function(y) y[[i]]) # Here survives only
                         )
           )
  }
  # x <- list(list(), list()) is assigned
  # seq_along(x) == c(1, 2)
  # Through lapply, we obtain a list of two lists:
  # [[1]]: list(...'s [[1]]s) <- is this a valid operation?
  # [[2]]: list(...'s [[2]]s)
  # When classification, there exists h such that cv.MSE[[2]] doesn't exist?
  
  ################################################################################
  
  # CV - procedure computed in parallel
  INTEREST <- foreach(h=1:n.iter,
                      .export=c("logit.cor", "rowM","selbal.aux", "bal.value",
                                "logit.acc", "cmultRepl","cmultRepl2", "F1_Score"),
                      # .export=c("F1_Score"),
                      .combine='comb',
                      .multicombine=TRUE,
                      .init=list(list(), list())
                      ) %dopar% {
                        tryCatch({cv.MSE(h)}, error = function(e){
                          return(paste0("Error: ", e))
                        })
                        # cv.MSE(h)
                      }
  # Stop the parallelization
  stopCluster(myCluster)
  stopImplicitCluster()
  
  cat(paste("\n . . . finished.",
            "\n#-------------------------------------------------------------#",
            "\n###############################################################"))
  
  #------------------------------------------------------------------------------#
  
  # Rebuild the objects from INTEREST
  Balances <- unlist(INTEREST[[1]], recursive = F)
  ACC.Matrix <- do.call(cbind,INTEREST[[2]])
  
  # ACC mean values for each number of variables
  ACC.mean <- apply(ACC.Matrix,1,mean)
  ACC.se <- apply(ACC.Matrix,1,function(x) sd(x)/sqrt(length(x)))
  
  if(classy == "numeric"){
    # Define the minimum mean value
    m <- which.min(ACC.mean)
    # The minimum value under ACC.mean[m] + SE
    if(length(which((ACC.mean<(ACC.mean[m] + ACC.se[m]))==T))>0){
      mv <- min(which((ACC.mean<(ACC.mean[m] + ACC.se[m]))==T))
    } else {mv<-m}
    
    # Depending on "opt.cri":
    if (opt.cri == "1se"){opt.M <- mv + 1
    }else {opt.M <- m + 1}
  }else{
    # Define the maximum ACC value
    m <- which.max(ACC.mean)
    # The minimum value whith ACC.mean over ACC.mean[m] - ACC.sd[m]
    if(length(which((ACC.mean>(ACC.mean[m] - ACC.se[m]))==T))>0){
      mv <- min(which((ACC.mean>(ACC.mean[m] - ACC.se[m]))==T))
    } else {mv<-m}
    
    # Depending on "opt.cri":
    if (opt.cri == "1se"){opt.M <- mv + 1
    }else { opt.M <- m + 1}
  }
  
  # Print a message indicating the number of optimal variables
  cat(paste("\n\n The optimal number of variables is:", opt.M, "\n\n"))
  
  if (!is.null(user_numVar)){
    opt.M <- user_numVar;
  }
  
  
  # Define NUM and DEN according to opt.M
  suppressMessages(BAL <- selbal.aux(x, y, th.imp = 0, covar = covar,
                                     logit.acc, logt=T, maxV = opt.M)) #diferencies logit.acc=logit.acc
  # Variables in the NUMERATOR and the DENOMINATOR
  NUM <- BAL[BAL[,2]=="NUM","Taxa"]
  DEN <- BAL[BAL[,2]=="DEN","Taxa"]
  
  # Information about the GLM
  # Final balance (number of components)
  k1 <- length(NUM); k2 <- length(DEN)
  
  # The final Balance
  FINAL.BAL <- sqrt((k1*k2)/(k1+k2))*(rowM(logc[,NUM])- rowM(logc[,DEN]))
  
  # Auxiliar data.frame for graphical representation
  U <- data.frame(dat, FINAL.BAL)
  colnames(U)[ncol(U)] <- "V1"
  # Regression model
  FIT.final <- glm(numy~., data=U, family = f.class)
  
  
  #------------------------------------------------------------------------------#
  #                           GRAPHICAL REPRESENTATION
  #------------------------------------------------------------------------------#
  
  # Build a data.frame with the information
  if(classy=="numeric"){
    df.boxplot <- data.frame(mean = ACC.mean, se = ACC.se, n =2:maxV)
    # Load library
    library(ggplot2)
    # The plot
    MSE.Boxplot <- ggplot(df.boxplot, aes(x=n, y=mean)) +
      geom_errorbar(aes(ymin=mean - se, ymax= mean + se),
                    width = 0.2, col = "gray") +
      geom_vline(xintercept = opt.M, linetype = "dotted",
                 col = "blue") +
      geom_point(color = "red", lwd=2) +
      theme_bw() +
      xlab("Number of variables") +
      ylab("Mean-Squared Error") +
      scale_x_continuous(breaks=seq(2,maxV,1)) +
      theme(strip.text.x = element_text(size=12, angle=0,
                                        face="bold",colour="white"),
            strip.text.y = element_text(size=12, face="bold"),
            strip.background = element_rect(colour="black",
                                            fill="black"),
            plot.title = element_text(size=20, vjust=2.25, hjust=0.5,
                                      face = "bold"),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank())
    
  }else{
    df.boxplot <- data.frame(mean = ACC.mean, se = ACC.se, n =2:maxV)
    ylabelName="Accuracy (AUC)";
    if (logit.acc=="Dev"){
      ylabelName="Explained Deviance";
    }
    # Load library
    library(ggplot2)
    # The plot
    MSE.Boxplot <- ggplot(df.boxplot, aes(x=n, y=mean)) +
      geom_errorbar(aes(ymin=mean - se, ymax= mean + se),
                    width = 0.2, col = "gray") +
      geom_vline(xintercept = opt.M, linetype = "dotted",
                 col = "blue") +
      geom_point(color = "red", lwd=2) +
      theme_bw() +
      xlab("Number of variables") +
      ylab(ylabelName) +
      scale_x_continuous(breaks=seq(2,maxV,1)) +
      theme(strip.text.x = element_text(size=12, angle=0,
                                        face="bold",colour="white"),
            strip.text.y = element_text(size=12, face="bold"),
            strip.background = element_rect(colour="black",
                                            fill="black"),
            plot.title = element_text(size=20, vjust=2.25, hjust=0.5,
                                      face = "bold"),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank())
  }
  
  
  
  # Extract information about the variables selected in the balances
  Sub.Balances <- lapply(Balances, function(x) x[1:min(nrow(x),opt.M),])
  # Complete the matrix with the variables appearing
  # Build the matrix
  BR<-matrix(0,nrow =n.fold*n.iter , ncol = length(x.nam))
  colnames(BR) <- x.nam
  # Complete the row of BAL.resume
  for (i in 1:length(Sub.Balances)){
    BR[ i, colnames(BR) %in% Sub.Balances[[i]][Sub.Balances[[i]][,"Group"]=="NUM","Taxa"]] <- 1
    BR[ i, colnames(BR) %in% Sub.Balances[[i]][Sub.Balances[[i]][,"Group"]=="DEN","Taxa"]] <- 2
  }
  
  # Complete the table
  ROB.TAB[,1] <- apply(BR!=0,2,function(x) 100*mean(x))
  ROB.TAB[,2] <- apply(BR==1,2,function(x) 100*mean(x))
  ROB.TAB[,3] <- apply(BR==2,2,function(x) 100*mean(x))
  
  # Variables with at least included in a balance
  fil1 <- which(ROB.TAB[,1]!=0)
  ord.ROB.TAB <- ROB.TAB[fil1,]
  # Order by the first row
  sel.ord <- order(ord.ROB.TAB[,1], decreasing = F)
  ord.ROB.TAB <- ord.ROB.TAB[sel.ord,]
  # Define a data.frame to plot the results
  BAL.SEL.TAB <- data.frame(name = row.names(ord.ROB.TAB),
                            sel = ord.ROB.TAB[,1])
  # Define the levels of the $name
  BAL.SEL.TAB$name <- factor(BAL.SEL.TAB$name, levels = BAL.SEL.TAB$name)
  # Define the color for variables in the numerator (as overall)
  COLOR.BAL <- rep(col[2],nrow(BAL.SEL.TAB))
  # If the variable is in the denominator modify the color:
  # Variables in the denominator
  vDEN <- row.names(ord.ROB.TAB)[which(ord.ROB.TAB[,"Prop_Denominator"]!=0)]
  COLOR.BAL[row.names(BAL.SEL.TAB) %in% vDEN] <- col[1]
  # Add COLOR.BAL to BAL.SEL.TAB
  BAL.SEL.TAB$COLOR.BAL <- factor(COLOR.BAL, levels = col, labels = col)
  
  
  
  
  #------------------------------------------#
  # Barplot representation
  #------------------------------------------#
  
  # Load library ggplot2
  suppressMessages(library(ggplot2))
  
  # IMP.plot
  IMP.plot <- ggplot(BAL.SEL.TAB, aes(x=factor(name), y=sel)) +
    geom_bar(stat="identity", aes(fill = COLOR.BAL),
             size=1) +
    guides(size = FALSE) + # Not to show the legend of the size
    scale_fill_manual(name = "Group of . . .",
                      values = c(col[1], col[2]),
                      breaks = c(col[1], col[2]),
                      labels=c("DEN","NUM")) +
    scale_color_manual(name ="Variables \n appearing in . . .",
                       values = c(col2,"white"),
                       breaks = c(col2,"white"),
                       labels = c("Both", "Global", "CV" ,"NONE"),
                       drop=F,
                       guide=guide_legend(
                         override.aes = list(fill="gray90"))) +
    ylab("% of times included in a balance") +
    xlab("") + theme_bw() +
    coord_flip() +
    ggtitle("Cross validation in balance selection") +
    theme(strip.text.x = element_text(size=12, angle=0,
                                      face="bold",colour="white"),
          strip.text.y = element_text(size=12, face="bold"),
          strip.background = element_rect(colour="black",
                                          fill="black"),
          plot.title = element_text(size=20, vjust=2.25, hjust=0.5,
                                    face = "bold"),
          legend.title = element_text(face="bold"),
          legend.text = element_text(face="bold"))
  
  
  #-----------------------------------#
  # Most repeated balances
  #-----------------------------------#
  
  # Balances' strings
  BAL.str <- apply(BR,1, function(x) paste(x, collapse=""))
  # Resume the information
  BAL.tab <- prop.table(table(BAL.str))
  # Names of appearing balances
  nam.str <- names(BAL.tab)
  # Values
  nam.A <- t(sapply(nam.str, FUN = function(x) unlist(strsplit(x,""))))
  # Variables included in the most abundant balances
  INC <- apply(nam.A, 1, function(x) x.nam[x!=0])
  # Variables included in the numerator of each selected balance
  INC.NUM <- alply(nam.A, 1, function(x) x.nam[x==1])
  # Variables included in the denominator of each selected balance
  INC.DEN <- alply(nam.A, 1, function(x) x.nam[x==2])
  
  # Variables selected
  UNIQUE.VAR <- unique(c(as.vector(unlist(INC)),NUM, DEN))
  # Build a data.frame to represent
  RESUME.BAL <- as.data.frame(matrix(0, nrow =length(UNIQUE.VAR),
                                     ncol = length(BAL.tab)))
  row.names(RESUME.BAL) <- UNIQUE.VAR
  
  # Put "NUM" if the variable is in the numerator of the balance
  RESUME.BAL[sapply(INC.NUM, function(x) UNIQUE.VAR %in% x)] <- "NUM"
  # Put "DEN" if the variable is in the denominator of the balance
  RESUME.BAL[sapply(INC.DEN, function(x) UNIQUE.VAR %in% x)] <- "DEN"
  
  # Add the relative frequency of the balances
  RESUME.BAL <- rbind(RESUME.BAL, FREQ=as.numeric(BAL.tab))
  
  
  # Add two new columns (one for Global Balance and another for percentages)
  RESUME.BAL <- cbind(RESUME.BAL, 0, 0)
  
  # Add the information of the global balance
  RESUME.BAL[row.names(RESUME.BAL)%in%NUM,ncol(RESUME.BAL)] <- "NUM"
  RESUME.BAL[row.names(RESUME.BAL)%in%DEN,ncol(RESUME.BAL)] <- "DEN"
  
  # NEW
  RESUME.BAL[-nrow(RESUME.BAL) ,ncol(RESUME.BAL) - 1] <-
    ROB.TAB[row.names(RESUME.BAL)[-nrow(RESUME.BAL)],1]
  
  # Order RESUME.BAL by FREQ
  RESUME.BAL <- RESUME.BAL[,c(ncol(RESUME.BAL), ncol(RESUME.BAL)-1,
                              order(RESUME.BAL[nrow(RESUME.BAL),
                                               -c(ncol(RESUME.BAL),
                                                  ncol(RESUME.BAL)-1)],
                                    decreasing = T))]
  
  # No frequency for the Global balance and the CV.Balance
  RESUME.BAL[nrow(RESUME.BAL),1:2]<- "-"
  
  
  # Data to plot (maximum 5 different balances)
  dat <- RESUME.BAL[,c(1,2:(min(5,ncol(RESUME.BAL))))]
  W <- which(apply(dat[,-2]==0,1,mean)==1)
  if(length(W)!=0){ dat <- dat[-as.numeric(W),]}
  # Change the order of first and second colum
  dat <- dat[,c(2,1,3:ncol(dat))]
  colnames(dat)[1:2]<-c("%","Global")
  
  # Order dat (rows ordered by their presence percentage)
  dat<-dat[c(order(as.numeric(dat[-nrow(dat),1]),decreasing=T),nrow(dat)),]
  
  
  #------------------------------------------------------------------------------#
  # GRAPHICAL REPRESENTATION OF MAIN BALANCES: global balance and cv - balance
  #------------------------------------------------------------------------------#
  
  # Define y for the plot
  ifelse(classy %in% c("numeric","integer"),
         y.plot <- y,
         y.plot <- 1:length(y))
  
  # PLOT GLOBAL
  PLOT.Global <- plot.bal(NUM,DEN,logc,y, covar, col = col, logit.acc)
  
  # Message starting the algorithm
  cat(paste("\n\n###############################################################",
            "\n . . . FINISHED.",
            "\n###############################################################"))
  
  # Build a list with the elements of interest
  L <- list(accuracy.nvar = MSE.Boxplot,
            var.barplot = IMP.plot,
            global.plot = PLOT.Global$Global.plot,
            global.plot2 = PLOT.Global$Global.plot2,
            ROC.plot = PLOT.Global$ROC.plot,
            cv.tab = dat,
            cv.accuracy = ACC.Matrix[(opt.M - 1),],
            global.balance = BAL,
            glm = FIT.final,
            opt.nvar = opt.M)
  
  return(L)
  
}

selbal.aux <- function(x, y, th.imp = 0, covar = NULL, logit.acc="AUC", positive=1,
                       logt=T, maxV = 1e10, zero.rep = "bayes"){
  
  #--------------------------------------------------------------------------#
  # STEP 0: load libraries and extract information
  #--------------------------------------------------------------------------#
  
  #----------------------------------------------#
  # 0.1: information about the response variable
  #----------------------------------------------#
  
  # Class of the response variable
  classy <- class(y)
  # Family for the glm (default: gaussian)
  f.class <- "gaussian"
  # numy to be y and it will be modified if y is a factor
  numy <- y
  # If y is a factor, compute the number of levels of y
  if (classy == "factor"){
    ylev <- levels(y)
    numy <- as.numeric(y) - 1
    f.class <- "binomial"
  }
  
  #------------------------------------------------------------------#
  # 0.2: information and transformation of the independent variables
  #------------------------------------------------------------------#
  
  # Load library
  suppressMessages(library(zCompositions))
  
  # Variables name
  var.nam <- rem.nam <- colnames(x)
  
  # Build a table with the response variable and covariates for correction
  if (!is.null(covar)){ dat <- data.frame(cbind(numy, covar))
  } else { dat <-data.frame(numy)}
  
  
  
  # The logCounts (with zero replacement)
  if (logt == F){ logCounts <- x
  } else{
    logCounts <- log(cmultRepl2(x, zero.rep = zero.rep))
  }
  
  
  #--------------------------------------------------------------------------#
  # 0.3: auxiliar functions
  #--------------------------------------------------------------------------#
  
  #--------------------------------------------------------------------------#
  # Auxiliar function to compute the first balance
  #--------------------------------------------------------------------------#
  
  first.bal<- function(logCounts, Y, covar=NULL){
    
    #------------------------------------------------------------------------#
    # STEP 0: extract information
    #------------------------------------------------------------------------#
    
    # Number and name of variables
    n <- ncol(logCounts)
    nam <- colnames(logCounts)
    
    
    #------------------------------------------------------------------------#
    # STEP 1: build the output matrix
    #------------------------------------------------------------------------#
    
    # The matrix
    if (classy=="factor"){ M<-matrix(0, nrow=n, ncol=n)
    }else{ M<-matrix(1e99, nrow=n, ncol=n)}
    # Row.names and colnames
    row.names(M)<-colnames(M)<-nam
    
    #------------------------------------------------------------------------#
    # STEP 2: complete the matrix
    #------------------------------------------------------------------------#
    
    if(classy=="factor"){
      
      # Solve the problem with libraries
      suppressWarnings(suppressMessages(library("CMA")))
      suppressMessages(detach("package:CMA", unload=TRUE))
      suppressMessages(library(pROC))
      
      for (i in 2:n){
        for (j in 1:(i-1)){
          # Build a table with the information
          TAB <- data.frame(cbind(Y,logCounts[,i]-logCounts[,j]))
          # Fit the regression model
          FIT <- glm(Y ~ .,data=TAB, family = f.class)
          
          # Add the value into the matrix
          ifelse(FIT$coefficients[2]>0,
                 M[i,j] <- logit.cor(FIT, y = y, covar = covar, logit.acc),#diferencies Y  covar
                 M[j,i] <- logit.cor(FIT, y = y, covar = covar, logit.acc)) #diferencies Y  covar
          
        } # End j
      } # End i
      
      # Indices for the highest logit.cor value
      r <- which(M == max(M), arr.ind = T)
      
      
    } else {
      for (i in 2:n){
        for (j in 1:(i-1)){
          # Build a table with the information
          TAB <- data.frame(cbind(Y,logCounts[,i]-logCounts[,j]))
          # Fit the regression model
          FIT <- glm(Y ~ .,data=TAB, family = f.class)
          # Complete the matrix
          ifelse(FIT$coefficients[2]>0,
                 M[i,j] <- mean(FIT$residuals^2),
                 M[j,i] <- mean(FIT$residuals^2))
        } # End j
      } # End i
      
      # Indices for the lowest MSE value
      r <- which(M == min(M), arr.ind = T)
    }
    
    
    # Return the row and column of the maximum value
    return(r)
  }
  
  #--------------------------------------------------------------------------#
  
  #--------------------------------------------------------------------------#
  # Auxiliar function to compute the "association value" when adding a new
  # variable into the balance
  #--------------------------------------------------------------------------#
  
  balance.adding.cor <- function(x, LogCounts, POS, NEG, numy, covar=NULL){
    
    #----------------------------------------#
    # If x added into the numerator, . .
    #----------------------------------------#
    
    # The "numerator"
    S1.pos <- rowM(LogCounts[,c(POS,x)]); s1 <- length(POS) + 1
    # The "denominator"
    S2.pos <- rowM(LogCounts[,NEG])     ; s2 <- length(NEG)
    # The balance
    BAL <- sqrt((s1*s2)/(s1+s2))*(S1.pos - S2.pos)
    
    # Data.frame with the variables
    D.pos <- data.frame(cbind(numy, BAL))
    
    # Regression model
    FIT.pos <- glm(numy~., data=D.pos, family=f.class)
    # The MSE or the corresponding value for dichotomous responses
    if(classy=="numeric"){ C.pos <- mean(FIT.pos$residuals^2)
    #}else{ C.pos <- logit.cor(FIT.pos,numy,covar = covar, logit.acc)}#diferencies covar
    }else{ C.pos <- logit.cor(FIT.pos,y,covar = covar, logit.acc)}#diferencies covar
    
    #----------------------------------------#
    # If x added into the numerator, . .
    #----------------------------------------#
    
    # The numerator
    S1.neg <- rowM(LogCounts[,POS])       ; s1 <- length(POS)
    # The denominator
    S2.neg <- rowM(LogCounts[,c(NEG,x)])  ; s2 <- length(NEG) + 1
    # The balance
    BAL <- sqrt((s1*s2)/(s1+s2))*(S1.neg - S2.neg)
    
    # Data.frame with the variables
    D.neg <- data.frame(cbind(numy, BAL))
    
    # Regression model
    FIT.neg <- glm(numy~., data=D.neg, family=f.class)
    # The MSE or the corresponding value for dichotomous responses
    if(classy=="numeric"){ C.neg <- mean(FIT.neg$residuals^2)
    #}else{ C.neg <- logit.cor(FIT.neg,numy,covar = covar, logit.acc)}  #diferencies covar
    }else{ C.neg <- logit.cor(FIT.neg,y,covar = covar, logit.acc)}  #diferencies covar
    # Correlation values
    COR <- c(C.pos, C.neg)
    # Return the values
    return(COR)
  }
  
  #------------------------------------------------------------------------------#
  
  
  
  #--------------------------------------------------------------------------#
  # STEP 1: depending on the response variable class, . . .
  #--------------------------------------------------------------------------#
  
  # Define the first balance
  A1 <- first.bal(logCounts, Y = numy, covar=covar)
  # Variables taking parti into the first balance
  POS <- colnames(x)[A1[1,1]]
  NEG <- colnames(x)[A1[1,2]]
  
  # Included variables in the model
  INC.VAR <- c(POS, NEG)
  # Delete these variables from rem.nam
  rem.nam <- setdiff(rem.nam, INC.VAR)
  
  # Define the initial balance (B_1)
  S1 <- logCounts[,POS]
  S2 <- logCounts[,NEG]
  # Assign the values to B
  B <- sqrt(1/2)*(S1 - S2)
  
  #--------------------------------------------------------------------------#
  # Information about the ACC for the Balance values
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  # NEW: A table with the included variables and the group
  #--------------------------------------------------------------------------#
  
  Tab.var<- data.frame(Taxa = c(POS,NEG), Group = c("NUM", "DEN"))
  Tab.var[,1]<-as.character(Tab.var[,1])
  
  
  # Build a new data.frame
  dat.ini <- cbind(dat, B)
  # Fit the regression model
  FIT.initial <- glm(numy ~ .,data=dat.ini, family = f.class)
  
  # Solve the problem with libraries
  suppressWarnings(suppressMessages(library("CMA")))
  suppressMessages(detach("package:CMA", unload=TRUE))
  suppressMessages(library(pROC))
  
  # Define the initial "accuracy" or "association" value
  if(classy=="numeric"){ ACC.Bal <- mean(FIT.initial$residuals^2)
  #}else{ ACC.Bal <- logit.cor(FIT.initial, numy, covar = covar, logit.acc)}#diferencies covar
  }else{ ACC.Bal <- logit.cor(FIT.initial, y, covar = covar, logit.acc)}#diferencies covar
  
  
  #----------------------------------------------------------------------------#
  
  # ACC reference
  ACC.ref <- ACC.Bal
  
  #------------------------------------------------------------------------#
  # Improve the balances
  #------------------------------------------------------------------------#
  # Define some parameters
  # The p.value to compare 2 balances (one of them with an additional
  # variable)
  ACC.set <- ACC.ref
  
  # Index of the number of variables for the balance
  nV <- 2
  
  
  #------------------------------#
  # For numeric responses, . . .
  #------------------------------#
  
  if (classy=="numeric"){
    
    # While there is an improvement and the maximun number of variables has not
    # been reached, . . .
    while (ACC.set <= ACC.ref && length(rem.nam)!=0 && nV<maxV){
      
      # The new p.bal.ref is the p.set of the previous step
      ACC.ref <- ACC.set
      
      # Function to extract the p-value
      add2bal.ACC <- matrix(,nrow = 0, ncol = 2)
      
      # Solve the problem with libraries
      suppressWarnings(suppressMessages(library("CMA")))
      suppressMessages(detach("package:CMA", unload=TRUE))
      suppressMessages(library(pROC))
      
      
      # Extract the p-values
      add2bal.ACC <- t(sapply(rem.nam, function(x)
        balance.adding.cor(x, LogCounts = logCounts, POS, NEG, numy = numy,
                           covar = covar)))
      # Add names to the rows
      row.names(add2bal.ACC) <- rem.nam
      
      
      # Extract which is the variable (only the first row)
      ACC.opt <- which(add2bal.ACC==min(add2bal.ACC),arr.ind = T)[1,]
      # Modify p.set
      ACC.set <- min(add2bal.ACC)
      
      
      # If there is an improvement, . . .
      #if (abs(ACC.set - ACC.ref) > th.imp){
      if ((ACC.set - ACC.ref) < th.imp){
        INC.VAR <- c(INC.VAR, rem.nam[ACC.opt[1]])
        nV <- nV + 1
        if (ACC.opt[2]==1){
          POS <- c(POS, rem.nam[ACC.opt[1]])
          Tab.var <- rbind(Tab.var, c(rem.nam[ACC.opt[1]], "NUM"))
        } else if (ACC.opt[2]==2){
          NEG <- c(NEG, rem.nam[ACC.opt[1]])
          Tab.var <- rbind(Tab.var, c(rem.nam[ACC.opt[1]], "DEN"))
        } else {ACC.set <- 0 }
        
        # Remainng variables (possible to add to the balance)
        rem.nam <- rem.nam[-ACC.opt[1]]
      }
      
    } # End while
    
  }else{
    
    #-----------------------------------#
    # For non-numeric responses, . . .
    #-----------------------------------#
    
    # While there is an improvement and the maximun number of variables has not
    # been reached, . . .
    while (ACC.set >= ACC.ref && length(rem.nam)!=0 && nV<maxV){
      
      # The new p.bal.ref is the p.set of the previous step
      ACC.ref <- ACC.set
      
      # Function to extract the p-value
      add2bal.ACC <- matrix(,nrow = 0, ncol = 2)
      
      # Solve the problem with libraries
      suppressWarnings(suppressMessages(library("CMA")))
      suppressMessages(detach("package:CMA", unload=TRUE))
      suppressMessages(library(pROC))
      
      
      # Extract the p-values
      add2bal.ACC <- t(sapply(rem.nam, function(x)
        balance.adding.cor(x, LogCounts = logCounts, POS, NEG, numy = numy,
                           covar = covar)))
      # Add names to the rows
      row.names(add2bal.ACC) <- rem.nam
      
      
      # Extract which is the variable (only the first row)
      ACC.opt <- which(add2bal.ACC==max(add2bal.ACC),arr.ind = T)[1,]
      # Modify p.set
      ACC.set <- max(add2bal.ACC)
      
      # If there is an improvement, . . .
      if ((ACC.set - ACC.ref) > th.imp){
        INC.VAR <- c(INC.VAR, rem.nam[ACC.opt[1]])
        nV <- nV + 1
        if (ACC.opt[2]==1){
          POS <- c(POS, rem.nam[ACC.opt[1]])
          Tab.var <- rbind(Tab.var, c(rem.nam[ACC.opt[1]], "NUM"))
        } else if (ACC.opt[2]==2){
          NEG <- c(NEG, rem.nam[ACC.opt[1]])
          Tab.var <- rbind(Tab.var, c(rem.nam[ACC.opt[1]], "DEN"))
        } else {ACC.set <- 0 }
      } # End if
      
      # Remainng variables (possible to add to the balance)
      rem.nam <- rem.nam[-ACC.opt[1]]
      
    }
  }
  
  # K1 and k2
  k1 <- length(POS); k2 <- length(NEG)
  
  # The final Balance
  FINAL.BAL <- sqrt((k1*k2)/(k1+k2))*
    (rowM(logCounts[,POS])- rowM(logCounts[,NEG]))
  
  return(Tab.var)
}

# Define the function logit.cor
logit.cor <- function(FIT, y, covar = NULL,logit.acc, positive=1){
  if (logit.acc == "AUC"){
    if (class(y) == "factor") {numy<-as.numeric(y)-1} else {numy<-y} #new diferences
    d <- as.numeric(auc(numy, FIT$fitted.values, quiet=TRUE)) # new diferences
  } else if (logit.acc == "Rsq"){
    d <- cor(as.numeric(y), FIT$fitted.values)^2
  } else if (logit.acc == "Tjur"){
    if (class(y) == "factor") {numy<-as.numeric(y)-1} else {numy<-y}
    d <- mean(FIT$fitted.values[numy==1]) - mean(FIT$fitted.values[numy==0])
  } else if (logit.acc == "Dev"){
    f.class <- ifelse (class(y) == "factor", "binomial", "gaussian")
    if (class(y) == "factor") {numy<-as.numeric(y)-1} else {numy<-y}
    #if (is.null(covar)){
    d<-1-(deviance(FIT)/deviance(glm(numy~1,family=f.class)))  # proportion of explained deviance
  } else if (logit.acc == "f1"){
    d <- F1_Score(y, round(FIT$fitted.values), positive=1)
  } else if (logit.acc == "accuracy"){
    d <- mean(y == round(FIT$fitted.values))
  }
  
  # Return the value
  return(d)
}
