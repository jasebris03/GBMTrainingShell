#libraries required
library(gbm)
library(dplyr)
library(data.table)

# functions
mcloss <- function (y_actual, y_pred) {
        dat <- rep(0, length(y_actual))
        for(i in 1:length(y_actual)){
                dat_x <- y_pred[i,y_actual[i,1],1]
                dat[i] <- max(1e-15,min(1-1e-15,dat_x))
        }
        return(-sum(log(dat))/length(y_actual))
}

stratified <- function(df, group, size, select = NULL, 
                       replace = FALSE, bothSets = FALSE) {
        if (is.null(select)) {
                df <- df
        } else {
                if (is.null(names(select))) stop("'select' must be a named list")
                if (!all(names(select) %in% names(df)))
                        stop("Please verify your 'select' argument")
                temp <- sapply(names(select),
                               function(x) df[[x]] %in% select[[x]])
                df <- df[rowSums(temp) == length(select), ]
        }
        df.interaction <- interaction(df[group], drop = TRUE)
        df.table <- table(df.interaction)
        df.split <- split(df, df.interaction)
        if (length(size) > 1) {
                if (length(size) != length(df.split))
                        stop("Number of groups is ", length(df.split),
                             " but number of sizes supplied is ", length(size))
                if (is.null(names(size))) {
                        n <- setNames(size, names(df.split))
                        message(sQuote("size"), " vector entered as:\n\nsize = structure(c(",
                                paste(n, collapse = ", "), "),\n.Names = c(",
                                paste(shQuote(names(n)), collapse = ", "), ")) \n\n")
                } else {
                        ifelse(all(names(size) %in% names(df.split)),
                               n <- size[names(df.split)],
                               stop("Named vector supplied with names ",
                                    paste(names(size), collapse = ", "),
                                    "\n but the names for the group levels are ",
                                    paste(names(df.split), collapse = ", ")))
                }
        } else if (size < 1) {
                n <- round(df.table * size, digits = 0)
        } else if (size >= 1) {
                if (all(df.table >= size) || isTRUE(replace)) {
                        n <- setNames(rep(size, length.out = length(df.split)),
                                      names(df.split))
                } else {
                        message(
                                "Some groups\n---",
                                paste(names(df.table[df.table < size]), collapse = ", "),
                                "---\ncontain fewer observations",
                                " than desired number of samples.\n",
                                "All observations have been returned from those groups.")
                        n <- c(sapply(df.table[df.table >= size], function(x) x = size),
                               df.table[df.table < size])
                }
        }
        temp <- lapply(
                names(df.split),
                function(x) df.split[[x]][sample(df.table[x],
                                                 n[x], replace = replace), ])
        set1 <- do.call("rbind", temp)
        
        if (isTRUE(bothSets)) {
                set2 <- df[!rownames(df) %in% rownames(set1), ]
                list(SET1 = set1, SET2 = set2)
        } else {
                set1
        }
}

# main code
# read in data into dataframe
train <- read.csv("~/Kaggle/otto-group-product-classification-challenge/train.csv", stringsAsFactors=FALSE)
# create dataframe without the 'id' component
df <- train[,-1]
# convert target column to factors
df$target <- as.factor(df$target)

set.seed(1234)
sets <- df %>%
        group_by(target) %>%
        filter(length(target) > 1) %>%
        stratified("target", .7, bothSets = T)

train_df <- as.data.frame(sets$SET1)
test_df <- as.data.frame(sets$SET2)

# setup attribute grid
grid <- expand.grid(hidden=c(c(10),c(5,5),c(5,5,5)),learningrate=c(0.8,0.6),momentum=c(0.5,0.25))
grid_results <- matrix(rep(0,dim(grid)[1] * (dim(grid)[2] + 2)),dim(grid)[1])
# setup resampling sets
number_of_sampling_sets <- 5


for(grid_id in 1:nrow(grid)){
        set.seed(1) # reset seed as we want each set of sampling to have the same ids otherwise you're comparing apples with genetically modified gorillas
        sample_sets <- matrix(rep(0,number_of_sampling_sets * 2),number_of_sampling_sets)
        for(sample_id in 1:number_of_sampling_sets){
                # randomly choose in-sample data (80% of training data)
                sets <- train_df %>%
                        group_by(target) %>%
                        filter(length(target) > 1) %>%
                        stratified("target", .8, bothSets = T)
                
                in_sample <- as.data.frame(sets$SET1)
                out_sample <- as.data.frame(sets$SET2)
                
                # setup model with current parameters
                fit <- gbm(target ~ ., data=in_sample, n.trees = grid[grid_id,]$n.trees, interaction.depth =  grid[grid_id,]$interaction.depth, shrinkage = grid[grid_id,]$shrinkage, distribution = "multinomial",verbose=T)
                preds <- predict(fit,newdata=out_sample,n.trees=10,type="response")
                obs <- as.data.frame(out_sample$target)
                multiclassLoss <- mcloss(y_actual = obs, y_pred = preds)
                sample_sets[sample_id,] <- cbind(sample_id,multiclassLoss)
                #cat(sample_id,": ",multiclassLoss,"\n")
        }
        avg_performance <- mean(sample_sets[,2])
        grid_results[grid_id,] <- as.numeric(cbind(grid_id, grid[grid_id,],avg_performance))
        cat(grid_id,":\t",grid[grid_id,]$n.trees,"\t",grid[grid_id,]$interaction.depth,"\t",grid[grid_id,]$shrinkage,"\t",avg_performance,"\n")
}

best <- grid_results[order(grid_results[,5]),][1,]

best_fit <- gbm(target ~ ., data = train_df, n.trees = best[2], interaction.depth = best[3], shrinkage = best[4], distribution = "multinomial",verbose=T)
# test on all train data
preds <- predict(fit,newdata=train_df,n.trees=10,type="response")
obs <- as.data.frame(train_df$target)
multiclassLoss <- mcloss(y_actual = obs, y_pred = preds)

# test on test data
preds <- predict(fit,newdata=test_df,n.trees=10,type="response")
obs <- as.data.frame(test_df$target)
multiclassLoss <- mcloss(y_actual = obs, y_pred = preds)