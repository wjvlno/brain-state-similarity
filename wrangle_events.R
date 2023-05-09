# packages (some, not all, are needed) ####
require(ggplot2)
require(lme4)
require(lmerTest)
require(reshape2)
require(psych)
require(sjPlot)
require(knitr)
require(tidyverse)
require(tidybayes)
require(beepr)
require(ggsci)
require(reghelper)
require(stringr)
require(sjstats)

# gather data ####
# exam-related variables (e.g., grades, grade predictions, prediction errors)
exam_data <- read.csv("C:/Users/User/Downloads/all_scan_grades_preds_pes.csv")

## event index files ####
bounds_dir <- "C:/Users/User/Downloads/triton_patterns_indices_01142023"
setwd(bounds_dir)
event_indices.files <- list.files(pattern = "*event_indices*")

evpat_dir <- "C:/Users/User/Downloads/triton_patterns_indices_01142023"
setwd(evpat_dir)
event_pat.files <- list.files(pattern = "*event_pat*")

# # sort by scan portion (first half, second half, full scan)
# event_indices.files.half1 <- event_indices.files[str_detect(event_indices.files, "b0_e400")]
# event_indices.files.half2 <- event_indices.files[str_detect(event_indices.files, "b400_e800")]
# event_indices.files.full <- event_indices.files[str_detect(event_indices.files, "b0_e891")]

# pull only full segmentations
event_indices.files.full <- str_subset(event_indices.files, "891")
event_pat.files.full <- str_subset(event_pat.files, "891")

# preprocess event patterns
fulldf <- c()
file <- event_pat.files.full[1]
for (file in event_pat.files.full) { 
  
  f <- read.csv(paste0(evpat_dir,"/", file), header = F)
  f <- f[2:nrow(f),2:ncol(f)]
  # f <- t(f)
  colnames(f) <- paste0("vertex_",as.numeric(str_sub(colnames(f),2)) - 1)
  f$event_index <- seq(1, nrow(f)) # as.numeric(rownames(f)) - 1

  f$filename <- file
  f$id <- str_extract(file, "(?<=s)(.*?)(?=_)")
  f$exam <- as.numeric(str_sub(str_extract(file, "(?<=ex)(.*?)(?=_)"),3))
  
  fulldf <- rbind(fulldf, f)
}
ev_pat.df <- fulldf

# add event timing
# id <- c()
# exam <- c()
# k <- c()
# run <- c()
# filename <- c()

# fix event index files
dir <- "C:/Users/User/Downloads/triton_outputs_12152022"
setwd(dir)
pred_seg.files <- list.files(pattern = "*pred_seg*")
pred_seg.files <- str_subset(pred_seg.files, "891")

pred_seg.df <- c()
for (file in pred_seg.files) { # event_indices.files.full
  # read file
  f <- read.csv(paste0(dir,"/", file), header = F)

  if (str_detect(file, "run")) {
    run <- as.numeric(str_extract(file, "(?<=run)(.*?)(?=_)"))
  } else {
    run <- 1
  }
  
  f <- f[-c(1),-c(1)]
  start <- c(1)
  for (i in 2:ncol(f)) {
    start <- c(start, which(f[,i] > f[,i-1])[which(which(f[,i] > f[,i-1]) > start[i-1])][1])
  }
  
  run.df <- data.frame(id = str_extract(file, "(?<=s)(.*?)(?=_)"), exam = str_extract(file, "(?<=ex)(.*?)(?=_)"), event_index = seq(1,ncol(f)), start = start, end = c(start[2:length(start)], nrow(f)))
  run.df$duration <- run.df$end - run.df$start
  run.df$id_num <- as.numeric(run.df$id)
  run.df$exam_num <- as.numeric(run.df$exam)
  run.df$k <- ncol(f)
  
  run.df$grade <- c(exam_data$Grade[which(exam_data$id == run.df$id_num[1] & exam_data$Exam == run.df$exam_num[1])],NA)[1]
  run.df$pred <- c(exam_data$prediction[which(exam_data$id == run.df$id_num[1] & exam_data$Exam == run.df$exam_num[1])],NA)[1]
  run.df$pe <- c(exam_data$pe[which(exam_data$id == run.df$id_num[1] & exam_data$Exam == run.df$exam_num[1])], NA)[1]
  
  run.df$run <- run

  pred_seg.df <- rbind(pred_seg.df, run.df)

}

pred_seg.df$rm <- 0
for (i in unique(pred_seg.df$id_num)) {
  for (j in unique(pred_seg.df$exam_num[which(pred_seg.df$id_num == i)])) {
    pred_seg.df$rm[which(pred_seg.df$id_num == i & 
                           pred_seg.df$exam_num == j & 
                           pred_seg.df$run != max(pred_seg.df$run[which(pred_seg.df$id_num == i & 
                                                                          pred_seg.df$exam_num == j)]))] <- 1
  }
}
pred_seg.df <- pred_seg.df[-c(which(pred_seg.df$rm == 1)),]

# remove exam 0
pred_seg.df.exams <- pred_seg.df[-c(which(pred_seg.df$exam_num == 0)),]

# add event patterns
newdf <- c()
for (i in 1:nrow(pred_seg.df.exams)) {
  print(paste0(round(i/nrow(pred_seg.df.exams)*100,2)))
  newdf <- rbind(newdf, ev_pat.df[which(ev_pat.df$id == pred_seg.df.exams$id[i] & ev_pat.df$exam == pred_seg.df.exams$exam_num[i]),1:400][pred_seg.df.exams$event_index[i],])
}
dim(newdf)
dim(pred_seg.df.exams)
newdf.exams <- cbind(pred_seg.df.exams, newdf)

# add temporal regressors
newdf.exams$pre_reveal <- 1
newdf.exams$pre_reveal[which(newdf.exams$start < 295)] <- 0
newdf.exams$post_reveal <- 0
newdf.exams$post_reveal[which(newdf.exams$start >= 295)] <- 1

newdf.exams$t_pre_reveal <- 0
newdf.exams$t_pre_reveal[which(newdf.exams$start < 295)] <- newdf.exams$start[which(newdf.exams$start < 295)] - 295
newdf.exams$t_post_reveal <- 0
newdf.exams$t_post_reveal[which(newdf.exams$start >= 295)] <- newdf.exams$start[which(newdf.exams$start >= 295)] - 295

newdf.exams$tminus_reveal <- 0
newdf.exams$tminus_reveal[which(newdf.exams$start < 295)] <- newdf.exams$start[which(newdf.exams$start < 295)] - 295
newdf.exams$tminus_reveal[which(newdf.exams$start >= 295)] <- newdf.exams$start[which(newdf.exams$start >= 295)] - 295

# add CBIG vertex labels ####
lh_labels <- read.csv("C:/Users/User/Downloads/CBIG_lh_400parc_17net_labels.csv")
rh_labels <- read.csv("C:/Users/User/Downloads/CBIG_rh_400parc_17net_labels.csv")

lh_labels <- lh_labels[2:nrow(lh_labels),]
rh_labels <- rh_labels[2:nrow(rh_labels),]

lh_labels$hemi <- "left"
rh_labels$hemi <- "right"

all_labels <- rbind(lh_labels, rh_labels)

all_labels$parcel_name_full <- all_labels$parcel_name

# remove numbers after last "_" in label
all_labels$parcel_name <- sub("_[^_]+$", "", all_labels$parcel_name_full)

all_labels$parcel_id_lvl2 <- NA
ct <- 1
for (i in unique(all_labels$parcel_name)) {
  all_labels$parcel_id_lvl2[which(all_labels$parcel_name == i)] <- ct
  ct <- ct + 1
}

# remove character after last "_" in label
all_labels$subnetwork_name <- sub("_[^_]+$", "", all_labels$parcel_name)

# save intermediate file for analyses
write.csv(all_labels, "C:/Users/User/Downloads/CBIG_bilat_400parc_17net_labels.csv")