# packages
require(reshape2)
require(ggplot2)
require(pspline)
require(zoo)
require(xts)
require(ksmooth)

############################
# intermediate preprocessing
############################

# load intermediate file for analyses
newdf.exams.m <- read.csv("C:/Users/User/Downloads/ESP_event_patterns_17net_400parc.csv")

# additional processing
newdf.exams.m$hemisphere <- NA
newdf.exams.m$hemisphere[which(newdf.exams.m$variable < 201)] <- "L"
newdf.exams.m$hemisphere[which(newdf.exams.m$variable >= 201)] <- "R"

newdf.exams.m$vertex_label <- paste0(newdf.exams.m$hemisphere, "_", newdf.exams.m$vertex_label)
newdf.exams.m$subnetwork_label <- paste0(newdf.exams.m$hemisphere, "_", newdf.exams.m$subnetwork_label)
newdf.exams.m$network_label <- paste0(newdf.exams.m$hemisphere, "_", newdf.exams.m$network_label)

# make event boundary summary file (bound.df)
ct <- 1
for (i in unique(newdf.exams.m$id)) {
  
  for (j in unique(newdf.exams.m$exam[which(newdf.exams.m$id == i)])) {
    
    dat <- newdf.exams.m[which(newdf.exams.m$id == i & newdf.exams.m$exam == j),][,4:6]
    dat.bounds.num <- rep(0, 891)
    dat.bounds.bin <- rep(0, 891)
    
    for (k in unique(dat$event_index)) {
      dat.bounds.num[dat$start[which(dat$event_index == k)][1]:dat$end[which(dat$event_index == k)][1]] <- k
      dat.bounds.bin[dat$start[which(dat$event_index == k)][1]] <- 1
    }
    
    if (ct == 1) {
      bound.df <- data.frame(id = rep(i,891), exam = rep(j,891), index = dat.bounds.num, bound = dat.bounds.bin)
    } else {
      bound.df <- rbind(bound.df, data.frame(id = rep(i,891), exam = rep(j,891), index = dat.bounds.num, bound = dat.bounds.bin))
    }
    ct <- ct + 1
  }
}

# import physiological data (heart rate, skin conductance)
physio.cat <- read.csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/WVillano/premed_physio_w_predictors.csv")
physio.cat <- physio.cat[which(physio.cat$end_time <= 1800),]

# match preprocessed 4D data by removing first 5 2-second segments
physio.cat$segment_num <- physio.cat$segment_num - 5
physio.cat <- physio.cat[which(physio.cat$segment_num <= 891 & physio.cat$segment_num > 0),]
physio.cat <- physio.cat[which(physio.cat$exam > 0),]
physio.cat$start_time <- physio.cat$start_time - 10
physio.cat$end_time <- physio.cat$end_time - 10

# preprocess physio data / add event boundary labels, indices
physio.cat$event_index <- NA
physio.cat$event_bound <- NA
for (i in unique(physio.cat$id)) {
  for (j in unique(physio.cat$exam)) {
    if (length(bound.df$index[which(bound.df$id == i & bound.df$exam == j)]) == 0) {
      next
    }
    physio.cat$event_index[which(physio.cat$id == i & physio.cat$exam == j)] <- bound.df$index[which(bound.df$id == i & bound.df$exam == j)]
    physio.cat$event_bound[which(physio.cat$id == i & physio.cat$exam == j)] <- bound.df$bound[which(bound.df$id == i & bound.df$exam == j)]
  }
}

# calculate first derivative of physio metrics: heart rate, mean skin conductance
physio.cat$mean_sc_zsub_deriv <- NA
physio.cat$mean_hr_zsub_deriv <- NA
for (i in unique(physio.cat$id)) {
  print(i)
  for (j in unique(physio.cat$exam)) {
    sc <- na.locf(physio.cat$mean_sc[which(physio.cat$id == i & physio.cat$exam == j)], na.rm = FALSE, fromLast = T)
    hr <- na.locf(physio.cat$mean_hr[which(physio.cat$id == i & physio.cat$exam == j)], na.rm = FALSE, fromLast = T)
    sc <- na.locf(sc, na.rm = FALSE, fromLast = F)
    hr <- na.locf(hr, na.rm = FALSE, fromLast = F)
    
    t <- physio.cat$segment_num[which(physio.cat$id == i & physio.cat$exam == j)]
    if (length(t) == 0) {
      next
    }
    
    physio.cat$mean_sc_zsub_deriv[which(physio.cat$id == i & physio.cat$exam == j)] <- predict(sm.spline(t, sc), t, 1)
    physio.cat$mean_hr_zsub_deriv[which(physio.cat$id == i & physio.cat$exam == j)] <- predict(sm.spline(t, hr), t, 1)
  }
}

# convolve binary event boundaries w/ gaussian kernel (minimal smoothing to account for TR length in event boundary locations)
physio.cat$event_bound_smth <- NA
for (i in unique(physio.cat$id)) { 

  for (j in unique(physio.cat$exam)) {
    t <- physio.cat$segment_num[which(physio.cat$id == i & physio.cat$exam == j)]
    
    if (length(t) == 0) {
      next
    }

    physio.cat$event_bound_smth[which(physio.cat$id == i & physio.cat$exam == j)] <- 
      ksmooth(time(physio.cat$event_bound[which(physio.cat$id == i & physio.cat$exam == j)]),physio.cat$event_bound[which(physio.cat$id == i & physio.cat$exam == j)],'normal',bandwidth=3)$y
  }  
}

# additional preprocessing (add continuous tally variable to index unique scans)
physio.cat$tally <- NA
ct <- 1
for (i in unique(physio.cat$id)) {
  for (j in unique(physio.cat$exam[which(physio.cat$id == i)])) {
    physio.cat$tally[which(physio.cat$id == i & physio.cat$exam == j)] <- ct
    ct <- ct + 1
  }
}

############################
# analysis branch 1 ####
############################

# remove all events that begin after TR=600
newdf.exams.m.t <- newdf.exams.m[which(newdf.exams.m$start < 800),2:ncol(newdf.exams.m)]

# reshape to wide 
event_sim.df <- newdf.exams.m.t[,c("id","exam","event_index","start","end","vertex_label","value", "hemisphere","grade","pred", "pe")]
event_sim.df.dc <- dcast(event_sim.df, id + exam + event_index + start + end + grade + pred + pe ~ vertex_label, value.var="value")

# add temporal variables
event_sim.df.dc.test$t_reveal <- event_sim.df.dc.test$start - 295
event_sim.df.dc.test$t_reveal[which(event_sim.df.dc.test$t_reveal < 0)] <- 0
event_sim.df.dc.test$t_reveal[which(event_sim.df.dc.test$t_reveal > 0)] <- event_sim.df.dc.test$t_reveal[which(event_sim.df.dc.test$t_reveal > 0)]*-1

# add average physio during event
event_sim.df.dc$mean_hr_zsub <- NA
event_sim.df.dc$mean_sc_zsub <- NA
for (i in unique(event_sim.df.dc$id)) {
  for (j in unique(event_sim.df.dc$exam[which(event_sim.df.dc$id == i)])) {
    for (k in event_sim.df.dc$event_index[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j)]) {
      s <- event_sim.df.dc$start[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$event_index == k)]
      e <- event_sim.df.dc$end[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$event_index == k)]
      
      event_sim.df.dc$mean_hr_zsub[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$event_index == k)] <- 
        mean(physio.cat$mean_hr_zsub[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num >= s & physio.cat$segment_num <= e)], na.rm = T)
      event_sim.df.dc$mean_sc_zsub[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$event_index == k)] <- 
        mean(physio.cat$mean_sc_zsub[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num >= s & physio.cat$segment_num <= e)], na.rm = T)
    }
  }
}

# correlation between grade-reveal state for each exam and all subsequent states within and across exams
event_sim.df.dc$exam_1_corr <- NA
event_sim.df.dc$exam_2_corr <- NA
event_sim.df.dc$exam_3_corr <- NA
event_sim.df.dc$exam_4_corr <- NA
for (i in unique(event_sim.df.dc$id)) {
  for (j in unique(event_sim.df.dc$exam[which(event_sim.df.dc$id == i)])) {
    s <- which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$start > 295)[1]
    e <- which(event_sim.df.dc$id == i)[length(which(event_sim.df.dc$id == i))]
    event_sim.df.dc[[paste0("exam_",j,"_corr")]][s:e] <- cor(t(event_sim.df.dc[s:e,9:(ncol(event_sim.df.dc) - 6)]))[1,]
  }
}
colnames(event_sim.df.dc)
crosscorr.df <- event_sim.df.dc[which(!is.na(event_sim.df.dc$exam_1_corr)), c("id","exam","event_index","start","end","grade","pred","pe", "exam_1_corr", "exam_2_corr", "exam_3_corr", "exam_4_corr", "mean_sc_zsub", "mean_hr_zsub")]

# build intermediate dataframe for representational similarity analysis
crosscorr.df.test <- event_sim.df.dc[, c("id","exam","event_index","start","end","grade","pred", "pe", "exam_1_corr", "exam_2_corr", "exam_3_corr", "exam_4_corr", "mean_sc_zsub", "mean_hr_zsub")]
crosscorr.df.test$t_reveal <- crosscorr.df.test$start - 295

bindcor.df <- c()
for (i in unique(crosscorr.df.test$id)) {
  for (j in unique(crosscorr.df.test$exam[which(crosscorr.df.test$id == i)])) {
    bindcor.df <- rbind(bindcor.df, crosscorr.df.test[which(crosscorr.df.test$id == i & crosscorr.df.test$exam == j & crosscorr.df.test$t_reveal > 0)[1],])
  }
}

# combine behavioral and neural data
comp.df <- data.frame(id = NA, x = c(1,1,1,2,2,3), y = c(2,3,4,3,4,4), corr = NA, x_grade = NA, y_grade = NA, x_pred = NA, y_pred = NA, x_pe = NA, y_pe = NA, x_sc = NA, y_sc = NA, x_hr = NA, y_hr = NA)
comp.df.full <- c()
for (i in unique(bindcor.df$id)) {
  print(i)
  bindcor.df[which(bindcor.df$id == i),grep("*corr",colnames(bindcor.df))][2:4,]
  
  comp.df.sub <- comp.df
  comp.df.sub$id <- i
  for (j in 1:nrow(comp.df)) {
    if (nrow(bindcor.df[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j]),]) == 0) {
      next
    }
    comp.df.sub$corr[j] <- c(bindcor.df[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j]),][[paste0("exam_",comp.df$x[j],"_corr")]],NA)[1]
    comp.df.sub$x_grade[j] <- c(bindcor.df$grade[which(bindcor.df$id == i & bindcor.df$exam == comp.df$x[j])],NA)[1]
    comp.df.sub$y_grade[j] <- c(bindcor.df$grade[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j])],NA)[1]
    comp.df.sub$x_pred[j] <- c(bindcor.df$pred[which(bindcor.df$id == i & bindcor.df$exam == comp.df$x[j])],NA)[1]
    comp.df.sub$y_pred[j] <- c(bindcor.df$pred[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j])],NA)[1]
    comp.df.sub$x_pe[j] <- c(bindcor.df$pe[which(bindcor.df$id == i & bindcor.df$exam == comp.df$x[j])],NA)[1]
    comp.df.sub$y_pe[j] <- c(bindcor.df$pe[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j])],NA)[1]
    comp.df.sub$x_sc[j] <- c(bindcor.df$mean_sc_zsub[which(bindcor.df$id == i & bindcor.df$exam == comp.df$x[j])],NA)[1]
    comp.df.sub$y_sc[j] <- c(bindcor.df$mean_sc_zsub[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j])],NA)[1]
    comp.df.sub$x_hr[j] <- c(bindcor.df$mean_hr_zsub[which(bindcor.df$id == i & bindcor.df$exam == comp.df$x[j])],NA)[1]
    comp.df.sub$y_hr[j] <- c(bindcor.df$mean_hr_zsub[which(bindcor.df$id == i & bindcor.df$exam == comp.df$y[j])],NA)[1]
  }
  comp.df.full <- rbind(comp.df.full, comp.df.sub)
}

# compute distance metrics for behavioral data
comp.df.full$grade_distance <- abs(comp.df.full$x_grade - comp.df.full$y_grade)
comp.df.full$pe_distance <- abs(comp.df.full$x_pe - comp.df.full$y_pe)
comp.df.full$exam_distance <- abs(comp.df.full$x - comp.df.full$y)
comp.df.full$pe_sameSign <- as.numeric(comp.df.full$x_pe*comp.df.full$y_pe > 0)
comp.df.full$pe_sameSign[which(comp.df.full$pe_sameSign == 0)] <- -1
comp.df.full$sc_distance <- abs(comp.df.full$x_sc - comp.df.full$y_sc)
comp.df.full$pe_sameSign <- as.factor(comp.df.full$pe_sameSign)
comp.df.full$t_distance <- comp.df.full$y - comp.df.full$x
comp.df.full$abs_pe_distance <- abs(abs(comp.df.full$x_pe) - abs(comp.df.full$y_pe))



# additional processing
eda_buffer = 1 # 2 second delay in EDA response
ct <- 1
for (i in unique(crosscorr.df$id)) {
  for (j in unique(crosscorr.df$exam[which(crosscorr.df$id == i)])) {
    
    g <- crosscorr.df$grade[which(crosscorr.df$id == i & crosscorr.df$exam == j)][1]
    p <- crosscorr.df$pred[which(crosscorr.df$id == i & crosscorr.df$exam == j)][1]
    pe <- crosscorr.df$grade[which(crosscorr.df$id == i & crosscorr.df$exam == j)][1] - crosscorr.df$pred[which(crosscorr.df$id == i & crosscorr.df$exam == j)][1]
    
    paste.df <- data.frame(id = i, exam_orig = j, grade = g, pred = p, pe = pe, 
               exam_comp = crosscorr.df$exam[which(crosscorr.df$id == i & 
                                                     !is.na(crosscorr.df[[paste0("exam_",j,"_corr")]]))],
               index = crosscorr.df$event_index[which(crosscorr.df$id == i & 
                                                        !is.na(crosscorr.df[[paste0("exam_",j,"_corr")]]))],
               start = crosscorr.df$start[which(crosscorr.df$id == i & 
                                                  !is.na(crosscorr.df[[paste0("exam_",j,"_corr")]]))],
               end = crosscorr.df$end[which(crosscorr.df$id == i & 
                                              !is.na(crosscorr.df[[paste0("exam_",j,"_corr")]]))],
               corr = crosscorr.df[which(crosscorr.df$id == i & 
                                           !is.na(crosscorr.df[[paste0("exam_",j,"_corr")]])),][[paste0("exam_",j,"_corr")]],
               curr_sc_zsub = c(physio.cat$mean_sc_zsub[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num == 295)],NA)[1],
               curr_hr_zsub = c(physio.cat$mean_hr_zsub[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num == 295)],NA)[1],
               curr_sc_deriv = c(physio.cat$mean_sc_zsub_deriv[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num == 295 + eda_buffer)],NA)[1],
               curr_hr_deriv = c(physio.cat$mean_hr_zsub_deriv[which(physio.cat$id == i & physio.cat$exam == j & physio.cat$segment_num == 295)],NA)[1])
    
    paste.df <- paste.df[2:nrow(paste.df),]
    
    if (ct == 1) {
      crosscorr.df.m <- paste.df
    } else {
      crosscorr.df.m <- rbind(crosscorr.df.m, paste.df)
    }
    ct <- ct + 1
  }
}

# subset crosscorr.df.m to MAX CORR within exam comparison
crosscorr.df.oneper <- c()
for (i in unique(crosscorr.df.m$id)) {
  for (j in c(1,2,3)) {
    
    crosscorr.df.oneper <- rbind(crosscorr.df.oneper, 
                                 crosscorr.df.m[which(crosscorr.df.m$id == i & crosscorr.df.m$exam_orig == j & crosscorr.df.m$exam_comp > j),][which(crosscorr.df.m$corr[which(crosscorr.df.m$id == i & crosscorr.df.m$exam_orig == j & crosscorr.df.m$exam_comp > j)] == max(crosscorr.df.m$corr[which(crosscorr.df.m$id == i & crosscorr.df.m$exam_orig == j & crosscorr.df.m$exam_comp > j)])),])
  
  }
}

# flag first five events after grade reveal
event_sim.df.dc$state_flag <- 0
# i <- unique(event_sim.df.dc$id)[1]
for (i in unique(event_sim.df.dc$id)) {
  # j <- 1
  for (j in unique(event_sim.df.dc$exam[which(event_sim.df.dc$id == i)])) {
    
    event_sim.df.dc$state_flag[which(event_sim.df.dc$id == i & event_sim.df.dc$exam == j & event_sim.df.dc$start > 295)[1:5]] <- 1
  }
}

# subset df to first five events
event_sim.df.dc.pr <- event_sim.df.dc[which(event_sim.df.dc$state_flag == 1),]

index <- c(1,2,3,4,5,6)
comp_1 <- c(1,1,1,2,2,3)
comp_2 <- c(2,3,4,3,4,4)
comp.df <- data.frame(index, comp_1, comp_2)


## grade-based similarity ####
diffVar <- "grade"
ct <- 1
for (i in unique(event_sim.df.dc.pr$id)) {
  
  for (j in 1:nrow(comp.df)) {
    
    if (nrow(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j]),9:408]) == 0 | nrow(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j]),9:408]) == 0) {
      next
    } 
    
    cormat <- cor(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j]),9:408], event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j]),9:408])

    # try absolute value
    diffScalar <- abs(event_sim.df.dc.pr[[diffVar]][which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j])][1] - event_sim.df.dc.pr[[diffVar]][which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j])][1])
    
    cormat*diffScalar
    
    if (ct == 1) {
      new.df <- cormat*diffScalar
    } else {
      new.df <- rbind(new.df, cormat*diffScalar)
    }
    ct <- ct + 1
  }
}

new.df.df <- as.data.frame(new.df)
hist(c(new.df), breaks = 4000)
new.df.m <- melt(new.df.df)
unique(new.df.m$variable[which(new.df.m$value > as.numeric(quantile(new.df.m$value, probs = 0.999999, na.rm = T)))])


## PE-based similarity ####
range(event_sim.df.dc.pr$pe, na.rm = T) # fuzzy range: -45 : 45

event_sim.df.dc.pr$pe_scaled <- event_sim.df.dc.pr$pe + 45


diffVar <- "pe_scaled"
ct <- 1
for (i in unique(event_sim.df.dc.pr$id)) {
  
  for (j in 1:nrow(comp.df)) {
    
    if (nrow(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j]),9:408]) == 0 | nrow(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j]),9:408]) == 0) {
      next
    } 
    
    cormat <- cor(event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j]),9:408], event_sim.df.dc.pr[which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j]),9:408])
    
    # try absolute value
    diffScalar <- abs(event_sim.df.dc.pr[[diffVar]][which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_1[j])][1] - event_sim.df.dc.pr[[diffVar]][which(event_sim.df.dc.pr$id == i & event_sim.df.dc.pr$exam == comp.df$comp_2[j])][1])
    
    cormat*diffScalar
    
    if (ct == 1) {
      new.df <- cormat*diffScalar
    } else {
      new.df <- rbind(new.df, cormat*diffScalar)
    }
    ct <- ct + 1
  }
}

new.df.df <- as.data.frame(new.df)
hist(c(new.df), breaks = 4000)
new.df.m <- melt(new.df.df)
unique(new.df.m$variable[which(new.df.m$value > as.numeric(quantile(new.df.m$value, probs = 0.999999, na.rm = T)))])