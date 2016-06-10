multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


library(dplyr)
library(ggplot2)
library(cowplot)
proc_num = 8
header = c("activity", "sparse_mode", "num_processes", "repl_fact", "exponent", "dim", "cells")
header = c(header, paste0("p", as.character(1:proc_num)))
res = read.csv(file("mw336071/result.csv"), sep="\t", header = F)
res = res[,-ncol(res)]
colnames(res) = header

mean = apply(res[,8:ncol(res)], 1, mean)
sd = apply(res[,8:ncol(res)], 1, sd)

stat = cbind(res[,1:7],mean,sd)

stat = stat %>% select(-num_processes, -exponent)

p = stat %>% group_by(dim,cells,activity) %>% summarise(means=list(mean), sds=list(sd), c=list(repl_fact), modes=list(sparse_mode))
library('scales')
plots = c()
for (i in 1:nrow(p)) {
  data = p[i,]
  dim = data[["dim"]]
  cells = data[["cells"]]
  if (data[["activity"]] == "comp")
    activ = "obliczenia"
  else
    activ = "początkowa komunikacja"
  title = paste(activ, "\ndim", dim, "cells", cells, "dens", percent(cells/dim^2))
  c = data$c[[1]]
  sds = data$sds[[1]]
  means = data$means[[1]]
  mode = data$modes[[1]]
  df = data.frame(mode = factor(mode),c = factor(c),mean = means, sd=sds)
  class(df)
  g <- ggplot(df, aes(c, fill=mode,weight=mean)) + geom_bar(position="dodge") + ylab("mean time [s]") + ggtitle(title) + xlab("replication factor")
  plots = c(plots, list(g))
}
plot_grid(plotlist = plots, ncol = 2)

####################

bench = read.csv(file("mw336071/benchmarki MPI - Arkusz2.csv"), header=F, dec = ",")
bench[,"V5"] = as.factor(bench[,"V5"])
means = apply(bench[,1:4], 1, mean)
bench = bench %>% select( -c(1:4) )
colnames(bench) = c("proc", "method") 
bench = cbind( bench, means )

bench = bench %>% group_by(proc, method) %>% summarise(m = mean(means))
df = bench
ggplot(df, aes(proc, fill=method, weight=m)) + geom_bar(position="dodge") + ylab("mean time [s]") + xlab("proc number")

