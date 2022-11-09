library(png)
n <- 289 ## number of small data
for (i in 0:n){
  dat_path <- paste0("/Users/abnerteng/Desktop/stock/split_data_10032/10032_split",i,".csv")
  d <- read.csv(dat_path)

  plot_ohlc = function(d,filename=paste0("/Users/abnerteng/Desktop/stock_fig/10032/plot_10032-",i,".png")) {
    png(filename,width=60,height=48+(48/4)) # +1 pixel for the line that separates price and volume charts
    layout(matrix(1:2,2,1),
          heights=c(4,1))    # the ratio between price and volume chart is 48:12
    par(mar=c(0,0,0,0),
        bg="black",
        fg="white",
        lwd=1)
    plot.new()
    plot.window(xlim=c(0,59),   # x starts from zero
                ylim=c(min(d$Low),max(d$High)),
                xaxs='i' ## internal
    )
    x.pos = seq(1,58,3)   # day 1 is x=1, and leave one pixel each at left and right (x=0,x=2)
    # draw vertical line High-Low
    segments(x.pos,d$Low,x.pos,d$High)
    # draw CLOSE notch
    segments(x.pos,d$Close,x.pos+1,d$Close)
    # draw OPEN notch
    segments(x.pos-1,d$Open,x.pos,d$Open)
    # draw MA line
    segments(x.pos-1, d$MA, x.pos+1, d$MA)
    # add Volume
    plot.new()
    plot.window(xlim=c(0,59),
                ylim=c(min(d$Volume),max(d$Volume)),
                xaxs='i'
    )
    segments(x.pos,0,x.pos,d$Volume)
  }
  plot_ohlc(d)
  dev.off()

  a = readPNG(paste0("/Users/abnerteng/Desktop/stock_fig/10032/plot_10032-",i,".png"))
  print(a)
  View(a[,,1])
}
