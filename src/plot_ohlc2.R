#library(quantmod)
#getSymbols("IBM", src="yahoo")

library(data.table)


IBM = readRDS("IBM.RDS")
df = as.data.table(IBM[,1:5])[,-1]
setnames(df,1:5,c("Open","High","Low","Close","Volume"))

group_id = sort(rep(1:(nrow(df)/20),20))
df =df[1:length(group_id)]
df[,group_id := group_id]


plot_ohlc = function(d,filename="plot01") {
png(paste0(filename,".png"),width=60,height=60) # +1 pixel for the line that separates price and volume charts
layout(matrix(1:2,2,1),
       heights=c(4,1))    # the ratio between price and volume chart is 60:15
par(mar=c(0,0,0,0),
    bg="black",
    fg="white",
    lwd=1)
plot.new()
plot.window(xlim=c(0,59),   # x starts from zero
            ylim=c(min(d[,Low]),max(d[,High])),
            xaxs='i'
            )
x.pos = seq(1,58,3)   # day 1 is x=1, and leave one pixel each at left and right (x=0,x=2)
# draw vertical line High-Low
segments(x.pos,d[,Low],x.pos,d[,High])
# draw CLOSE notch
segments(x.pos,d[,Close],x.pos+1,d[,Close])
# draw OPEN notch
segments(x.pos-1,d[,Open],x.pos,d[,Open])
# add Volume
plot.new()
plot.window(xlim=c(0,59),
            ylim=range(d[,Volume]),
            xaxs='i'
            )
segments(x.pos,0,x.pos,d[,Volume])
dev.off()
}

df[,plot_ohlc(.SD,filename=group_id),
   by=group_id]





