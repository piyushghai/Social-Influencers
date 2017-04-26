Winedata = read.csv("mod.csv")
library(ggplot2)
#install.packages("gridExtra")
library (gridExtra)
#install.packages("plyr")
library(plyr)
library(corrplot)


head(Winedata)

summary(Winedata)

str(Winedata)



#sapply(2:12, function(x) sd(Winedata[,x]))

correlationMatrix=cor(Winedata, use="complete.obs", method="pearson")
corrplot(correlationMatrix, method = 'number', tl.cex = 0.5)
