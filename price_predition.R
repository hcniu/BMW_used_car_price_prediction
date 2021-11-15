library(tidyverse)
library(caret)
library(sjmisc)
library(party)
library(MASS)
library(e1071)
library(MASS)
library(agricolae)
library(randomForest)
library(car)
###1. Import and clean data
setwd("~/Desktop/UMN/Intro to Stat/Project")
df<-read.csv("bmw.csv")
df$model<-as.factor(df$model)
df$transmission<-as.factor(df$transmission)
df$fuelType<-as.factor(df$fuelType)
df$model<-relevel(df$model,ref = " 3 Series")
df$fuelType<-relevel(df$fuelType,ref = "Petrol")
unique(df$transmission)
#Eliminate the error row
df<-df%>%filter(price<max(df$price))

###2. Data Exploration:
ggplot(data = df,aes(x=fct_reorder(model,price,mean),y=price))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle=90))+
  labs(x="Model",y="Price",title = "Price among different models")

ggplot(data=df,aes(y=price))+
  geom_boxplot()

ggplot(data = df,aes(x=fct_reorder(fuelType,price,mean),y=price))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle=0))+
  labs(x="Fuel Types",y="Price",title = "Price among different fuel types")

tem<-df%>%group_by(model)%>%summarise("Average Price"=mean(price))
tem$Cartype<-"None"
for(i in 1:nrow(tem)){
  if(str_detect(tem$model[i],"Series")){
    tem$Cartype[i]<-"Sedan"
  } else if(str_detect(tem$model[i],"M")){
    tem$Cartype[i]<-"Sport"
  } else if(str_detect(tem$model[i],"X")){
    tem$Cartype[i]<-"SUV"
  } else if(str_detect(tem$model[i],"Z")){
    tem$Cartype[i]<-"Convertible"
  } else{
    tem$Cartype[i]<-"Electric Car"
  }
}
ggplot(data=tem,aes(x=reorder(model,`Average Price`),y=`Average Price`,fill=Cartype))+
  geom_bar(stat="identity")+
  theme_bw()+
  theme(axis.text.x = element_text(angle=60,hjust = 1))+
  labs(x=NULL,fill="Car Type",title = "Average Price of different car model and type")

ggplot(data=df,aes(x=price,y=mileage,color=model))+
  geom_smooth(method = "glm",se=F)+
  theme_bw()+
  labs(y="Mileage",x="Price",title = "Mileage Effect across different Models")+
  theme(axis.text.x = element_blank())+
  coord_flip()

tem<-df%>%group_by(year)%>%summarise("Average Price"=mean(price))
ggplot(data = tem,aes(x=2020-year,y=`Average Price`,fill=`Average Price`))+
  geom_bar(stat = "identity")+
  labs(x="Age of the car",title = "Average Price over Car Age")+
  theme_bw()+
  theme(legend.position = "none")

###3. Data Spliting
set.seed(100)
train_row<-createDataPartition(df$price,p=0.8,list=FALSE)
df_train<-df[train_row,]
df_all<-df[-train_row,]
validation_row<-createDataPartition(df_all$price,p=0.5,list=FALSE)
df_validation<-df_all[validation_row,]
df_test<-df_all[-validation_row,]


###4. Multiple Linear Regression
#Use stepwise regression with AIC as performance measurement to find the best model
model_lm<-lm(price~.,data=df_train)
summary(model_lm)
result<-stepAIC(model_lm,direction = "both")
summary(result)
predict_lm<-predict(model_lm,df_validation)
RMSE(predict_lm,df_validation$price)
df_validation$pred_lm<-predict_lm
#Eventually we found out the full model is the best model, with Multiple R-squared = 0.8807
#Assumptions check 1: Constant Variance
ggplot(data=df_train,aes(x=result$fitted.values,y=result$residuals))+
  geom_point()+
  geom_hline(yintercept = 0,color="red",alpha=0.6,linetype=2)+
  labs(x="Fitted Value",y="Residuals",title = "Residual Plot")
#Apparently there is some kind of pattern within residuals, hence this assumption is not satisfied
#Assumptions check 2: Residual Normal Distribution
ggplot(data=df_train,aes(sample=result$residuals))+
  geom_qq(color="blue",alpha=0.4)+
  geom_qq_line(linetype=2)+
  theme_bw()+
  labs(title = "QQ-plot")

###5. Change Y to log(Y) to fix the assumption problem
df_train$log10_price<-log10(df_train$price)
model_lm_log10<-lm(log10_price~.+model:year,data=dplyr::select(df_train,-price))
summary(model_lm_log10)
step_model<-stepAIC(model_lm_log10,direction = "both")
summary(step_model)
predict_lm_log10<-predict(step_model,df_validation)
RMSE(10^predict_lm_log10,df_validation$price)

#Assumptions check 1: Constant Variance
ggplot(data=df_train,aes(x=step_model$fitted.values,y=rstandard(step_model)))+
  geom_point()+
  geom_hline(yintercept = 0,color="red",alpha=0.6,linetype=2)+
  labs(x="Fitted Value",y="Standardize Residuals",title = "Residual Plot")+
  theme_bw()
#Assumptions check 2: Normality
ggplot(data=df_train,aes(sample=step_model$residuals))+
  geom_qq(color="blue",alpha=0.4)+
  geom_qq_line(linetype=2)+
  labs(title = "QQ-plot")+
  theme_bw()

ggplot(data = df_train,aes(x=step_model$residuals))+
  geom_histogram(bins = 60,fill="grey40")+
  geom_freqpoly(bins=60)+
  coord_cartesian(xlim = c(-0.25,0.25))+
  labs(x="Residual",y="Freq",title = "Residual Histogram")+
  theme_bw()

#Numeric Variable Collinearity Check
#Apparently Mileage and Year are highly correlated (corr >0.7), so we should drop one of them
cor(dplyr::select(df_train,-c(price,log10_price,model,transmission,fuelType)))
vif(step_model)
?vif
#Check Collinearity for model type
result<-anova(aov(mileage~model,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(year~model,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(tax~model,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(mpg~model,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(engineSize~model,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

#Check Collinearity for transmission
result<-anova(aov(mileage~transmission,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(year~transmission,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(tax~transmission,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(mpg~transmission,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(engineSize~transmission,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

#Check Collinearity for fuel type
result<-anova(aov(mileage~fuelType,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(year~fuelType,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(tax~fuelType,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(mpg~fuelType,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

result<-anova(aov(engineSize~fuelType,data = df_train))
R_square<-result["Sum Sq"][[1]][1]/sum(result["Sum Sq"][[1]])
R_square

###6. Multiple Linear regression model without Year
model_lm_log10_v2<-lm(log10_price~.+model:mileage-year,data=dplyr::select(df_train,-price))
summary(model_lm_log10_v2)
par(mfrow=c(2,2))
plot(model_lm_log10_v2)
predict_lm_log10_v2<-predict(model_lm_log10_v2,df_validation)
RMSE(10^predict_lm_log10_v2,df_validation$price)

#Assumptions check 1: Constant Variance
std_res<-rstandard(model_lm_log10_v2)
ggplot(data=df_train,aes(x=model_lm_log10_v2$fitted.values,y=std_res))+
  geom_point()+
  geom_hline(yintercept = 0,color="red",alpha=0.6,linetype=2)+
  labs(x="Fitted Value",y="Standardized Residuals",title = "Residual Plot")
#Assumptions check 2: Normality
ggplot(data=df_train,aes(sample=model_lm_log10_v2$residuals))+
  geom_qq(color="blue",alpha=0.4)+
  geom_qq_line(linetype=2)+
  theme_bw()+
  labs(title = "QQ-plot")

ggplot(data = df_train,aes(x=model_lm_log10_v2$residuals))+
  geom_histogram(bins = 60)+
  geom_freqpoly(bins=60)+
  labs(x="residual",y="Freq")


###7.Evaluation on Test data
predict_lm_log<-predict(model_lm_log10,df_test)
RMSE(10^predict_lm_log,df_test$price)

predict_lm_log_v2<-predict(model_lm_log10_v2,df_test)
RMSE(10^predict_lm_log_v2,df_test$price)
###8. Conditional Tree Model
model_ct<-ctree(price~.,data = dplyr::select(df_train,-log10_price),control = ctree_control(maxdepth = 4))
summary(model_ct)
model_ct

pred_ct<-predict(model_ct,df_validation)
df_validation$pred_ct<-pred_ct
RMSE(df_validation$pred_ct,df_validation$price)
#Evaluation on Test data
predict_ct<-predict(model_ct,df_test)
RMSE(predict_ct,df_test$price)

###10. Conclusion: Eventually, considering both complexity and accuracy, I choose the final multiple linear regression model as final model
