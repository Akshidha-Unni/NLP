#read the dataset
data = read.csv("C:\\Users\\akshi\\Downloads\\archive (1)\\chatgpt.csv",stringsAsFactors = FALSE)

head(data)

#delete unwanted columns
data = data[,-1]
data = data[1:6000,]

options(warn=-1)

#install text mining packages
install.packages('tm')

library(tm)

#create a Vcorpus object 
corpus = VCorpus(VectorSource(data$tweets))

#cleaning the text
#convert to lowercase
corpus = tm_map(corpus,content_transformer(tolower))

as.character(corpus[[5]])

#removing links from text
remove_links = function(text) {
  cleaned_text = gsub("(http|https)://[[:alnum:]/\\.]+", "", text)
  return(cleaned_text)
}

for (i in 1:length(corpus)) {
  corpus[[i]] <- content_transformer(remove_links)(corpus[[i]])
}


#removing any special characters 
remove_special_chars <- function(text) {
  cleaned_text <- gsub("[^a-z ]+", "", text)
  return(cleaned_text)
  
}
for (i in 1:length(corpus)) {
  corpus[[i]] <- content_transformer(remove_special_chars)(corpus[[i]])
}

#remove stopwords and perform stemming
corpus = tm_map(corpus,removeWords,stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace)

as.character(corpus[[5]])

#creating the bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)

dtm

#convert to dataframe
dataset = as.data.frame(as.matrix(dtm))

#barplot
w = colSums(dataset)
w = subset(w, w>=200)
barplot(w, 
        las = 2,
        col = rainbow(50),
        main = "Frequent Words - Bar plot")

install.packages("wordcloud", repos='http://cran.us.r-project.org')

library(wordcloud)

#word cloud 
w = sort(colSums(dataset),decreasing = TRUE)
wordcloud(words = names(w),
          freq = w,
          random.order = FALSE,
          colors = brewer.pal(8,"Dark2"),
          max.words = 100,
          rot.per = 0,
          main = "Word Cloud")

#pie chart
freq = table(data$labels)
#freq = rowSums(data$labels)
pie(freq, labels = names(freq), col = c("skyblue","pink","yellow"),main = "Sentiment Distribution")

#finding associations between three most frequent words
associations = findAssocs(dtm, terms = c("chatgpt","openai","ask"), corlimit = 0.1)
print(associations)

#associations between higher frequency terms
asso = findAssocs(dtm, terms = findFreqTerms(dtm, lowfreq = 300), corlimit = 0.1)
print(asso)

#encoding the labels
data$labels = factor(data$labels,
                     levels = c('neutral','good','bad'),
                     labels = c(1,2,3))
dataset$labels = data$labels

install.packages("caTools",repos='http://cran.us.r-project.org')

library(caTools)

#splitting the dataset into train and test
set.seed(223)
split = sample.split(dataset$labels)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)


dimen = dim(train)
dimen

library(class)

y_pred = knn(train = train[-1308],
             test = test[-1308],
             cl = train$labels,
             k = 3,
             prob = TRUE)

cn_mx = table(test$labels,y_pred)

cn_mx

accuracy = sum(y_pred == test$labels) / length(test$labels)
accuracy

library(e1071)

classifier = svm(formula = labels ~ .,
                 data = train,
                 type = 'C-classification',
                 kernel = 'linear')

pred = predict(classifier, newdata = test[-1308])

mx = table(test$labels,pred)
mx

accuracy = sum(pred == test$labels) / length(test$labels)
accuracy

library(rpart)
library(rpart.plot)

tree = rpart(labels ~ .,data = train, method = "class")
prp(tree)
pred = predict(tree, newdata = test[-1308], type = "class")

mx = table(test$labels,pred)
mx

accuracy = sum(pred == test$labels) / length(test$labels)
accuracy
