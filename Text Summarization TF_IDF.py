# -*- coding:utf-8 -*-

import math
from bnlp import NLTKTokenizer  # Importing Bangali Language Processing Tokenizer

bnltk = NLTKTokenizer()     # Initializing Bangali Language Processing Tokenizer

# Reading The Main Text
with open('MainText.txt', encoding='utf-8') as f:
    text = f.read()
    f.close()
    
# Tokenizing Words from text "আমি", "ভাত", "খাই"
word_tokens = bnltk.word_tokenize(text) 

# Getting the StopWords and Punctuations from BNLP Library
from bnlp.corpus import stopwords, punctuations
stopwords = stopwords()
punctuations = punctuations + "‘" + "’" + "``" + "''"

# Calculating Word Frequency of the Main Text
wordfreq = {}
for word in word_tokens:
    if word not in stopwords:
        if word not in punctuations:
            if word not in wordfreq.keys():
                wordfreq[word] = 1
            else:
                wordfreq[word] +=1

lenghtofTerms = len(wordfreq)

# Getting the Maximum Value of Frequency of Word
max_frequency = max(wordfreq.values()) 

# Normalizing the word frequency
Norm_wordfreq = {}
for word in wordfreq.keys():
    Norm_wordfreq[word] = wordfreq[word]/max_frequency   

# Calculating Term Frequency of the Main Text
TF = {}
for word in wordfreq.keys():
    TF[word] = wordfreq[word] / lenghtofTerms

# Tokenizing the Sentence from the Main Text
sentence_tokens = bnltk.sentence_tokenize(text)

Total_Sentence = len(sentence_tokens)

# How many sentences contain a word
Word_Appered_In_How_Many_sentences = {} 

for word in TF.keys():
    for sent in sentence_tokens:
        sent_word_tokens = bnltk.word_tokenize(sent) # Tokenizing the Words of a Specific Sentence 
        for token in sent_word_tokens:
            if (token == word):
                if word not in Word_Appered_In_How_Many_sentences.keys():
                    Word_Appered_In_How_Many_sentences[word] = 1
                    break
                else:
                    Word_Appered_In_How_Many_sentences[word] += 1
                    break


# Calculating Inverse Document Frequency of the Main Text
IDF = {}
for word in Word_Appered_In_How_Many_sentences:
    IDF[word] = math.log(Total_Sentence/Word_Appered_In_How_Many_sentences[word])


# Calculating TF-IDF 
TF_IDF = {}
for word in TF:
    if word in IDF.keys():
        TF_IDF[word] = TF[word] * IDF[word]


# Scoring Sentences with the TF_IDF Values of WORDS
sentence_score = {} 

for sent in sentence_tokens:
    sent_word_tokens = bnltk.word_tokenize(sent) # Tokenizing the Words of a Specific Sentence 
    for word in sent_word_tokens:                
        if word in TF_IDF.keys(): 
            if sent not in sentence_score.keys():
                sentence_score[sent] = TF_IDF[word]
            else:
                sentence_score[sent] += TF_IDF[word]


# Making a Ranking (High to low) according to the Sentence Scores
from heapq import nlargest
length_thirty = int(len(sentence_score)*0.2)
length_fourty = int(len(sentence_score)*0.3)
length_fifty = int(len(sentence_score)*0.4)        # 0.3 = 30% of Main Text , 0.4 =40% of Main Text , 0.5 = 50% of Main Text 

# Making a Summary from the Ranking
summary20 = nlargest(length_thirty, sentence_score, key = sentence_score.get)
summary30 = nlargest(length_fourty, sentence_score, key = sentence_score.get)
summary40 = nlargest(length_fifty, sentence_score, key = sentence_score.get)
#print(summary)


# Converting List Type Summary to String Type Summary
Summary20 = ' '.join(map(str,summary20))
Summary30 = ' '.join(map(str,summary30))
Summary40 = ' '.join(map(str,summary40))

DocumentNumber = 77

# Writing the summary in a Text File
with open('summary_'+str(DocumentNumber)+'_20%.txt', 'w', encoding='utf-8') as f:
    f.write(Summary20)
with open('summary_'+str(DocumentNumber)+'_30%.txt', 'w', encoding='utf-8') as f:
    f.write(Summary30)
with open('summary_'+str(DocumentNumber)+'_40%.txt', 'w', encoding='utf-8') as f:
    f.write(Summary40)





#----------------- Similarity Check -----------------------------


# Jaccard Similarity
#a = set(text.split())
#b = set(SummaryString.split())
#c = a.intersection(b)

#jaccard_similarity = float(len(c)) / (len(a) + len(b) - len(c))

#print(jaccard_similarity*100)



# Cosine Similarity


# Reading The Summary Text
with open('Summary_'+str(DocumentNumber)+'_20%.txt', encoding='utf-8') as f:
    text = f.read()
    f.close()
    
# Tokenizing Words from text "আমি", "ভাত", "খাই"
word_tokens_summary = bnltk.word_tokenize(text)


# Calculating Word Frequency of the Summary Text
wordfreqSummary = {}
for word in word_tokens_summary:
    if word not in stopwords:
        if word not in punctuations:
            if word not in wordfreqSummary.keys():
                wordfreqSummary[word] = 1
            else:
                wordfreqSummary[word] +=1

# Getting the Maximum Value of Frequency of Word
max_frequency_Summary = max(wordfreqSummary.values()) 

# Normalizing the word frequency
Norm_Summarywordfreq = {}
for word in wordfreqSummary.keys():
    Norm_Summarywordfreq[word] = wordfreqSummary[word]/max_frequency_Summary   

MAIN_MULTI_SUMMARY = 0

for word in wordfreqSummary:
    if word in wordfreq.keys():
        MAIN_MULTI_SUMMARY += (wordfreq[word] * wordfreqSummary[word])



MAIN_SUM = 0
for word in wordfreq:
    MAIN_SUM += wordfreq[word] * wordfreq[word]

MAIN_ROOT = math.sqrt(MAIN_SUM)


Summary_SUM = 0
for word in wordfreqSummary:
    Summary_SUM += (wordfreqSummary[word] * wordfreqSummary[word])

Summary_ROOT = math.sqrt(Summary_SUM)


Cossine_Similarity = (MAIN_MULTI_SUMMARY / (MAIN_ROOT * Summary_ROOT))*100
print(Cossine_Similarity)



#------------------------------------------------------- 


# Cosine Similarity


# Reading The Summary Text
with open('Summary_'+str(DocumentNumber)+'_30%.txt', encoding='utf-8') as f:
    text = f.read()
    f.close()
    
# Tokenizing Words from text "আমি", "ভাত", "খাই"
word_tokens_summary = bnltk.word_tokenize(text)


# Calculating Word Frequency of the Summary Text
wordfreqSummary = {}
for word in word_tokens_summary:
    if word not in stopwords:
        if word not in punctuations:
            if word not in wordfreqSummary.keys():
                wordfreqSummary[word] = 1
            else:
                wordfreqSummary[word] +=1

# Getting the Maximum Value of Frequency of Word
max_frequency_Summary = max(wordfreqSummary.values()) 

# Normalizing the word frequency
Norm_Summarywordfreq = {}
for word in wordfreqSummary.keys():
    Norm_Summarywordfreq[word] = wordfreqSummary[word]/max_frequency_Summary   

MAIN_MULTI_SUMMARY = 0

for word in wordfreqSummary:
    if word in wordfreq.keys():
        MAIN_MULTI_SUMMARY += (wordfreq[word] * wordfreqSummary[word])



MAIN_SUM = 0
for word in wordfreq:
    MAIN_SUM += wordfreq[word] * wordfreq[word]

MAIN_ROOT = math.sqrt(MAIN_SUM)


Summary_SUM = 0
for word in wordfreqSummary:
    Summary_SUM += (wordfreqSummary[word] * wordfreqSummary[word])

Summary_ROOT = math.sqrt(Summary_SUM)


Cossine_Similarity = (MAIN_MULTI_SUMMARY / (MAIN_ROOT * Summary_ROOT))*100
print(Cossine_Similarity)

#----------------------------------------------------------------------------------