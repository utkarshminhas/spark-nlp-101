#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp


# In[2]:


spark = sparknlp.start()
print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)


# # using pretrained model 

# In[3]:


# from sparknlp.pretrained import PretrainedPipeline
# pretrained_pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')


# In[4]:


# text = "The Mona Lisa is a 16th century oil painting created by Leonardo. It's held at the Louvre in Paris."

# result = pretrained_pipeline.annotate(text)
# list(zip(result['token'], result['ner']))


# # Training a model(approach)

# In[5]:


# from urllib.request import urlretrieve
# urlretrieve('https://github.com/JohnSnowLabs/spark-nlp/raw/master/src/test/resources/conll2003/eng.train',
#            'dataset/eng.train')
# urlretrieve('https://github.com/JohnSnowLabs/spark-nlp/raw/master/src/test/resources/conll2003/eng.testa',
#            'dataset/eng.test')


# In[6]:


from sparknlp.training import CoNLL
training_data = CoNLL().readDataset(spark, './dataset/eng.train')
training_data.show(5)


# In[7]:


test_data = CoNLL().readDataset(spark, './dataset/eng.test')
# test_data.show(3)


# In[8]:


bert_annotator = BertEmbeddings.pretrained('bert_base_cased', 'en')  .setInputCols(["sentence",'token']) .setOutputCol("bert") .setCaseSensitive(False)

bert_annotator.transform(test_data.limit(limit))
test_data.show(3)


# In[10]:


# test_data.select("bert.result","bert.embeddings",'label.result').show()


# In[11]:

limit = 500
test_data.limit(limit).write.parquet("dataset/test_withEmbeds.parquet")


# In[15]:


nerTagger = NerDLApproach()  .setInputCols(["sentence", "token", "bert"])  .setLabelColumn("label")  .setOutputCol("ner")  .setMaxEpochs(1)  .setLr(0.001)  .setPo(0.005)  .setBatchSize(8)  .setRandomSeed(0)  .setVerbose(1)  .setValidationSplit(0.2)  .setEvaluationLogExtended(True)   .setEnableOutputLogs(True)  .setIncludeConfidence(True)  .setTestDataset("dataset/test_withEmbeds.parquet")


# In[16]:


pipeline = Pipeline(
    stages = [
    bert_annotator,
    nerTagger
  ])


# In[17]:


ner_model = pipeline.fit(training_data.limit(limit))


# In[ ]:


predictions = ner_model.transform(test_data)


# In[ ]:


predictions.select('token.result','label.result','ner.result').show(truncate=40)


# In[ ]:


import pyspark.sql.functions as F
predictions.select(F.explode(F.arrays_zip('token.result','label.result','ner.result')).alias("cols")) .select(F.expr("cols['0']").alias("token"),
        F.expr("cols['1']").alias("ground_truth"),
        F.expr("cols['2']").alias("prediction")).show(truncate=False)

